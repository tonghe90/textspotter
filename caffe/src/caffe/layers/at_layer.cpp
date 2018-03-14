#include "caffe/layers/at_layer.hpp"

namespace caffe {

template <typename Dtype>
inline int AffineTransformerLayer<Dtype>::calcT(int length, int cont_index) {
	for (int i = 0; i < scale_shift[cont_index].size(); ++i) {
		length /= scale_shift[cont_index][i].first;
		length += scale_shift[cont_index][i].second;
	}
	return length;
}

template <typename Dtype>
void AffineTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
	const auto &param = this->layer_param_.at_param();
	output_H_ = param.output_h();
	LOG(INFO) << "output_H_ = " << output_H_;
	if (param.has_output_w()) {
		output_W_ = param.output_w();
		const_W_ = true;
		LOG(INFO) << "output_W_ = " << output_W_;
	} else const_W_ = false;
	CHECK_EQ(param.cont_size(), top.size() - 1);
	if (top.size() > 1) {
		scale_shift.resize(param.cont_size());
		for (size_t i = 0; i < scale_shift.size(); ++i) {
			int op_num = param.cont(i).scale_size();
			CHECK_EQ(op_num, param.cont(i).shift_size());
			CHECK_GE(op_num, 1);
			scale_shift[i].resize(op_num);
			for (int j = 0; j < op_num; ++j) {
				scale_shift[i][j].first = param.cont(i).scale(j);
				scale_shift[i][j].second = param.cont(i).shift(j);
			}
		}
	}
	pre_defined_count = 0;
	for (size_t i = 0; i < 6; ++i)
		is_pre_defined_theta[i] = false;
	if (param.theta_size()) {
		CHECK_EQ(param.theta_size(), 6);
		for (size_t i = 0; i < 6; ++i) {
			is_pre_defined_theta[i] = false;
			if (fabs(param.theta(i)) <= 1e4) {
				is_pre_defined_theta[i] = true;
				++pre_defined_count;
				pre_defined_theta[i] = param.theta(i);
				LOG(INFO) << "Getting pre-defined theta[" << i / 3 << "][" << i % 3 << "] = " << pre_defined_theta[i];
			}
		}
	}
}

template <typename Dtype>
void AffineTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
	// check the validation for the parameter theta
	CHECK_GE(bottom[1]->shape().size(), 3);
	CHECK_EQ(bottom[1]->shape(2), 6) << "The dimension of theta is not six!"
	                                 << bottom[1]->shape(2);
	nSrc = bottom[0]->shape(0);
	CHECK(bottom[1]->shape(0) == nSrc) << "The first dimension of theta and " <<
	                                   "U should be the same";
	nPerSrc = bottom[1]->shape(1);
	N = nSrc * nPerSrc;
	if (!const_W_) {
		CHECK_EQ(bottom.size(), 3);
		CHECK_EQ(bottom[2]->shape(0), nSrc);
		CHECK_EQ(bottom[2]->shape(1), nPerSrc);
		CHECK_EQ(bottom[2]->count(2), 1);
		output_W_ = output_H_;
		const Dtype * w = bottom[2]->cpu_data();
		for (size_t i = 0; i < bottom[2]->count(); ++i) {
			output_W_ = std::max(float(w[i]), float(output_W_));
		}
	}
	if (top.size() > 1) {
		for (size_t i = 1; i < top.size(); ++i) {
			int T = calcT(output_W_, i - 1);
			top[i]->Reshape(vector<int> {T, N});
		}
	}

	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);

	// reshape V
	top[0]->Reshape(vector<int> {N, C, output_H_, output_W_});

	// reshape dTheta_tmp
	dTheta_tmp.Reshape(vector<int> {N, 2, 3, C * output_H_, output_W_});

	// init all_ones_2
	all_ones_2.Reshape(vector<int> {output_H_ * output_W_ * C});

	if (pre_defined_count)
		full_theta.Reshape(vector<int> {N, 6});
}

#ifdef CPU_ONLY
STUB_GPU(AffineTransformerLayer);
#endif

INSTANTIATE_CLASS(AffineTransformerLayer);
REGISTER_LAYER_CLASS(AffineTransformer);

}  // namespace caffe
