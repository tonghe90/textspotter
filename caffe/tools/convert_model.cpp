#include <algorithm>
#include <vector>
#include <string>

#include <tr1/unordered_map>
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

/***************************************
******  remove bn layers  **************
******  author: Liang Ding  ************
******  email: liangding@sensetime.com *
******  date: 2017. 4. 6 ***************
***************************************/

using namespace caffe;
typedef std::tr1::unordered_map<std::string, std::string> umap;
struct index_name {
	int index, top_ref_count;
	std::string name;
	index_name (int i, std::string n, int t = 0) {
		index = i;
		name = n;
		top_ref_count = t;
	}
	index_name () {}
};
typedef std::tr1::unordered_map<string, index_name> bn_map;

bn_map bn_bottom2index;
bn_map bn_top2index;
std::tr1::unordered_map<string, vector<string*> > bn_replace;

#define var_eps_ float(1e-10)

typedef void (*integ_inner)(const float *scale_data, const float *shift_data,
                            float *w, float *b, int num_input, int num_output);
NetParameter remove_split(NetParameter &src_param);
void process_bn(float *scale_data, float *shift_data,
                const float *his_mean_data, const float *his_var_data, int _channels, float scale = 1.f);
void process_bn_only(float *his_mean_data, float *his_var_data, int _channels, float scale = 1.f);
bool integrate(bool is_forward, int blob_id,
               const string blob_name, NetParameter &src_net_param,
               LayerParameter *dst_layer, integ_inner fun);

void integrate_forward(const float *scale_data, const float *shift_data,
                       float *w, float *b, int num_input, int num_output) {
	caffe_cpu_gemm(CblasNoTrans, CblasNoTrans,
	               num_output, 1, num_input, float(1),
	               w, shift_data, float(1), b);
	for (int c = 0; c < num_output; ++c, w += num_input)
		caffe_mul(num_input, w, scale_data, w);
}

void integrate_backward(const float *scale_data, const float *shift_data,
                        float *w, float *b, int num_input, int num_output) {
	for (int c = 0; c < num_output; ++c)
		caffe_scal(num_input, scale_data[c], w + c * num_input);
	caffe_mul(num_output, b, scale_data, b);
	caffe_add(num_output, b, shift_data, b);
}

int main(int argc, char** argv) {
	if (argc != 3 && argc != 4) {
		LOG(ERROR) << "convert_model net_proto_file_in net_proto_file_out [--remove_param]";
		return 1;
	}
	bool rm_param = argc == 4;

	NetParameter src_net_param;
	ReadNetParamsFromBinaryFileOrDie(argv[1], &src_net_param);
	int _layers_size = src_net_param.layer_size();
	LOG(INFO) << "Total layers: " << _layers_size;
	src_net_param = remove_split(src_net_param);
	LOG(INFO) << "Recompute BN parameter...";
	_layers_size = src_net_param.layer_size();

	for (int i = 0; i < _layers_size; ++i) {
		LayerParameter * current_layer = src_net_param.mutable_layer(i);
		string layer_type = current_layer->type();
		std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::tolower);
		if (layer_type.find("bn") != string::npos) {
			CHECK_EQ(current_layer->bottom_size(), 1) << "BN layer is only allowed to have one bottom!";
			LOG(INFO) << "BN layer: " << current_layer->name();
			bn_bottom2index[current_layer->bottom(0)] = index_name(i, current_layer->top(0), 1);
			bn_top2index[current_layer->top(0)] = index_name(i, current_layer->bottom(0));
			float* scale_data = current_layer->mutable_blobs(0)->mutable_data()->mutable_data();
			float* shift_data = current_layer->mutable_blobs(1)->mutable_data()->mutable_data();
			const float* his_mean_data = current_layer->blobs(2).data().data();
			const float* his_var_data = current_layer->blobs(3).data().data();
			process_bn(scale_data, shift_data, his_mean_data,
			           his_var_data, current_layer->blobs(0).data_size());
		} else if (layer_type == "batchnorm") {
			CHECK_EQ(current_layer->bottom_size(), 1) << "BN layer is only allowed to have one bottom!";
			LOG(INFO) << "BN layer: " << current_layer->name();
			if (i + 1 < _layers_size && src_net_param.layer(i + 1).type() == "Scale") {
				LayerParameter * scale_layer = src_net_param.mutable_layer(i + 1);
				CHECK(scale_layer->blobs_size() == 2);
				bn_bottom2index[current_layer->bottom(0)] = index_name(i + 1, scale_layer->top(0), 1);
				bn_top2index[scale_layer->top(0)] = index_name(i + 1, current_layer->bottom(0));
				float* scale_data = scale_layer->mutable_blobs(0)->mutable_data()->mutable_data();
				float* shift_data = scale_layer->mutable_blobs(1)->mutable_data()->mutable_data();
				const float* his_mean_data = current_layer->blobs(0).data().data();
				const float* his_var_data = current_layer->blobs(1).data().data();
				process_bn(scale_data, shift_data, his_mean_data,
				           his_var_data, current_layer->blobs(0).data_size(), current_layer->blobs(2).data(0));
			} else {
				bn_bottom2index[current_layer->bottom(0)] = index_name(i, current_layer->top(0), 1);
				bn_top2index[current_layer->top(0)] = index_name(i, current_layer->bottom(0));
				float* his_mean_data = current_layer->mutable_blobs(0)->mutable_data()->mutable_data();
				float* his_var_data = current_layer->mutable_blobs(1)->mutable_data()->mutable_data();
				process_bn_only(his_mean_data, his_var_data,
				                current_layer->blobs(0).data_size(), current_layer->blobs(2).data(0));
			}
		} else if (layer_type != "dropout") {
			bool has_blobs = current_layer->blobs_size() > 0 || layer_type == "reverse";
			for (int j = 0; j < current_layer->bottom_size(); ++j) {
				string *name = current_layer->mutable_bottom(j);
				if (bn_top2index.find(*name) != bn_top2index.end()) {
					if (has_blobs) bn_top2index[*name].top_ref_count++;
					bn_replace[*name].push_back(name);
				}
			}
			for (int j = 0; j < current_layer->top_size(); ++j) {
				string *name = current_layer->mutable_top(j);
				if (bn_top2index.find(*name) != bn_top2index.end()) {
					bn_replace[*name].push_back(name);
				}
			}
		}
	}

	LOG(INFO) << "Integrate BN start...";
	NetParameter dst_net_param;
	vector<int> layer_needed;
	std::tr1::unordered_map<std::string, int> bottom_to_conv_index;
	for (int i = 0; i < _layers_size; ++i) {
		LayerParameter * dst_layer = src_net_param.mutable_layer(i);
		if ( dst_layer->bottom_size() == 0 ) continue;
		string layer_type = dst_layer->type();
		std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::tolower);
		if ( layer_type.find("bn") != string::npos || layer_type == "dropout" ||
		        layer_type == "batchnorm") continue;
		if (i > 1 && src_net_param.layer(i - 1).type() == "BatchNorm" && layer_type == "scale") continue;
		// LOG(INFO) << dst_layer->name() << " " << layer_type;
		if ( layer_type.find("convolution") != string::npos ) {
			integrate(false, 0, dst_layer->top(0), src_net_param, dst_layer, integrate_backward);
			bottom_to_conv_index[dst_layer->top(0)] = i;
		} else if ( layer_type == "slgrnn" || layer_type == "sllstm" ) {
			string blob_name;
			if (src_net_param.layer(i - 1).type() == "Reverse") {
				blob_name = src_net_param.layer(i - 1).bottom(1);
				if (bn_bottom2index.find(blob_name) != bn_bottom2index.end())
					blob_name = bn_bottom2index[blob_name].name;
			} else blob_name = dst_layer->bottom(0);
			integrate(true, 2, blob_name, src_net_param, dst_layer, integrate_forward);
		} else if ( layer_type.find("innerproduct") != string::npos ) {
			// interage bn before
			integrate(true, 0, dst_layer->bottom(0), src_net_param, dst_layer, integrate_forward);
			// interage bn after
			integrate(false, 0, dst_layer->top(0), src_net_param, dst_layer, integrate_backward);
		} else if ( layer_type == "eltwise" &&
		            dst_layer->eltwise_param().operation() == EltwiseParameter_EltwiseOp_SUM &&
		            i < _layers_size - 1 && (
		                src_net_param.layer(i + 1).type() == "BatchNorm" ||
		                src_net_param.layer(i + 1).type().find("BN") != string::npos)) {
			bn_bottom2index[dst_layer->top(0)].top_ref_count += dst_layer->bottom_size() - 1;
			for (size_t i = 0; i < dst_layer->bottom_size(); ++i) {
				// not consider coeff
				if (bottom_to_conv_index.find(dst_layer->bottom(i)) != bottom_to_conv_index.end()) {
					integrate(false, 0, dst_layer->top(0), src_net_param,
					          src_net_param.mutable_layer(bottom_to_conv_index[dst_layer->bottom(i)]),
					          integrate_backward);
				}
			}
		}
		if (rm_param)
			dst_layer->clear_param();
		layer_needed.push_back(i);
	}
	for (auto index : layer_needed)
		dst_net_param.add_layer()->CopyFrom(src_net_param.layer(index));
	if (bn_top2index.size() || bn_bottom2index.size()) {
		for (bn_map::iterator i = bn_top2index.begin(); i != bn_top2index.end(); ++i)
			LOG(ERROR) << src_net_param.layer(i->second.index).name() << " not merged";
		LOG(FATAL) << "convert failed!!!";
		return -1;
	}

	LOG(INFO) << "Snapshotting to " << argv[2];
	WriteProtoToBinaryFile(dst_net_param, argv[2]);
	return 0;
}

NetParameter remove_split(NetParameter &src_param) {
	NetParameter dst_layer;
	umap c_map;
	for (int i = 0; i < src_param.layer_size(); ++i) {
		LayerParameter *layer = src_param.mutable_layer(i);
		if (layer->type() == "Split") {
			LOG(INFO) << "layer " << layer->name() << " removed";
			for (int j = 0; j < layer->top_size(); ++j)
				c_map[layer->top(j)] = layer->bottom(0);
		} else {
			layer->clear_phase();
			for (int j = 0; j < layer->bottom_size(); ++j)
				if (c_map.find(layer->bottom(j)) != c_map.end())
					layer->set_bottom(j, c_map[layer->bottom(j)]);
			dst_layer.add_layer()->CopyFrom(*layer);
		}
	}
	return dst_layer;
}

void process_bn(float *scale_data, float *shift_data,
                const float *his_mean_data, const float *his_var_data, int _channels, float scale) {
	float * batch_statistic_ptr = new float[_channels];
	float * his_mean = new float[_channels];
	caffe_cpu_scale(_channels, 1.f / scale, his_mean_data, his_mean);
	/** compute statistic value: scale' = \gamma / \sqrt(Var(x) + \epsilon) **/
	caffe_cpu_scale(_channels, 1.f / scale, his_var_data, batch_statistic_ptr);
	// var(x) + \epsilon
	caffe_add_scalar(_channels, var_eps_, batch_statistic_ptr);
	// \sqrt(var(x) + \epsilon)
	caffe_powx(_channels, batch_statistic_ptr, 0.5f, batch_statistic_ptr);
	// \gamma / \sqrt(Var(x) + \epsilon)
	caffe_div(_channels, scale_data, batch_statistic_ptr, scale_data);

	/** compute statistic value: shift' = \beta - \mu * scale' **/
	caffe_mul(_channels, scale_data, his_mean, batch_statistic_ptr);

	// \beta - \mu * scale'
	caffe_sub(_channels, shift_data, batch_statistic_ptr, shift_data);
	delete batch_statistic_ptr;
	delete his_mean;
}

void process_bn_only(float *his_mean_data, float *his_var_data, int _channels, float scale) {
	float * batch_statistic_ptr = new float[_channels];
	float * his_mean = new float[_channels];
	caffe_cpu_scale(_channels, 1.f / scale, his_mean_data, his_mean);
	/** compute statistic value: scale' = 1 / \sqrt(Var(x) + \epsilon) **/
	float * scale_data = his_mean_data;
	float * shift_data = his_var_data;
	caffe_cpu_scale(_channels, 1.f / scale, his_var_data, batch_statistic_ptr);
	// var(x) + \epsilon
	caffe_add_scalar(_channels, var_eps_, his_var_data);
	// 1 / \sqrt(Var(x) + \epsilon)
	caffe_powx(_channels, his_var_data, -0.5f, scale_data);

	/** compute statistic value: shift' = - \mu * scale' **/
	caffe_mul(_channels, scale_data, batch_statistic_ptr, shift_data);
	caffe_scal(_channels, -1.f, shift_data);
	delete batch_statistic_ptr;
	delete his_mean;
}

bool integrate(bool is_forward, int blob_id,
               const string blob_name, NetParameter &src_net_param,
               LayerParameter *dst_layer, integ_inner fun) {
	bn_map &bn2index = is_forward ? bn_top2index : bn_bottom2index;
	bn_map &bn2index2 = !is_forward ? bn_top2index : bn_bottom2index;
	if ( bn2index.find(blob_name) == bn2index.end() ) {
		LOG(ERROR) << blob_name << " not found";
		return false;
	}
	const LayerParameter &src_param = src_net_param.layer(bn2index[blob_name].index);
	LOG(INFO) << "Integrate " << src_param.name() << " into " << dst_layer->name();
	int num_output = src_param.blobs(0).data_size();
	int num_input = dst_layer->blobs(blob_id).data_size() / num_output;
	if (is_forward) std::swap(num_output, num_input);
	float * wx = dst_layer->mutable_blobs(blob_id)->mutable_data()->mutable_data();
	// if bias_term == false
	if (blob_id == 0 &&
	        ( (dst_layer->has_convolution_param() && !dst_layer->convolution_param().bias_term()) ||
	          (dst_layer->has_inner_product_param() && !dst_layer->inner_product_param().bias_term())
	        )) {
		BlobProto * bias = dst_layer->add_blobs();
		bias->mutable_shape()->add_dim(num_output);
		for (int j = 0; j < num_output; ++j)
			bias->add_data(float(0));
		if (dst_layer->has_convolution_param())
			dst_layer->mutable_convolution_param()->clear_bias_term();
		else dst_layer->mutable_inner_product_param()->clear_bias_term();
	}
	float * b = dst_layer->mutable_blobs(1)->mutable_data()->mutable_data();
	const float* scale_data = src_param.blobs(0).data().data();
	const float* shift_data = src_param.blobs(1).data().data();
	fun(scale_data, shift_data, wx, b, num_input, num_output);
	string src_name = bn2index[blob_name].name;
	string dst_name = blob_name;
	if (is_forward) {
		// bn2index == bn_top2index
		swap(src_name, dst_name);
	}
	if (--bn2index[blob_name].top_ref_count == 0) {
		vector<string *> &to_replace = bn_replace[src_name];
		for (auto &iter : to_replace)
			*iter = dst_name;
		bn2index2.erase(bn2index[blob_name].name);
		bn2index.erase(blob_name);
	}

	return true;
}
