#ifndef AT_LAYER_HPP_
#define AT_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class AffineTransformerLayer : public Layer<Dtype> {

public:
	explicit AffineTransformerLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	                        const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	                     const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "AffineTransformer"; }
	virtual inline int MinBottomBlobs() const { return 2; }
	virtual inline int MaxBottomBlobs() const { return 3; }
	virtual inline int MinTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	                         const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	                         const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{NOT_IMPLEMENTED;}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	int calcT(int length, int cont_index);
	virtual inline bool AllowForceBackward(int bottom_index) const {
		// Cannot propagate back to w.
		return bottom_index != 2;
	}
private:
	int nSrc, C, H, W;
	int nPerSrc, output_H_, output_W_;
	int N;
	bool const_W_;

	Blob<Dtype> dTheta_tmp;	// used for back propagation part in GPU implementation
	Blob<Dtype> all_ones_2;	// used for back propagation part in GPU implementation

	Blob<Dtype> full_theta;	// used for storing data and diff for full six-dim theta
	Dtype pre_defined_theta[6];
	bool is_pre_defined_theta[6];
	int pre_defined_count;
	vector<vector<pair<int, int> > > scale_shift;
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
