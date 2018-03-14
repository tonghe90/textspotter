#ifndef CAFFE_SUM_LAYER_HPP_
#define CAFFE_SUM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SumLayer : public Layer<Dtype> {
 public:
  explicit SumLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Sum"; }
   virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //    const vector<Blob<Dtype>*>& top);
   // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
   //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   private:
  int axis_;
  Blob<int> bottom_counts_;
  int num_axis_;
  Blob<int> top_counts_;

};

}  // namespace caffe

#endif  // CAFFE_SUMCONV_LAYER_HPP_
