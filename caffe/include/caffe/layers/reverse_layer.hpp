#ifndef CAFFE_REVERSE_LAYER_HPP_
#define CAFFE_REVERSE_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
  template <typename Dtype>
  class ReverseLayer : public Layer<Dtype> {
   public:
    explicit ReverseLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Reverse"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   private:
    ReverseParameter reverse_param_;
    Blob<int> bottom_counts_;
    int axis_;
  };
}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_
