#ifndef CAFFE_UNITBOX_LOSS_LAYER_HPP_
#define CAFFE_UNITBOX_LOSS_LAYER_HPP_

#include <vector>
#include <cfloat>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class UnitboxLossLayer : public LossLayer<Dtype> {
 public:
  explicit UnitboxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "UnitboxLoss"; }
virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
   //virtual inline int ExactNumTopBlobs() const { return 1; }
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 //   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
 //     const vector<Blob<Dtype>*>& top);

  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int width_;
  int height_;
  int channels_;
  int num_;
  int pos_count_;
  bool output_map_;



};

}  // namespace caffe

#endif  // CAFFE_UNITBOX_LOSS_LAYER_HPP_
