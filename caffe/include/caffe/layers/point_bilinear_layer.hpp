#ifndef CAFFE_POINT_BILINEAR_LAYER_HPP_
#define CAFFE_POINT_BILINEAR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class PointBilinearLayer : public Layer<Dtype> {
 public:
  explicit PointBilinearLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PointBilinear"; }
   virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  /// @copydoc AbsValLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
   //    const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
   //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
  Dtype ratio_;
  int num_;
  int channel_;
  int height_;
  int width_;
  int point_num_;
  bool has_id_;
  // Blob<Dtype> buffer_;









};

}  // namespace caffe

#endif  
