// ------------------------------------------------------------------
// Written by Tong He
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/sum_layer.hpp"

namespace caffe {

template <typename Dtype>
void sumation_forward(const int count, const Dtype* from_data, Dtype* to_data, 
  const int* bottom_counts, const int* top_counts, const int num_axes, const int axis) {
  int from_inds[kMaxBlobAxes] = {0};
  for (int index = 0; index < count; index++) {
    int from_index = index, to_index = 0;
    for (int i = 0; i < num_axes; i++) {
      from_inds[i] = from_index / bottom_counts[i];
      from_index = from_index % bottom_counts[i];
    }
    for (int i = 0; i < num_axes; i++) {
      if (i != axis)
        to_index += from_inds[i] * top_counts[i];
    }

    *(to_data+to_index) += *(from_data+index);
  }
}


template <typename Dtype>
void sumation_backward(const int count, Dtype* bottom_diff, const Dtype* top_diff, 
  const int* bottom_counts, const int* top_counts, const int num_axes, const int axis) {
  int from_inds[kMaxBlobAxes] = {0};
  for (int index = 0; index < count; index++) {
    int from_index = index, to_index = 0;
    for (int i = 0; i < num_axes; i++) {
      from_inds[i] = from_index / bottom_counts[i];
      from_index = from_index % bottom_counts[i];
    }
    for (int i = 0; i < num_axes; i++) {
      if (i != axis)
        to_index += from_inds[i] * top_counts[i];
    }

    *(bottom_diff+index) = *(top_diff+to_index);
    
  }
}




  template <typename Dtype>
  void SumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    SumParameter sum_param = this->layer_param_.sum_param();
    axis_ = sum_param.axis();
  }



  template <typename Dtype>
  void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    vector<int> shape = bottom[0]->shape();
    CHECK_LT(axis_, shape.size()) << "axis should < dimension of the bottom";
    CHECK_GE(axis_, 0) << "axis should be >= 0";

    shape[axis_] = 1;
    top[0]->Reshape(shape);
    caffe_set(top[0]->count(), Dtype(0.), top[0]->mutable_cpu_data());
    num_axis_ = bottom[0]->shape().size();

    shape.clear();
    shape.push_back(num_axis_);
    bottom_counts_.Reshape(shape);
    top_counts_.Reshape(shape);

    int* bottom_counts_data=bottom_counts_.mutable_cpu_data();
    int* top_counts_data = top_counts_.mutable_cpu_data();

    for (int i = 1; i < num_axis_; i++){
        *bottom_counts_data = bottom[0]->count(i);
        *top_counts_data = top[0]->count(i);
        bottom_counts_data++;
        top_counts_data++;
    }

    *bottom_counts_data = 1;
    *top_counts_data = 1;

  }




  template <typename Dtype>
  void SumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const int count = bottom[0]->count();
      Dtype* top_data = top[0]->mutable_cpu_data();
      const int* bottom_counts_data = bottom_counts_.cpu_data();
      const int* top_counts_data = top_counts_.cpu_data();
      sumation_forward(count, bottom_data, top_data, bottom_counts_data, top_counts_data, num_axis_, axis_);

  }



  template <typename Dtype>
  void SumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
      //if (!propagate_down[0])
      //  return;
      const int* bottom_counts_data = bottom_counts_.cpu_data();
      const int* top_counts_data = top_counts_.cpu_data();
      const int count = bottom[0]->count();
      sumation_backward(count, bottom_diff, top_diff, bottom_counts_data, top_counts_data, num_axis_, axis_);

  }



INSTANTIATE_CLASS(SumLayer);
REGISTER_LAYER_CLASS(Sum);


}  // namespace caffe
