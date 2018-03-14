#include "caffe/layers/sequence_layers.hpp"
#include "caffe/filler.hpp"

namespace caffe {
template <typename Dtype>
inline Dtype AbstractRNNLayer<Dtype>::sigmoid_cpu(Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename Dtype>
inline Dtype AbstractRNNLayer<Dtype>::tanh_cpu(Dtype x) {
  return Dtype(2) * sigmoid_cpu(Dtype(2) * x) - Dtype(1);
}

template <typename Dtype>
void AbstractRNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // feature dim
  input_feature_dim_ = bottom[0]->shape(2);
  output_feature_dim_ = this->layer_param_.recurrent_param().num_output();
  clipping_threshold_ = this->layer_param_.recurrent_param().clipping_threshold();
  // blobs
  NumOfBlobs = bottom.size() == 3 ? 4 : 3;
  if (this->blobs_.size() > 0) {
    LOG(INFO) << this->layer_param_.name() << " Skipping parameter initialization.";
  } else {
    this->blobs_.resize(NumOfBlobs);
    // WX, WS
    vector<int> shape(2);
    shape[0] = NumOfGates * output_feature_dim_;
    shape[1] = input_feature_dim_;
    for (int i = WX; i < NumOfBlobs; i++) {
      this->blobs_[i].reset(new Blob<Dtype>(shape));
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
              this->layer_param_.recurrent_param().weight_filler()));
        weight_filler->Fill(this->blobs_[i].get());
    }
    // UH
    shape[1] = output_feature_dim_;
    this->blobs_[UH].reset(new Blob<Dtype>(shape));
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.recurrent_param().weight_filler()));
      weight_filler->Fill(this->blobs_[UH].get());
    // B
    shape.resize(1);
    shape[0] = NumOfGates * output_feature_dim_;
    this->blobs_[B].reset(new Blob<Dtype>(shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
            this->layer_param_.recurrent_param().bias_filler()));
      bias_filler->Fill(this->blobs_[B].get());
    this->param_propagate_down_.resize(this->blobs_.size(), true);
  }
}

template <typename Dtype>
void AbstractRNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  vector<int> shape(bottom[0]->shape());
  shape[2] = output_feature_dim_;
  top[0]->Reshape(shape);
  shape[2] = NumOfGates * output_feature_dim_;
  gates_.Reshape(shape);
  if (bottom.size() == 3) {
    CHECK_EQ(NumOfBlobs, 4);
    shape[0] = 1;
    x_static_ws_.Reshape(shape);
  } else CHECK_EQ(NumOfBlobs, 3);
  shape[0] = shape[1];
  shape[1] = output_feature_dim_;
  shape.resize(2);
  buffer_h_prev_.Reshape(shape);
  vector<int> bias_shape(1, bottom[0]->num() * bottom[0]->channels());
  bias_multiplier_.Reshape(bias_shape);
  caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void AbstractRNNLayer<Dtype>::copy_prev_cpu(int t, int count, const Dtype *cont_t,
    const Dtype *h_t, Dtype *h_prev, const Dtype *c_t, Dtype *c_prev) {
  if (t > 0) {
    if (cont_t) {
      int batch = count / output_feature_dim_;
      for (int i = 0; i < batch; i++) {
        if (cont_t[i] > 0) {
          if (c_prev) caffe_copy(output_feature_dim_, c_t - count + i * output_feature_dim_,
                                c_prev + i * output_feature_dim_);
          caffe_copy(output_feature_dim_, h_t - count + i * output_feature_dim_,
                     h_prev + i * output_feature_dim_);
        } else {
          if (c_prev) caffe_set(output_feature_dim_, Dtype(0), c_prev + i * output_feature_dim_);
          caffe_set(output_feature_dim_, Dtype(0), h_prev + i * output_feature_dim_);
        }
      }
    } else {
      if (c_prev) caffe_copy(count, c_t - count, c_prev);
      caffe_copy(count, h_t - count, h_prev);
    }
  } else {
    if (c_prev) caffe_set(count, Dtype(0), c_prev);
    caffe_set(count, Dtype(0), h_prev);
  }
}

#ifdef CPU_ONLY
STUB_GPU(AbstractRNNLayer);
#endif

INSTANTIATE_CLASS(AbstractRNNLayer);
} // namespace caffe
