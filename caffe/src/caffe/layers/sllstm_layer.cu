#include "caffe/layers/sequence_layers.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype sigmoid(const Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename Dtype>
__device__ Dtype tanh(const Dtype x) {
  return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
}

template <typename Dtype>
__global__ void lstm_forward_kernel(const int count, const int output_feature_dim_,
                                    Dtype * gates, Dtype * h, Dtype * c,
                                    const Dtype * c_prev) {
  CUDA_KERNEL_LOOP(index, count) {
    const int index_batch = index / output_feature_dim_,
              index_feature = index % output_feature_dim_;
    const int offset = index_batch * 4 * output_feature_dim_ + index_feature;
    const int fi = SLLSTMLayer<Dtype>::I * output_feature_dim_ + offset,
              ff = SLLSTMLayer<Dtype>::F * output_feature_dim_ + offset,
              fo = SLLSTMLayer<Dtype>::O * output_feature_dim_ + offset,
              fg = SLLSTMLayer<Dtype>::G * output_feature_dim_ + offset;
    gates[fi] = sigmoid(gates[fi]);
    gates[ff] = sigmoid(gates[ff]);
    gates[fo] = sigmoid(gates[fo]);
    gates[fg] = tanh(gates[fg]);

    c[index] = gates[fi] * gates[fg] + gates[ff] * c_prev[index];
    h[index] = gates[fo] * tanh(c[index]);
  }
}

template <typename Dtype>
__global__ void lstm_backward_kernel(const int batch, const int output_feature_dim_,
                                     const Dtype * gates, Dtype * gates_diff,
                                     const Dtype * c, const Dtype * c_diff,
                                     const Dtype * c_prev, Dtype * c_backpropagate,
                                     const Dtype * h_diff) {
  CUDA_KERNEL_LOOP(index, batch * output_feature_dim_) {
    const int index_batch = index / output_feature_dim_,
              index_feature = index % output_feature_dim_;
    const int offset = index_batch * 4 * output_feature_dim_ + index_feature;
    const int fi = SLLSTMLayer<Dtype>::I * output_feature_dim_ + offset,
              ff = SLLSTMLayer<Dtype>::F * output_feature_dim_ + offset,
              fo = SLLSTMLayer<Dtype>::O * output_feature_dim_ + offset,
              fg = SLLSTMLayer<Dtype>::G * output_feature_dim_ + offset;
    const Dtype tanhc = tanh(c[index]);

    gates_diff[fo] = tanhc * h_diff[index];
    Dtype c_term_diff = c_diff[index] + (Dtype(1) - tanhc * tanhc)
                        * gates[fo] * h_diff[index];
    gates_diff[ff] = c_prev[index] * c_term_diff;
    c_backpropagate[index] = gates[ff] * c_term_diff;
    gates_diff[fi] = gates[fg] * c_term_diff;
    gates_diff[fg] = gates[fi] * c_term_diff;
  }
}

template <typename Dtype>
__global__ void lstm_acts_backward(const int count, const int output_feature_dim_,
                                   const Dtype * gates, Dtype * gates_diff) {
  const int x_dim = 4 * output_feature_dim_;
  CUDA_KERNEL_LOOP(index, count) {
    const int d = index % x_dim;
    const Dtype x_act = gates[index];
    if (d < 3 * output_feature_dim_)
      gates_diff[index] = x_act * (Dtype(1) - x_act) * gates_diff[index];
    else
      gates_diff[index] = (Dtype(1) - x_act * x_act) * gates_diff[index];
  }
}

#define sllstm_gpu_forward                                                           \
lstm_forward_kernel<Dtype> <<< CAFFE_GET_BLOCKS(count),                              \
                        CAFFE_CUDA_NUM_THREADS>>>(count, this->output_feature_dim_,  \
                            gates_t, h_t, c+t*count, c_prev);                        \
CUDA_POST_KERNEL_CHECK;

template <typename Dtype>
void SLLSTMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  Dtype * c = cell_.mutable_gpu_data();
  Dtype * c_prev = buffer_c_prev_.mutable_gpu_data();
  Dtype * buffer_uh = this->gates_.mutable_gpu_data(); // used as gates
  FORWARD(gpu, sllstm_gpu_forward);
}

#define sllstm_gpu_backward                                                          \
FLAG = !FLAG;                                                                        \
const Dtype * c_t_diff = c_diff[FLAG];                                               \
Dtype * c_backpropagate = c_diff[!FLAG];                                             \
caffe_gpu_set(count, Dtype(0), h_backpropagate);                                     \
lstm_backward_kernel<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>   \
  (batch, this->output_feature_dim_, gates_t, gates_t_diff, c_t, c_t_diff, c_prev,   \
  c_backpropagate, h_t_diff);                                                        \
CUDA_POST_KERNEL_CHECK;                                                              \
lstm_acts_backward<Dtype> <<< CAFFE_GET_BLOCKS(count*this->NumOfGates),              \
  CAFFE_CUDA_NUM_THREADS>>>(count * this->NumOfGates, this->output_feature_dim_,     \
  gates_t, gates_t_diff);                                                            \
CUDA_POST_KERNEL_CHECK;


#define sllstm_gpu_copy                                                              \
this->copy_indicator(count, this->output_feature_dim_,                               \
                     cont_t, h_backpropagate, h_backpropagate);                      \
this->copy_indicator(count, this->output_feature_dim_,                               \
                     cont_t, c_backpropagate, c_backpropagate);

template <typename Dtype>
void SLLSTMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  // clean c_prev(and diff) & h_prev(and diff)
  Dtype * c_diff[2];
  c_diff[0] = buffer_c_diff_.mutable_gpu_data();
  c_diff[1] = buffer_c_diff_.mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->shape(1) * this->output_feature_dim_, Dtype(0), c_diff[0]);
  caffe_gpu_set(bottom[0]->shape(1) * this->output_feature_dim_, Dtype(0), c_diff[1]);
  Dtype * c_prev = buffer_c_prev_.mutable_gpu_data();
  const Dtype * c = cell_.gpu_data();
  bool FLAG = true;
  Dtype * buffer_uh = this->gates_.mutable_gpu_diff(); // used as gates_diff

  BACKWARD(gpu, sllstm_gpu_backward, sllstm_gpu_copy);
}

INSTANTIATE_LAYER_GPU_FUNCS(SLLSTMLayer);
}; // namespace caffe
