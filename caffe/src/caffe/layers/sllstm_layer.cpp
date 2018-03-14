#include "caffe/layers/sequence_layers.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
void SLLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  AbstractRNNLayer<Dtype>::Reshape(bottom, top);
  cell_.ReshapeLike(*top[0]);
  buffer_c_prev_.ReshapeLike(this->buffer_h_prev_);
  buffer_c_diff_.ReshapeLike(this->buffer_h_prev_);
}

#define sllstm_cpu_forward                                                   \
Dtype * c_t = c + t * count;                                                 \
for (int i = 0; i < batch; i++) {                                            \
  for (int f = 0; f < this->output_feature_dim_; f++) {                      \
    const int offset = i * hidden_dim_ + f,                                  \
              index = i * this->output_feature_dim_ + f;                     \
    const int fi = I * this->output_feature_dim_ + offset,                   \
              ff = F * this->output_feature_dim_ + offset,                   \
              fo = O * this->output_feature_dim_ + offset,                   \
              fg = G * this->output_feature_dim_ + offset;                   \
    gates_t[fi] = this->sigmoid_cpu(gates_t[fi]);                            \
    gates_t[ff] = this->sigmoid_cpu(gates_t[ff]);                            \
    gates_t[fo] = this->sigmoid_cpu(gates_t[fo]);                            \
    gates_t[fg] = this->tanh_cpu(gates_t[fg]);                               \
    c_t[index] = gates_t[fi] * gates_t[fg] + gates_t[ff] * c_prev[index];    \
    h_t[index] = gates_t[fo] * this->tanh_cpu(c_t[index]);                   \
  }                                                                          \
}

template <typename Dtype>
void SLLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  Dtype * c = cell_.mutable_cpu_data();
  Dtype * c_prev = buffer_c_prev_.mutable_cpu_data();
  Dtype * buffer_uh = this->gates_.mutable_cpu_data(); // used as gates
  FORWARD(cpu, sllstm_cpu_forward);
}

#define sllstm_cpu_backward                                                  \
FLAG = !FLAG;                                                                \
const Dtype * c_t_diff = c_diff[FLAG];                                       \
Dtype * c_backpropagate = c_diff[!FLAG];                                     \
caffe_cpu_set(count, Dtype(0), h_backpropagate);                             \
for (int i = 0; i < batch; i++) {                                            \
  for (int f = 0; f < this->output_feature_dim_; f++) {                      \
    const int offset = i * hidden_dim_ + f,                                  \
              index = i * this->output_feature_dim_ + f;                     \
    const int fi = I * this->output_feature_dim_ + offset,                   \
              ff = F * this->output_feature_dim_ + offset,                   \
              fo = O * this->output_feature_dim_ + offset,                   \
              fg = G * this->output_feature_dim_ + offset;                   \
    const Dtype tanhc = this->tanh_cpu(c_t[index]);                          \
    Dtype c_term_diff = c_t_diff[index] + (1 - tanhc * tanhc)                \
                        * gates_t[fo] * h_t_diff[index];                     \
    gates_t_diff[fi]  = gates_t[fg] * c_term_diff                            \
                        * gates_t[fi] * (1 - gates_t[fi]);                   \
    gates_t_diff[ff]  = c_prev[index] * c_term_diff                          \
                        * gates_t[ff] * (1 - gates_t[ff]);                   \
    gates_t_diff[fo]  = tanhc * h_t_diff[index]                              \
                        * gates_t[fo] * (1 - gates_t[fo]);                   \
    gates_t_diff[fg]  = gates_t[fi] * c_term_diff                            \
                        * (1 - gates_t[fg] * gates_t[fg]);                   \
    c_backpropagate[index] = gates_t[ff] * c_term_diff;                      \
  }                                                                          \
}

#define sllstm_cpu_copy                                                      \
for (int i = 0; i < batch; i++) {                                            \
  if (cont_t[i] <= 0) {                                                      \
    caffe_set(this->output_feature_dim_, Dtype(0),                           \
              h_backpropagate + i * this->output_feature_dim_);              \
    caffe_set(this->output_feature_dim_, Dtype(0),                           \
              c_backpropagate + i * this->output_feature_dim_);              \
  }                                                                          \
}

template <typename Dtype>
void SLLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  // clean c_prev(and diff) & h_prev(and diff)
  Dtype * c_diff[2];
  c_diff[0] = buffer_c_diff_.mutable_cpu_data();
  c_diff[1] = buffer_c_diff_.mutable_cpu_diff();
  caffe_cpu_set(bottom[0]->shape(1) * this->output_feature_dim_, Dtype(0), c_diff[0]);
  caffe_cpu_set(bottom[0]->shape(1) * this->output_feature_dim_, Dtype(0), c_diff[1]);
  Dtype * c_prev = buffer_c_prev_.mutable_cpu_data();
  const Dtype * c = cell_.cpu_data();
  bool FLAG = true;
  Dtype * buffer_uh = this->gates_.mutable_cpu_diff(); // used as gates_diff

  BACKWARD(cpu, sllstm_cpu_backward, sllstm_cpu_copy);
}

#ifdef CPU_ONLY
STUB_GPU(SLLSTMLayer);
#endif

INSTANTIATE_CLASS(SLLSTMLayer);
REGISTER_LAYER_CLASS(SLLSTM);
} // namespace caffe
