#ifndef CAFFE_SEQUENCE_LAYERS_HPP_
#define CAFFE_SEQUENCE_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include <tr1/unordered_map>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/3rdparty/ctc.h"

namespace caffe {
#define caffe_cpu_add caffe_add
#define caffe_cpu_set caffe_set

/************************************** FORWARD() ************************************/
#define FORWARD(type, LOOP_KERNEL)                                                    \
const Dtype * x = bottom[0]->type##_data();                                           \
const Dtype * cont = (bottom.size() > 1) ? bottom[1]->type##_data() : NULL;           \
const Dtype * x_static = (bottom.size() > 2) ? bottom[2]->type##_data() : NULL;       \
const int T = bottom[0]->shape(0);                                                    \
const int batch = bottom[0]->shape(1);                                                \
const int count = batch * this->output_feature_dim_;                                  \
const int hidden_dim_ = this->NumOfGates * this->output_feature_dim_;                 \
const Dtype * wx = this->blobs_[this->WX]->type##_data();                             \
const Dtype * ws = (x_static) ? this->blobs_[this->WS]->type##_data() : NULL;         \
const Dtype * uh = this->blobs_[this->UH]->type##_data();                             \
const Dtype * b = this->blobs_[this->B]->type##_data();                               \
const Dtype * bias_multiplier = this->bias_multiplier_.type##_data();                 \
Dtype * x_static_ws = (x_static) ? this->x_static_ws_.mutable_##type##_data() : NULL; \
Dtype * gates = this->gates_.mutable_##type##_data();                                 \
Dtype * h = top[0]->mutable_##type##_data();                                          \
Dtype * h_prev = this->buffer_h_prev_.mutable_##type##_data();                        \
caffe_##type##_gemm(CblasNoTrans, CblasTrans, T * batch, hidden_dim_,                 \
               this->input_feature_dim_, Dtype(1), x, wx, Dtype(0), gates);           \
caffe_##type##_gemm(CblasNoTrans, CblasNoTrans, T * batch, hidden_dim_, 1,            \
               Dtype(1), bias_multiplier, b, Dtype(1), gates);                        \
if (x_static)  caffe_##type##_gemm(CblasNoTrans, CblasTrans, batch, hidden_dim_,      \
                              this->input_feature_dim_, Dtype(1), x_static, ws,       \
                              Dtype(0), x_static_ws);                                 \
for (int t = 0; t < T; t++) {                                                         \
  const Dtype * cont_t = (cont ? cont + t * batch : NULL);                            \
  Dtype * gates_t = gates + t * this->NumOfGates * count;                             \
  Dtype * h_t = h + t * count;                                                        \
  Dtype * buffer_uh_t = buffer_uh + t * this->NumOfGates * count;                     \
  if (x_static)                                                                       \
    caffe_##type##_add(this->x_static_ws_.count(), x_static_ws, gates_t, gates_t);    \
  this->copy_prev_##type(t, count, cont_t, h_t, h_prev, c+t*count, c_prev);           \
  caffe_##type##_gemm(CblasNoTrans, CblasTrans, batch, hidden_dim_,                   \
        this->output_feature_dim_, Dtype(1), h_prev, uh, Dtype(1), buffer_uh_t);      \
  LOOP_KERNEL                                                                         \
}

/********************************** BACKWARD() ***************************************/
#define BACKWARD(type, LOOP_KERNEL, COPY)                                             \
if (propagate_down.size() > 1)                                                        \
  CHECK(!propagate_down[1]) << "Cannot back-propagate to continuous indicator.";      \
if (!propagate_down[0] && (propagate_down.size() < 3 || !propagate_down[2]))          \
  return;                                                                             \
const Dtype * x_static = NULL, *ws = NULL;                                            \
Dtype * ws_diff = NULL, *x_static_diff = NULL;                                        \
if (bottom.size() > 2) {                                                              \
  ws = this->blobs_[this->WS]->type##_data();                                         \
  if (this->param_propagate_down_[this->WS])                                          \
    ws_diff = this->blobs_[this->WS]->mutable_##type##_diff();                        \
  x_static = bottom[2]->type##_data();                                                \
  if (propagate_down[2]) {                                                            \
    x_static_diff = bottom[2]->mutable_##type##_diff();                               \
    caffe_##type##_set(bottom[2]->count(), Dtype(0),                                  \
      bottom[2]->mutable_##type##_diff());                                            \
  }                                                                                   \
}                                                                                     \
const int T = bottom[0]->shape(0);                                                    \
const int batch = bottom[0]->shape(1);                                                \
const int count = batch * this->output_feature_dim_;                                  \
Dtype * h_prev = this->buffer_h_prev_.mutable_##type##_data();                        \
Dtype * h_backpropagate = this->buffer_h_prev_.mutable_##type##_diff();               \
caffe_##type##_set(count, Dtype(0), h_backpropagate);                                 \
const Dtype * x = bottom[0]->type##_data();                                           \
const Dtype * cont = (bottom.size() > 1) ? bottom[1]->type##_data() : NULL;           \
const Dtype * h = top[0]->type##_data();                                              \
const Dtype * gates = this->gates_.type##_data();                                     \
const Dtype * wx = this->blobs_[this->WX]->type##_data();                             \
const Dtype * uh = this->blobs_[this->UH]->type##_data();                             \
const Dtype * bias_multiplier = this->bias_multiplier_.type##_data();                 \
Dtype * h_diff = top[0]->mutable_##type##_diff();                                     \
Dtype * gates_diff = this->gates_.mutable_##type##_diff();                            \
Dtype * wx_diff = this->param_propagate_down_[this->WX] ?                             \
                  this->blobs_[this->WX]->mutable_##type##_diff() : NULL;             \
Dtype * uh_diff = this->param_propagate_down_[this->UH] ?                             \
                  this->blobs_[this->UH]->mutable_##type##_diff() : NULL;             \
Dtype * b_diff = this->param_propagate_down_[this->B] ?                               \
                 this->blobs_[this->B]->mutable_##type##_diff() : NULL;               \
Dtype * x_diff = propagate_down[0] ? bottom[0]->mutable_##type##_diff() : NULL;       \
const int hidden_dim_ = this->NumOfGates * this->output_feature_dim_;                 \
for (int t = T - 1; t >= 0; t--) {                                                    \
  const Dtype * cont_t = cont ? cont + t * batch : NULL;                              \
  int offset = t * count;                                                             \
  const Dtype * h_t = h + offset;                                                     \
  const Dtype * c_t = c + offset;                                                     \
  const Dtype * gates_t = gates + offset * this->NumOfGates;                          \
  Dtype * gates_t_diff = gates_diff + offset * this->NumOfGates;                      \
  Dtype * h_t_diff = h_diff + offset;                                                 \
  Dtype * buffer_uh_t = buffer_uh + offset * this->NumOfGates;                        \
  caffe_##type##_add(count, h_backpropagate, h_t_diff, h_t_diff);                     \
  this->copy_prev_##type(t, count, cont_t, h_t, h_prev, c_t, c_prev);                 \
  LOOP_KERNEL                                                                         \
  if (this->clipping_threshold_ > 0)                                                  \
    caffe_##type##_bound(count * this->NumOfGates, gates_t_diff,                      \
                        -this->clipping_threshold_,                                   \
                        this->clipping_threshold_, gates_t_diff);                     \
  if (x_static_diff) caffe_##type##_gemm(CblasNoTrans, CblasNoTrans,                  \
                                 batch, this->input_feature_dim_, hidden_dim_,        \
                                 Dtype(1), gates_t_diff, ws,                          \
                                 Dtype(1), x_static_diff);                            \
  if (ws_diff) caffe_##type##_gemm(CblasTrans, CblasNoTrans,                          \
                                 hidden_dim_, this->input_feature_dim_, batch,        \
                                 Dtype(1), gates_t_diff, x_static,                    \
                                 Dtype(1), ws_diff);                                  \
  caffe_##type##_gemm(CblasTrans, CblasNoTrans, hidden_dim_, this->output_feature_dim_,\
                 batch, Dtype(1), buffer_uh_t, h_prev, Dtype(1), uh_diff);            \
  caffe_##type##_gemm(CblasNoTrans, CblasNoTrans, batch, this->output_feature_dim_,   \
                 hidden_dim_, Dtype(1), buffer_uh_t,                                  \
                 uh, Dtype(1), h_backpropagate);                                      \
  if (t > 0 && cont_t) { COPY }                                                       \
}                                                                                     \
if (x_diff) caffe_##type##_gemm(CblasNoTrans, CblasNoTrans, batch * T,                \
                             this->input_feature_dim_, hidden_dim_,                   \
                             Dtype(1), gates_diff, wx, Dtype(0), x_diff);             \
if (wx_diff) caffe_##type##_gemm(CblasTrans, CblasNoTrans,                            \
                             hidden_dim_, this->input_feature_dim_, T * batch,        \
                             Dtype(1), gates_diff, x, Dtype(1), wx_diff);             \
if (b_diff) caffe_##type##_gemv<Dtype>(CblasTrans, T * batch, hidden_dim_, Dtype(1),  \
                      gates_diff, bias_multiplier, Dtype(1), b_diff);                 \

/**
 * Abstract RNN.
 *  1. Bottom blobs:
 *      [0]: required. input sequence.
 *        size: Sequence_length x Batch_size x Feature_dim
 *      [1]: optional. continuous indicator.
 *        size: Sequence_length x Batch_size
 *      [2]: optional. static input.
 *        size: Batch_size x Feature_dim
 *  2. Output blobs:
 *      [0]: output sequence.
 */
template <typename Dtype>
class AbstractRNNLayer : public Layer<Dtype> {
public:
  explicit AbstractRNNLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {NumOfGates = 0;}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AbstractRNN"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline bool AllowForceBackward(int bottom_index) const {
    // Cannot propagate back to continuous indicator.
    return bottom_index != 1;
  }

protected:
  // tutorial
  // add your own definition here
  // if cell is not used, please set them to NULL
  // if buffer_uh is not used, please set it equal to gates
  // finally call FORWARD(...) or BACKWARD(...)
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) = 0;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) = 0;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) = 0;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) = 0;

  enum BlobName {
    UH = 0, B, WX, WS
  };

  void copy_prev_gpu(int t, int count, const Dtype *cont_t,
                     const Dtype *h_t, Dtype *h_prev,
                     const Dtype *c_t = NULL, Dtype *c_prev = NULL);
  void copy_prev_cpu(int t, int count, const Dtype *cont_t,
                     const Dtype *h_t, Dtype *h_prev,
                     const Dtype *c_t = NULL, Dtype *c_prev = NULL);
  Dtype sigmoid_cpu(Dtype x);
  Dtype tanh_cpu(Dtype x);
  void copy_indicator(const int count, const int output_feature_dim_,
                      const Dtype * cont_t, const Dtype * src, Dtype * dst);

  Blob<Dtype> gates_,
       buffer_h_prev_,
       x_static_ws_;

  int input_feature_dim_,
      output_feature_dim_,
      NumOfBlobs,
      NumOfGates;
  Blob<Dtype> bias_multiplier_;
  Dtype clipping_threshold_;
};

/**
 * Single-layered LSTM.
 */
template <typename Dtype>
class SLLSTMLayer : public AbstractRNNLayer<Dtype> {
public:
  explicit SLLSTMLayer(const LayerParameter& param)
    : AbstractRNNLayer<Dtype>(param) {this->NumOfGates = 4;}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SLLSTM"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> cell_,    // C
       buffer_c_prev_,
       buffer_c_diff_;
public:
  enum GatesName {
    I = 0, F, O, G
  };
};

/**
 * Single-layered GRNN.
 */
template <typename Dtype>
class SLGRNNLayer : public AbstractRNNLayer<Dtype> {
public:
  explicit SLGRNNLayer(const LayerParameter& param)
    : AbstractRNNLayer<Dtype>(param) {this->NumOfGates = 3;}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SLGRNN"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> buffer_uh_;
public:
  enum {
    Z = 0, R, G
  };
};

#define EPS Dtype(1e-20)
#define BLANK 0
#define CHECK_CTC_STATUS(ret_status) do{ctcStatus_t status = ret_status; \
                                        CHECK_EQ(status, CTC_STATUS_SUCCESS) \
<< std::string(ctcGetStatusString(status));} while(0);

#define NGRAM 3

template <typename Dtype>
class LM {
public:
  void Init(const char *filename, Dtype alpha) {
    _alpha = alpha;
    FILE *fp = fopen(filename, "rb");
    CHECK(fp) << "Unable to open " << filename;
    size_t count[3];
    fread(count, sizeof(size_t), 3, fp);
    gram1.resize(count[0]);
    gram2.resize(count[1]);
    gram3.resize(count[2]);
    for (auto &g : gram1) fread(&g, sizeof(g), 1, fp);
    for (auto &g : gram2) fread(&g, sizeof(g), 1, fp);
    for (auto &g : gram3) fread(&g, sizeof(g), 1, fp);
    fclose(fp);
  }
  struct NGramNode {
    int _word;
    int _pointer;
    Dtype _logp;
    Dtype _logb;
  };
  struct NGramNodeLast {
    int _word;
    Dtype _logp;
  };
  double languageModelProb(const std::vector<int> &path) const {
    const int path_len = path.size();
    const size_t start = std::max(0, path_len - NGRAM);
    Dtype logp = getLMpriority(&path[start], path_len - start);
    return std::pow(10, logp * _alpha); // add the alpha constrain
  }
private:
  template <class T>
  int binary_search(const int &p1, const int &p2,
                    const std::vector<T> &gram, const int &target) const {
    if (p1 >= p2) return -1;
    const int mid = (p1 + p2) >> 1;
    const auto &g = gram[mid];
    if (g._word == target) return mid;
    else if (g._word < target) return binary_search(mid + 1, p2, gram, target);
    else return binary_search(p1, mid - 1, gram, target);
  }
  Dtype getLMbow(int a, int b) const {
    const int result = binary_search(gram1[b]._pointer,
                                     gram1[b + 1]._pointer,
                                     gram2, a);
    if (result == -1) return 0.;
    return gram2[result]._logb;
  }

  Dtype getLMpriority(const int *index, const int index_size) const {
    // assume index_size <= 3
    if (index_size <= 0) return 0;
    int idx = index[index_size - 1];
    if (index_size == 1) return gram1[idx]._logp;
    int result = binary_search(gram1[idx]._pointer,
                               gram1[idx + 1]._pointer,
                               gram2,
                               index[index_size - 2]);
    if (result != -1) {
      if (index_size == 2) {
        return gram2[result]._logp;
      } else {
        result = binary_search(gram2[result]._pointer,
                               gram2[result + 1]._pointer,
                               gram3,
                               *index);
        if (result != -1) return gram3[result]._logp;
        return getLMbow(index[0], index[1]) + gram2[result]._logp;
      }
    } else {
      if (index_size == 2)
        return gram1[index[0]]._logb + gram1[index[1]]._logp;
      else
        return getLMbow(index[0], index[1]) + gram1[index[1]]._logb + gram1[index[2]]._logp;
    }
  }
  std::vector<NGramNode> gram1, gram2;
  std::vector<NGramNodeLast> gram3;
  Dtype _alpha;
};

/**
 * @brief Implement CTC (Connectionist Temporal Classification) loss function in [1].
 *
 * [1] Graves A, Fernández S, Gomez F, et al. Connectionist temporal
 *     classification: labelling unsegmented sequence data with recurrent
 *     neural networks[C]//Proceedings of the 23rd international
 *     conference on Machine learning. ACM, 2006: 369-376.
 */
template <typename Dtype>
class StandardCTCLayer : public Layer<Dtype> {
public:
  explicit StandardCTCLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "StandardCTC"; }
  virtual inline int MaxBottomBlobs() const { return 4; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }
protected:
  /// @copydoc StandardCTCLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  typedef std::tr1::unordered_map<int, vector<int> > tmap;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  Dtype Backward_internal(const vector<Blob<Dtype>*>& top,
                          const vector<Blob<Dtype>*>& bottom);
  void BestPathDecode(Dtype *target_seq, const Dtype *y, const int tt,
                      Dtype *target_indicator, Dtype *target_score);
  void BestPathThresDecode(Dtype *target_seq, const Dtype *y, const int tt,
                           Dtype *target_indicator, Dtype *target_score);
  void PrefixSearchDecode(Dtype *target_seq, const Dtype *y, const int tt);
  void PrefixSearchDecode_inner(Dtype * &target_seq, const Dtype *y, const int tt);
  void PrefixLMDecode(Dtype *&target_seq, const Dtype *y, const int tt);
  void FlattenLabels(const Blob<Dtype>* label_blob, const Blob<Dtype>* cont_blob = NULL);
  Dtype AlphaBeta( const int *target, int L, const Dtype * input_seq,
                   const int tt, Dtype * bottom_diff);
  void Test(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  int T, N, C, gap_per_T, thread_num;
  CTCParameter_Decoder decode_type;
  bool warpctc;
  std::vector<int> input_lengths_, label_lengths_, flat_labels_;
  int beam_size;
  Dtype thres_cum;
  Dtype thres_above;
  Dtype thres_below;
  Dtype alpha;
  Dtype beta;
  LM<Dtype> lm;
  Blob<Dtype> * mask;
  int pos_count;
};

/**
 * @brief Computes the accuracy for a text
 *        classification task.
 */
template <typename Dtype>
class AccuracyTextLayer : public Layer<Dtype> {
public:
  explicit AccuracyTextLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AccuracyText"; }
  virtual inline int MaxBottomBlobs() const { return 4; }
  virtual inline int MinBottomBlobs() const { return 3; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- AccuracyTextLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  /// Whether to compute edit-distance based accuracy.
  bool has_edit_distance;
  int T, N;
  vector<wchar_t> dict;
};

/**
 * @brief Reverse a sequence, used for bidirectional RNN
 */
template <typename Dtype>
class ReverseLayer : public Layer<Dtype> {
public:
  explicit ReverseLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Reverse"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int T, N, feat_len;
};

/*
 * @brief Exchange dimensions
 *   [0, 1, 2, 3] ----> [3, 0, 1, 2]
 */
template <typename Dtype>
class ExchangeLayer : public Layer<Dtype> {
public:
  explicit ExchangeLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Exchange"; }
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
};

#ifdef USE_SENSEDNN
/**
 *  cuDNN implementation of RNN.
 *  1. Bottom blobs:
 *      [0]: required. input sequence.
 *        size: Sequence_length x Batch_size x Feature_dim
 *      [1]: optional. continuous indicator.
 *        size: Sequence_length x Batch_size
 *      [2]: optional. static input.
 *        size: Batch_size x Feature_dim
 *  2. Output blobs:
 *      [0]: output sequence.
 */
/*
template <typename Dtype>
class SenseDNNRNNLayer : public Layer<Dtype> {
public:
  explicit SenseDNNRNNLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SenseDNN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;};
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {NOT_IMPLEMENTED;};
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  int input_feature_dim_,
      output_feature_dim_;
  float dropout_ratio;
  Blob<Dtype> dropoutStates;

  bool handles_setup_;
  cudnnHandle_t               handle_;
  cudnnTensorDescriptor_t     bottom_desc_;
  cudnnTensorDescriptor_t     top_desc_;
  cudnnDropoutDescriptor_t    dropout_desc_;
  cudnnRNNDescriptor_t        rnn_desc_;
};
*/
#endif

}  // namespace caffe

#endif  // CAFFE_SEQUENCE_LAYERS_HPP_