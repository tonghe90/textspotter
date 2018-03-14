#ifndef CAFFE_ATT_LSTM_LAYER_HPP_
#define CAFFE_ATT_LSTM_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/recurrent_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/lstm_new_layer.hpp"
#include "caffe/util/format.hpp"

namespace caffe {

template <typename Dtype> class RecurrentLayer;

/**
 * @brief Processes sequential inputs using a "Long Short-Term Memory" (LSTM)
 *        [1] style recurrent neural network (RNN). Implemented by unrolling
 *        the LSTM computation through time.
 *
 * The specific architecture used in this implementation is as described in
 * "Learning to Execute" [2], reproduced below:
 *     i_t := \sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
 *     f_t := \sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
 *     o_t := \sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
 *     g_t :=    \tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
 *     c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
 *     h_t := o_t .* \tanh[c_t]
 * In the implementation, the i, f, o, and g computations are performed as a
 * single inner product.
 *
 * Notably, this implementation lacks the "diagonal" gates, as used in the
 * LSTM architectures described by Alex Graves [3] and others.
 *
 * [1] Hochreiter, Sepp, and Schmidhuber, Jürgen. "Long short-term memory."
 *     Neural Computation 9, no. 8 (1997): 1735-1780.
 *
 * [2] Zaremba, Wojciech, and Sutskever, Ilya. "Learning to execute."
 *     arXiv preprint arXiv:1410.4615 (2014).
 *
 * [3] Graves, Alex. "Generating sequences with recurrent neural networks."
 *     arXiv preprint arXiv:1308.0850 (2013).
 */
template <typename Dtype>
class AttLstmLayer : public RecurrentLayer<Dtype> {
 public:
  explicit AttLstmLayer(const LayerParameter& param)
      : RecurrentLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "AttLstm"; }

 protected:
  virtual void FillUnrolledNet(NetParameter* net_param) const;
  virtual void RecurrentInputBlobNames(vector<string>* names) const;
  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
  virtual void OutputBlobNames(vector<string>* names) const;
  // virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  // virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);


  virtual inline int MinBottomBlobs() const {
    return 2;
  }
  virtual inline int MaxBottomBlobs() const { return MinBottomBlobs() + 1; }
  virtual inline int ExactNumTopBlobs() const {
    return 2;
  }


  int E_T_;
};


}  // namespace caffe

#endif  // CAFFE_ATT_LSTM_LAYER_HPP_
