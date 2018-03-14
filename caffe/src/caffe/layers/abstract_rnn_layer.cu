#include "caffe/layers/sequence_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void copy_indicator_cuda(const int count, const int output_feature_dim_,
                                    const Dtype * cont_t, const Dtype * src, Dtype * dst) {
  CUDA_KERNEL_LOOP(i, count) {
    const int b = i / output_feature_dim_;
    dst[i] = (cont_t[b] > 0) ? src[i] : Dtype(0);
  }
}

template <typename Dtype>
void AbstractRNNLayer<Dtype>::copy_indicator(const int count,
    const int output_feature_dim_, const Dtype * cont_t, const Dtype * src, Dtype * dst) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  copy_indicator_cuda<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, output_feature_dim_, cont_t, src, dst);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void AbstractRNNLayer<Dtype>::copy_prev_gpu(int t, int count, const Dtype *cont_t,
    const Dtype *h_t, Dtype *h_prev, const Dtype *c_t, Dtype *c_prev) {
  if (t > 0) {
    if (cont_t) {
      if (c_prev) copy_indicator(count, output_feature_dim_,
                                cont_t, c_t - count, c_prev);
      copy_indicator(count, output_feature_dim_,
                     cont_t, h_t - count, h_prev);
    } else {
      if (c_prev) caffe_copy(count, c_t - count, c_prev);
      caffe_copy(count, h_t - count, h_prev);
    }
  } else {
    if (c_prev) caffe_gpu_set(count, Dtype(0), c_prev);
    caffe_gpu_set(count, Dtype(0), h_prev);
  }
}

#define INST_INDICATOR(classname, type)                               \
template void classname<type>::copy_indicator(const int, const int,   \
    const type *, const type *, type *)
INSTANTIATE(INST_INDICATOR, AbstractRNNLayer);

#define INST_COPY_PREV(classname, type)                               \
template void classname<type>::copy_prev_gpu(int, int, const type *,  \
    const type *, type *, const type *, type *);
INSTANTIATE(INST_COPY_PREV, AbstractRNNLayer);

}; // namespace caffe
