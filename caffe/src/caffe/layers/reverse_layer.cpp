#include "caffe/layers/sequence_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReverseLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  T = bottom[0]->num();
  N = bottom[0]->channels();
  if (bottom.size() == 2) {
    feat_len = bottom[1]->count()/(T*N);
    CHECK_EQ(bottom[1]->num(), T);
    CHECK_EQ(bottom[1]->channels(), N);
    top[0]->ReshapeLike(*bottom[1]);
  } else {
    CHECK_EQ(bottom.size(), 1);
    feat_len = bottom[0]->count()/(T*N);
    top[0]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int gap_per_T = N * feat_len;
  if (bottom.size() == 2) {
    for (int j = 0; j < N; ++j) {
      const Dtype * cont_seq = bottom[0]->cpu_data() + j;
      const Dtype * feature_seq = bottom[1]->cpu_data() + j * feat_len;
      Dtype * feature_seq_reverse = top[0]->mutable_cpu_data() + j * feat_len;
      int cont_end = 0;
      int tt = -1;
      while (cont_end < T && *cont_seq != 0) {
        // get next sequence's length
        tt = -(int)*cont_seq;
        int start = cont_end;
        do {
          ++cont_end;
          cont_seq += N;
        } while ( cont_end < T && *cont_seq == 1 );
        CHECK_EQ(tt, cont_end - start) << "sequence length should be equal";
        const Dtype *feature_seq_end = feature_seq + tt * gap_per_T;
        for (int l = start; l < cont_end; ++l) {
          feature_seq_end -= gap_per_T;
          caffe_copy( feat_len, feature_seq_end, feature_seq_reverse);
          feature_seq_reverse += gap_per_T;
        }
        feature_seq += tt * gap_per_T;
      }
    }
  } else {
    CHECK_EQ(bottom.size(), 1);
    Dtype *feature_start = top[0]->mutable_cpu_data();
    const Dtype *feature_seq_end = bottom[0]->cpu_data() + (T-1) * gap_per_T;
    for (int t = 0; t < bottom[0]->count(); t+=gap_per_T)
      caffe_copy(gap_per_T, feature_seq_end - t, feature_start + t);
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int gap_per_T = N * feat_len;
  if (bottom.size() == 2) {
    if (!propagate_down[1]) return;
    caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
    for (int j = 0; j < N; ++j) {
      const Dtype * cont_seq = bottom[0]->cpu_data() + j;
      Dtype * feature_seq = bottom[1]->mutable_cpu_diff() + j * feat_len;
      const Dtype * feature_seq_reverse = top[0]->cpu_diff() + j * feat_len;
      int cont_end = 0;
      int tt = -1;
      while (cont_end < T && *cont_seq != 0) {
        // get next sequence's length
        tt = -(int)*cont_seq;
        int start = cont_end;
        do {
          ++cont_end;
          cont_seq += N;
        } while ( cont_end < T && *cont_seq == 1 );
        CHECK_EQ(tt, cont_end - start) << "sequence length should be equal";
        Dtype *feature_seq_end = feature_seq + tt * gap_per_T;
        for (int l = start; l < cont_end; ++l) {
          feature_seq_end -= gap_per_T;
          caffe_copy( feat_len, feature_seq_reverse, feature_seq_end);
          feature_seq_reverse += gap_per_T;
        }
        feature_seq += tt * gap_per_T;
      }
    }
  } else {
    CHECK_EQ(bottom.size(), 1);
    Dtype *feature_start = bottom[0]->mutable_cpu_diff();
    const Dtype *feature_seq_end = top[0]->cpu_diff() + (T-1) * gap_per_T;
    for (int t = 0; t < bottom[0]->count(); t+=gap_per_T)
      caffe_copy(gap_per_T, feature_seq_end - t, feature_start + t);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ReverseLayer);
#endif

INSTANTIATE_CLASS(ReverseLayer);
REGISTER_LAYER_CLASS(Reverse);

}  // namespace caffe
