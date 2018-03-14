#include "caffe/layers/sequence_layers.hpp"
#include <locale>

namespace caffe {

inline int min3(int x, int y, int z);
int edit_distance(const std::vector<int>& s1, const std::vector<int>& s2);

template <typename Dtype>
void AccuracyTextLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_edit_distance = top.size() > 1;
  AccuracyTextParameter param = this->layer_param_.accuracy_text_param();
  if (param.has_dictionary()) {
    std::locale loc( "zh_CN.UTF-8" );
    std::locale::global( loc );
    std::wifstream fin(param.dictionary().c_str());
    CHECK(fin.is_open()) << param.dictionary() << " doesn't exist";
    wchar_t temp = 0;
    dict.push_back(temp);
    while (fin >> temp) dict.push_back(temp);
    fin.close();
  }
}

template <typename Dtype>
void AccuracyTextLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  T = bottom[0]->num();
  N = bottom[0]->channels();
  CHECK_GE(bottom.size(), 3);
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1)
    top[1]->Reshape(top_shape);
}

inline int min3(int x, int y, int z) {
  return std::min(std::min(x, y), z);
}

int edit_distance(const std::vector<int>& word1, const std::vector<int>& word2) {
  const int insert_cost = 1;
  const int delete_cost = 1;
  const int source_len = word1.size();
  const int target_len = word2.size();
  std::vector<std::vector<int> > vec(source_len + 1);
  std::vector<int> ivec(target_len + 1);
  int i = 0, j = 0;
  for (int i = 0; i <= target_len; ++i)
    ivec[i] = i;
  vec[0] = ivec;
  memset(&ivec[1], 0, target_len * sizeof(int));
  for (int i = 1; i <= source_len; ++i) {
    ivec[0] = i;
    vec[i] = ivec;
  }
  for (i = 0; i < target_len; ++i) {
    for (j = 0; j < source_len; ++j) {
      vec[j + 1][i + 1] = min3(vec[j + 1][i] + insert_cost, \
                               vec[j][i + 1] + delete_cost, \
                               vec[j][i] + (word1[j] != word2[i]) );
    }
  }
  return vec[source_len][target_len];
}

#include <boost/locale/encoding_utf.hpp>
using boost::locale::conv::utf_to_utf;
// convert wstring to UTF-8 string
inline std::string wstring_to_utf8 (const std::wstring& str)
{
  return utf_to_utf<char>(str.c_str(), str.c_str() + str.size());
}

template <typename Dtype>
void AccuracyTextLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  float sum_number = 0;
  float total_character = 0;
  float accuracy = 0;
  float edit_dist = 0;
  const Dtype *mask_ptr = bottom.size() == 4 ? bottom[3]->cpu_data() : nullptr;
  
  for (int j = 0; j < N; ++j) {
    if (mask_ptr && mask_ptr[j] == 255) {
      continue;
    }
    const Dtype * output_seq = bottom[0]->cpu_data() + j;
    const Dtype *   cont_seq = bottom[1]->cpu_data() + j;
    const Dtype * target_seq = bottom[2]->cpu_data() + j;
    int cont_end = 0;
    int tt = -1;
    vector< int > target;
    vector< int > output_result;

    while (cont_end < T && *cont_seq != 0) {
      // get next sequence's length
      tt = -(int) * cont_seq;
      int start = cont_end;
      do {
        ++cont_end;
        cont_seq += N;
      } while ( cont_end < T && *cont_seq == 1 );
      CHECK_EQ(tt, cont_end - start) << "sequence length should be equal";
      target.clear();
      output_result.clear();
      for (int l = 0; l < tt && *target_seq != -1; ++l, target_seq += N)
        target.push_back(int(*target_seq));
      for (int l = 0; l < tt && *output_seq != -1; ++l, output_seq += N)
        output_result.push_back(int(*output_seq));
      if (!dict.empty() && target != output_result) {
        std::wstring pred, gt;
        
        for (int l = 0; l < output_result.size(); ++l)
          if (output_result[l] >= dict.size())
            pred += L'#';
          else pred += dict[output_result[l]];
        std::wcout << std::endl;
        for (int l = 0; l < target.size(); ++l)
          if (target[l] >= dict.size())
            gt += L'#';
          else gt += dict[target[l]];
        LOG(INFO) << "pred: " << wstring_to_utf8(pred);
        LOG(INFO) << "g  t: " << wstring_to_utf8(gt);
        LOG(INFO) << "==================================";
        //getchar();
      }
      if (target == output_result)
        ++accuracy;

      if (has_edit_distance)
        edit_dist += edit_distance(target, output_result);
      int L = target.size();
      int L_result = output_result.size();
      ++sum_number;
      total_character += L;
      target_seq += (tt - L) * N;
      output_seq += (tt - L_result) * N;
    }
  }

  top[0]->mutable_cpu_data()[0] = sum_number ? accuracy / sum_number : 0;
  if (has_edit_distance) {
    top[1]->mutable_cpu_data()[0] = 1 - total_character ? edit_dist / total_character : 0;
  }
}

INSTANTIATE_CLASS(AccuracyTextLayer);
REGISTER_LAYER_CLASS(AccuracyText);

}  // namespace caffe
