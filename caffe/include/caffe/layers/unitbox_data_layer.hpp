#ifndef CAFFE_UNITBOX_DATA_LAYER_HPP_
#define CAFFE_UNITBOX_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <tr1/unordered_map>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

//Addef 2017/02/21 by liuxuebo to implement "EAST: An Efficient and Accurate Scene Text Detector"
template <typename Dtype>
class UnitBoxDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit UnitBoxDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~UnitBoxDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "UnitBoxData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }
  virtual inline bool AutoTopBlobs() const { return true; }

  class Line {
    public:
      Line (std::string x, vector<float> y, vector<float> z, vector<string> s) {
        im_name = x;
        bbox = y;
        dont_care_bbox = z;
        gt_text = s;
      }
      std::string im_name;
      vector<float> bbox;
      vector<float> dont_care_bbox;
      vector<string> gt_text;
  };
 protected:
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

 protected:
  Blob<Dtype> transformed_label_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  vector<Line> lines_;
  int lines_id_;
  int img_height_, img_width_, img_channels_;
  int mask_height_, mask_width_;
  bool has_mean_value_, recog_;
  vector<float> mean_values_;
  std::tr1::unordered_map<wchar_t, int> dict;
};

}  // namespace caffe

#endif  // CAFFE_UNITBOX_DATA_LAYER_HPP_
