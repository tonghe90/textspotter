#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/point_bilinear_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class PointBilinearLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PointBilinearLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(2, 5, 8, 8)),
        blob_bottom_pts_(new Blob<Dtype>(10,2,1,1)),
        blob_bottom_pts_id_(new Blob<Dtype>(10,1,1,1)),
        blob_top_data_(new Blob<Dtype>()) {
    // fill the values
    
    //for (int i = 0; i < blob_bottom_data_->count(); ++i) {
    //  blob_bottom_data_->mutable_cpu_data()[i] = caffe_rng_rand() % 20 - 10 + Dtype(-0.2);
    //}
    caffe_rng_gaussian<Dtype>(blob_bottom_data_->count(), Dtype(1),
        Dtype(0.01), blob_bottom_data_->mutable_cpu_data());

    caffe_rng_gaussian<Dtype>(blob_bottom_pts_->count(), Dtype(3),
        Dtype(5), blob_bottom_pts_->mutable_cpu_data());

    for (int i = 0; i < blob_bottom_pts_id_->count(); ++i) {
      blob_bottom_pts_id_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;
    }

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_pts_);
    blob_bottom_vec_.push_back(blob_bottom_pts_id_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~PointBilinearLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_pts_;
    delete blob_bottom_pts_id_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_pts_;
  Blob<Dtype>* const blob_bottom_pts_id_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PointBilinearLayerTest, TestDtypesAndDevices);
TYPED_TEST(PointBilinearLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  PointBilinearParameter* point_bilinear_param = layer_param.mutable_point_bilinear_param();
  point_bilinear_param->set_ratio(0.25);

  PointBilinearLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-1, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
}  // namespace caffe
