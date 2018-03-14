#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/deform_conv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class DeformConvLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DeformConvLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 2, 7, 8)),
        blob_bottom_delta_(new Blob<Dtype>(1, 18, 7, 8)),
        blob_top_data_(new Blob<Dtype>()) {
          LOG(INFO) << "test0";
    // fill the values
    for (int i = 0; i < blob_bottom_data_->count(); ++i) {
      blob_bottom_data_->mutable_cpu_data()[i] = caffe_rng_rand() % 100;
    }
    for (int i = 0; i < blob_bottom_delta_->count(); ++i) {
      //blob_bottom_delta_->mutable_cpu_data()[i] = caffe_rng_rand() % 20 - 10 + Dtype(-0.2);
      caffe_rng_gaussian<Dtype>(blob_bottom_delta_->count(), Dtype(0),
        Dtype(0.01), blob_bottom_delta_->mutable_cpu_data());
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_delta_);
    blob_top_vec_.push_back(blob_top_data_);
    LOG(INFO) << "test1";
  }
  virtual ~DeformConvLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_delta_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_delta_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DeformConvLayerTest, TestDtypesAndDevices);

TYPED_TEST(DeformConvLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
LOG(INFO) << "test3";
  LayerParameter layer_param;
  DeformConvParameter* deformconv_param = layer_param.mutable_deformconv_param();
  deformconv_param->set_kernel_w(3);
  deformconv_param->set_kernel_h(3);
deformconv_param->set_scale(1.5);
  DeformConvLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-1, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  LOG(INFO) << "test4";
}
}  // namespace caffe
