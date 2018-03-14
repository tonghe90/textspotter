// ------------------------------------------------------------------
// Written by Tong He
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/deform_conv_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {


template <typename Dtype>
  void bilinear_interpolate(const Dtype* bottom_data, const int height, const int width, Dtype h, 
    Dtype w, Dtype &oh, Dtype &ow, Dtype &maxval){


// deal with cases that inverse elements are out of feature map boundary

  if (h < 0) h = 0;
  if (w < 0) w = 0;
  
  int h_low = (int) h;
  int w_low = (int) w;
  int h_high;
  int w_high;
  
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (Dtype) h_low;
  } else {
    h_high = h_low + 1;
  }
  
  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (Dtype) w_low;
  } else {
    w_high = w_low + 1;
  }
  
  Dtype lh = h - h_low;
  Dtype lw = w - w_low;
  Dtype hh = 1 - lh, hw = 1 - lw;
  // do bilinear interpolation
  Dtype v1 = bottom_data[h_low * width + w_low];
  Dtype v2 = bottom_data[h_low * width + w_high];
  Dtype v3 = bottom_data[h_high * width + w_low];
  Dtype v4 = bottom_data[h_high * width + w_high];
  Dtype w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  
  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
 
  maxval = val;
  oh = h;
  ow = w;
 
  
}








  template <typename Dtype>
  void DeformConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // bottom[0]: conv_features with shape (N * C * H * W)
    // bottom[1]: deform_variation with shape (N * 2*(k_h*k_w) * H * W)
    // param: k_h, k_w
    // top[0]: deformed features with shape (N * C * (H*k_h) * (W*k_w)), which can be followed by a regular conv layer
  
    DeformConvParameter deform_param = this->layer_param_.deformconv_param();

    // CHECK(deform_param.has_kernel_size() || 
    //     (deform_param.has_kernel_h() && deform_param.has_kernel_w()))
    //     << "for non-square filters both kernel_h and kernel_w are required";
    // if (deform_param.has_kernel_size()){
    //     kernel_w_ = kernel_h_ = deform_param.kernel_size();
    // } else{
    //     kernel_w_ = deform_param.kernel_w();
    //     kernel_h_ = deform_param.kernel_h();
    // }
    kernel_w_ = deform_param.kernel_w();
    kernel_h_ = deform_param.kernel_h();
    scale_ = deform_param.scale();
    CHECK_GT(kernel_h_, 0) << "Filter dimensions must be > 0";
    CHECK_GT(kernel_w_, 0) << "Filter dimensions must be > 0";
    CHECK_EQ(kernel_w_*kernel_h_*2, bottom[1]->channels()) 
        << "number of deform offsets must equel to 2*kernel_h * kernel_w (deltaX and deltaY for each point)";
    
  }

  template <typename Dtype>
  void DeformConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channel_ = bottom[0]->channels();
    num_ = bottom[0]->num();
    width_ = bottom[0]->width();
    height_ = bottom[0]->height();

    out_width_ = width_ * kernel_w_;
    out_height_ = height_ * kernel_h_;
    top[0]->Reshape(num_, channel_, out_height_, out_width_);
    buffer_.Reshape(num_, 2, out_height_, out_width_);
    caffe_set(buffer_.count(), Dtype(0.), buffer_.mutable_cpu_data());
    caffe_set(top[0]->count(), Dtype(0.), top[0]->mutable_cpu_data());
  }




  template <typename Dtype>
  void DeformConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* feature_data = bottom[0]->cpu_data();
    const Dtype* offset_data = bottom[1]->cpu_data();
    Dtype* out_data = top[0]->mutable_cpu_data();
    Dtype* buffer_data = buffer_.mutable_cpu_data();
    //LOG(INFO) << "num:" << num_ << " channel:"<<channel_ << " out_height:"<<out_height_<< " out_width:" << out_width_;
   
    for (int n = 0; n < num_; n++){
        for (int c = 0; c < channel_; c++){
            for (int oh = 0; oh < out_height_; oh++){
                for (int ow = 0; ow < out_width_; ow++){
                    int bottom_h = int(oh / kernel_h_);
                    int bottom_w = int(ow / kernel_w_);
                    int offset_bottom_h = oh % kernel_h_ - int(kernel_h_ / 2);
                    int offset_bottom_w = ow % kernel_w_ - int(kernel_w_ / 2); 

                    if ((bottom_h + offset_bottom_h) < 0 || (bottom_h + offset_bottom_h) >= height_ || 
                        (bottom_w + offset_bottom_w) < 0 || (bottom_w + offset_bottom_w) >= width_ ){
                        *(out_data + top[0]->offset(n,c,oh,ow)) = Dtype(0.);
                        
                        continue;

                    }

                    int channel_delta_x = int((oh % kernel_h_) * kernel_w_ + (ow %kernel_w_));
                    int channel_delta_y = channel_delta_x + kernel_h_*kernel_w_;

                    Dtype delta_x = *(offset_data + bottom[1]->offset(n, channel_delta_x, bottom_h, bottom_w)) * scale_;
                    Dtype delta_y = *(offset_data + bottom[1]->offset(n, channel_delta_y, bottom_h, bottom_w)) * scale_;
                    Dtype ih = bottom_h + offset_bottom_h + delta_y;
                    Dtype iw = bottom_w + offset_bottom_w + delta_x;


                    
                    Dtype interp_val = Dtype(0.);
                    Dtype outh = bottom_h;
                    Dtype outw = bottom_w;
                    bilinear_interpolate(feature_data + bottom[0]->offset(n,c), height_, width_, ih, iw, outh, outw, interp_val);
                    *(out_data + top[0]->offset(n,c,oh,ow)) = interp_val;
                    *(buffer_data + buffer_.offset(n,0,oh,ow)) = outh;
                    *(buffer_data + buffer_.offset(n,1,oh,ow)) = outw;
          

                }

            }

        }



    }


  }



template <typename Dtype>
void get_feature_gradient(Dtype top_diff, Dtype h, Dtype w, const int height, const int width, Dtype* bottom_diff){

  if (h < 0) h = 0;
  if (w < 0) w = 0;
  
  int h_low = (int) h;
  int w_low = (int) w;
  int h_high;
  int w_high;

  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (Dtype) h_low;
  } else {
    h_high = h_low + 1;
  }
  
  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (Dtype) w_low;
  } else {
    w_high = w_low + 1;
  }
  
  Dtype lh = h - h_low;
  Dtype lw = w - w_low;
  Dtype hh = 1 - lh, hw = 1 - lw;


  Dtype weight = 0;
  //left top

  
  weight = hh * hw;
  *(bottom_diff + h_low*width + w_low) += weight * top_diff;
  //LOG(INFO) << " hlow:" << h_low << " h_high:" << h_high << " w_low:" << w_low << " w_high:" << w_high;
  
  //right top
  weight = hh * lw;
  *(bottom_diff + h_low*width + w_high) += weight * top_diff;

  //left bottom
  weight = lh * hw;
  *(bottom_diff + h_high*width + w_low) += weight * top_diff;

  //right bottom
  weight = lh * lw;
  *(bottom_diff + h_high*width + w_high) += weight * top_diff;
  
}


template <typename Dtype>
void get_coord_gradient(Dtype top_diff, Dtype scale, Dtype ih, Dtype iw, int height, int width, Dtype* coord_diff_x, Dtype* coord_diff_y, const Dtype* bottom_data){

  if (ih < 0) ih = 0;
  if (iw < 0) iw = 0;
  
  int h_low = (int) ih;
  int w_low = (int) iw;
  int h_high;
  int w_high;
  
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    ih = (Dtype) h_low;
  } else {
    h_high = h_low + 1;
  }
  
  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    iw = (Dtype) w_low;
  } else {
    w_high = w_low + 1;
  }
  
  Dtype th = ih - h_low;
  Dtype lw = iw - w_low;
  Dtype bh = 1 - th, rw = 1 - lw;

  //left top
  Dtype v1 = *(bottom_data + h_low*width + w_low);
  Dtype v2 = *(bottom_data + h_low*width + w_high);
  Dtype v3 = *(bottom_data + h_high*width + w_low);
  Dtype v4 = *(bottom_data + h_high*width + w_high);

  *coord_diff_x += top_diff * scale * (-v1 * bh + v2 * bh - v3 * th + v4 * th);
  *coord_diff_y += top_diff * scale * (-v1 * rw - v2 * lw + v3 * rw + v4 * lw);

}



  template <typename Dtype>
  void DeformConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      const Dtype* feature_data = bottom[0]->cpu_data();
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* feature_diff = bottom[0]->mutable_cpu_diff();
      Dtype* offset_diff = bottom[1]->mutable_cpu_diff();
      const Dtype* buffer_data = buffer_.cpu_data();
      caffe_set(bottom[0]->count(), Dtype(0.), feature_diff);
      caffe_set(bottom[1]->count(), Dtype(0.), offset_diff);

     if (propagate_down[0]){
          for (int n = 0; n < num_; n++){
              for (int c = 0; c < channel_; c++){
                  for (int oh = 0; oh < out_height_; oh++){
                      for(int ow = 0; ow < out_width_; ow++){


                          
                          
                          int bottom_h = int(oh / kernel_h_);
                          int bottom_w = int(ow / kernel_w_);
                          int offset_bottom_h = oh % kernel_h_ - int(kernel_h_ / 2);
                          int offset_bottom_w = ow % kernel_w_ - int(kernel_w_ / 2); 


                          if ((bottom_h + offset_bottom_h) < 0 || (bottom_h + offset_bottom_h) >= height_ || 
                              (bottom_w + offset_bottom_w) < 0 || (bottom_w + offset_bottom_w) >= width_ )
                              continue;

                          Dtype ih = *(buffer_data + buffer_.offset(n,0,oh,ow));
                          Dtype iw = *(buffer_data + buffer_.offset(n,1,oh,ow));
                          Dtype diff = *(top_diff + top[0]->offset(n,c,oh,ow));

                          
                          get_feature_gradient(diff, ih, iw, height_, width_, feature_diff+bottom[0]->offset(n,c));
                          

                          if (propagate_down[1]){
                               int channel_delta_x = int((oh % kernel_h_) * kernel_w_ + (ow %kernel_w_));
                               int channel_delta_y = channel_delta_x + kernel_h_*kernel_w_;
                              
                               get_coord_gradient(diff, scale_, ih, iw, height_, width_, offset_diff+bottom[1]->offset(n,channel_delta_x,bottom_h,bottom_w), 
                                       offset_diff+bottom[1]->offset(n,channel_delta_y,bottom_h,bottom_w), feature_data+bottom[0]->offset(n,c));
                              
                          }

                      }
                  }
              }
          }
      

    }
  }


#ifdef CPU_ONLY
STUB_GPU(DeformConvLayer);
#endif
  INSTANTIATE_CLASS(DeformConvLayer);
  REGISTER_LAYER_CLASS(DeformConv);


}  // namespace caffe
