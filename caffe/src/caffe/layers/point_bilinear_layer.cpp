#include <vector>

#include "caffe/layers/point_bilinear_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
  void bilinear_interpolate(const Dtype* bottom_data, const int height, const int width, Dtype h, 
    Dtype w, Dtype &maxval){


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
  
  //maxval = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  maxval = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

 
  //maxval = val;

}

template <typename Dtype>
void PointBilinearLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PointBilinearParameter point_bilinear_param = this->layer_param_.point_bilinear_param();
  ratio_ = point_bilinear_param.ratio();
   CHECK_GT(ratio_, 0) << "ratio must be > 0";
    //CHECK_LE(ratio_, 1) << "ratio must be <= 1";
  has_id_ = bool(bottom.size() > 2);
}

template <typename Dtype>
void PointBilinearLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //bottom: conv_feature 1*C*H*W
  //bottom: points N*2*1*1
    channel_ = bottom[0]->channels();
    num_ = bottom[0]->num();
    width_ = bottom[0]->width();
    height_ = bottom[0]->height();
    //CHECK_EQ(num_, 1) << "only support batchsize == 1";
    CHECK_EQ(bottom[1]->channels(), 2) << "rois should be N*2*1*1";
    point_num_ = bottom[1]->num();
    top[0]->Reshape(point_num_, channel_, 1, 1);
    //buffer_.Reshape(point_num_,4*channel_,2,1);

}

template <typename Dtype>
void PointBilinearLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   const Dtype* feature_data = bottom[0]->cpu_data();
    const Dtype* roi_data = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* id_data = 0;
    if (has_id_)
      id_data = bottom[2]->cpu_data();
    //Dtype* buffer_data = buffer_.mutable_cpu_data();
  for (int n = 0; n < point_num_; n++){
      int point_id = 0;
      if (has_id_){
         point_id = int(id_data[n]);
         
       }
      int bottom_ind = point_id*channel_*width_*height_;
      for (int c = 0; c < channel_; c++){
      
         
          Dtype cur_x = roi_data[n * 2];
          Dtype cur_y = roi_data[n * 2 + 1];

          Dtype w = cur_x * Dtype(ratio_);
          Dtype h = cur_y * Dtype(ratio_);
          

          int top_index = n * channel_ + c;
          // Dtype interp_val = Dtype(0.);

          int fea_ind = bottom_ind + c*width_*height_;
          bilinear_interpolate(feature_data+fea_ind, height_, width_, h, w, top_data[top_index]);
          
          // int buffer_index =  n * 4 * channel_ * 2 + c * 2;
          // buffer_data[buffer_index] = ow;
          // buffer_data[buffer_index+1] = oh;


      }


  }
  
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
void PointBilinearLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

   
      const Dtype* top_diff = top[0]->cpu_diff();
      const Dtype* roi_data = bottom[1]->cpu_data();
      Dtype* feature_diff = bottom[0]->mutable_cpu_diff();
      Dtype* roi_diff = bottom[1]->mutable_cpu_diff();
      const Dtype* feature_data = bottom[0]->cpu_data();
      caffe_set(bottom[0]->count(), Dtype(0.), feature_diff);
      caffe_set(bottom[1]->count(), Dtype(0.), roi_diff);
      const Dtype* id_data = 0;
      if (has_id_){
        id_data = bottom[2]->cpu_data();
    }
      //Dtype* buffer_data = buffer_.mutable_cpu_data();
     for (int n = 0; n < point_num_; n++){
        int point_id = 0;
        if (has_id_)
          point_id = int(id_data[n]);

        int bottom_index = point_id * channel_ * width_ * height_;
        for (int c = 0; c < channel_; c++){
            
            Dtype cur_x = roi_data[n * 2];
            Dtype cur_y = roi_data[n * 2 + 1];

             // Dtype w = cur_x * Dtype(ratio_);
             // Dtype h = cur_y * Dtype(ratio_);
          

            int bottom_ind = bottom_index + c * height_ * width_;
            //int buffer_index =  n * 4 * channel_ * 2 + c * 2;
            //Dtype iw = buffer_data[buffer_index];
            //Dtype ih = buffer_data[buffer_index+1];
            Dtype iw = cur_x * Dtype(ratio_);
            Dtype ih = cur_y * Dtype(ratio_);
            Dtype diff = top_diff[n*channel_ + c];
            get_feature_gradient(diff, ih, iw, height_, width_, feature_diff+bottom_ind);


            if (propagate_down[1]){
                 int tmp = bottom_index + c*width_*height_;
                 get_coord_gradient(diff, Dtype(1.), ih, iw, height_, width_, roi_diff+n*0, roi_diff+n*1, feature_data+tmp);



            }

        }

     }


}

INSTANTIATE_CLASS(PointBilinearLayer);
REGISTER_LAYER_CLASS(PointBilinear);

}  // namespace caffe
