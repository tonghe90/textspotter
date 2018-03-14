// ------------------------------------------------------------------
// Written by Tong He
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/deform_conv_layer.hpp"
#include "caffe/util/gpu_util.cuh"
using std::max;
using std::min;

namespace caffe {


 template <typename Dtype>
  __global__ void DeformConvForwardAtomic(
  const int nthreads,
  Dtype scale, 
  const int num,
  const int channel,
  const int kernel_h,
  const int kernel_w,
  const int height,
  const int width,
  const int out_height,
  const int out_width,
  const Dtype* feature_data,
  const Dtype* offset_data,
  Dtype* buffer_data,
  Dtype* out_data){
    CUDA_KERNEL_LOOP(index, nthreads){
      int ow = index % out_width;
      int oh = (index / out_width) % out_height;
      int c = (index / out_width / out_height) % channel;
      int n = index / out_width / out_height / channel;
      int bottom_h = int(oh / kernel_h);
      int bottom_w = int(ow / kernel_w);

      int offset_bottom_h = oh % kernel_h - int(kernel_h / 2);
      int offset_bottom_w = ow % kernel_w - int(kernel_w / 2);

      
      int buffer_offset_y = n*2*out_height*out_width + oh*out_width + ow; // c=0
      int buffer_offset_x = n*2*out_height*out_width + out_height*out_width + oh*out_width + ow; //c=1

      if ((bottom_h + offset_bottom_h) < 0 || (bottom_h + offset_bottom_h) >= height || 
           (bottom_w + offset_bottom_w) < 0 || (bottom_w + offset_bottom_w) >= width ){
            *(out_data + index) = Dtype(0.);
            *(buffer_data + buffer_offset_x) = Dtype(-1);
            *(buffer_data + buffer_offset_y) = Dtype(-1);
        } else {

          int channel_delta_x = int((oh % kernel_h) * kernel_w + (ow %kernel_w));
          int channel_delta_y = channel_delta_x + kernel_h*kernel_w;

          int offset_channel = kernel_h * kernel_w * 2;
          int offsetx_offset = n*offset_channel*height*width + channel_delta_x*height*width + bottom_h*width + bottom_w;
          int offsety_offset = n*offset_channel*height*width + channel_delta_y*height*width + bottom_h*width + bottom_w;
          Dtype delta_x = *(offset_data + offsetx_offset) * scale;
          Dtype delta_y = *(offset_data + offsety_offset) * scale;

          Dtype ih = bottom_h + offset_bottom_h + delta_y;
          Dtype iw = bottom_w + offset_bottom_w + delta_x;

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


        Dtype lh = ih - h_low;
        Dtype lw = iw - w_low;
        Dtype hh = 1 - lh;
        Dtype hw = 1 - lw;

        int feature_offset = n*channel*height*width + c*height*width;
        Dtype v1 = feature_data[feature_offset + h_low * width + w_low];
        Dtype v2 = feature_data[feature_offset + h_low * width + w_high];
        Dtype v3 = feature_data[feature_offset + h_high * width + w_low];
        Dtype v4 = feature_data[feature_offset + h_high * width + w_high];
        Dtype w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;


        *(out_data + index) = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;

        *(buffer_data + buffer_offset_x) = iw;
      *(buffer_data + buffer_offset_y) = ih;

      }
    }
    



  }


template <typename Dtype>
  void DeformConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // Forward_cpu(bottom, top);
    // return;
    const Dtype* feature_data = bottom[0]->gpu_data();
    const Dtype* offset_data = bottom[1]->gpu_data();
    Dtype* out_data = top[0]->mutable_gpu_data();
    Dtype* buffer_data = buffer_.mutable_gpu_data();

    int count = top[0]->count();

    DeformConvForwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, scale_, num_, channel_, kernel_h_, kernel_w_, height_, width_, out_height_, out_width_,
        feature_data, offset_data, buffer_data, out_data);
    
    CUDA_POST_KERNEL_CHECK;
}







 template <typename Dtype>
  __global__ void DeformConvBackwardAtomic(
  const int nthreads,
  Dtype scale,
  const int num,
  const int channel,
  const int kernel_h,
  const int kernel_w,
  const int height,
  const int width,
  const int out_height,
  const int out_width,
  const Dtype* feature_data,
  const Dtype* top_diff,
  const Dtype* buffer_data,
  Dtype* feature_diff,
  Dtype* coord_diff,
  bool propagate_feature,
  bool propagate_coord){
    CUDA_KERNEL_LOOP(index, nthreads) {
      //if (propagate_feature){
        int ow = index % out_width;
        int oh = (index / out_width) % out_height;
        int c = (index / out_width / out_height) % channel;
        int n = index / out_width / out_height / channel;
        int bottom_h = int(oh / kernel_h);
        int bottom_w = int(ow / kernel_w);

        int buffer_offset_y = n*2*out_height*out_width + oh*out_width + ow; // c=0
        int buffer_offset_x = n*2*out_height*out_width + out_height*out_width + oh*out_width + ow; //c=1
        int offset_bottom_h = oh % kernel_h - int(kernel_h / 2);
        int offset_bottom_w = ow % kernel_w - int(kernel_w / 2);

        // LOG(ERROR)<<index<<", "<<*(top_diff+index);
        if ((bottom_h + offset_bottom_h) < 0 || (bottom_h + offset_bottom_h) >= height || 
           (bottom_w + offset_bottom_w) < 0 || (bottom_w + offset_bottom_w) >= width ){
          // LOG(ERROR)<<"in if";
          // CHECK(false);
          // LOG(ERROR)<<index;
          // LOG(ERROR)<<(bottom_h + offset_bottom_h)<<", "<<(bottom_w + offset_bottom_w);
          // CHECK(false);
          continue;
        }

        // LOG(ERROR)<<"out of if";
        // CHECK(false);
        // if(*(top_diff + index)) {
        //   assert(false);
        // }
        Dtype iw = *(buffer_data + buffer_offset_x);
        Dtype ih = *(buffer_data + buffer_offset_y);


        Dtype diff = *(top_diff + index);

          // backward feature
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
        
        Dtype lh = ih - h_low;
        Dtype lw = iw - w_low;
        Dtype hh = 1 - lh, hw = 1 - lw;

        Dtype weight = 0;

        int feature_offset = n*channel*height*width + c*height*width;
        //left top
        weight = hh * hw;
        caffe_gpu_atomic_add(weight*diff, feature_diff + feature_offset + h_low*width + w_low);
        //*(feature_diff + feature_offset + h_low*width + w_low) += weight * diff;

        //right top
        weight = hh * lw;
        caffe_gpu_atomic_add(weight*diff, feature_diff + feature_offset + h_low*width + w_high);
        //*(feature_diff + feature_offset + h_low*width + w_high) += weight * diff;

        //left bottom
        weight = lh * hw;
        caffe_gpu_atomic_add(weight*diff, feature_diff + feature_offset + h_high*width + w_low);
        //*(feature_diff + feature_offset + h_high*width + w_low) += weight * diff;

        //right bottom
        weight = lh * lw;
        caffe_gpu_atomic_add(weight*diff, feature_diff+ feature_offset + h_high*width + w_high);
        //*(feature_diff+ feature_offset + h_high*width + w_high) += weight * diff;


        //if (propagate_coord){
             Dtype th = ih - h_low;

            Dtype bh = 1 - th, rw = 1 - lw;

          //left top
          Dtype v1 = *(feature_data + feature_offset + h_low*width + w_low);
          Dtype v2 = *(feature_data + feature_offset + h_low*width + w_high);
          Dtype v3 = *(feature_data + feature_offset + h_high*width + w_low);
          Dtype v4 = *(feature_data + feature_offset + h_high*width + w_high);
          int channel_delta_x = int((oh % kernel_h) * kernel_w + (ow %kernel_w));
         int channel_delta_y = channel_delta_x + kernel_h*kernel_w;

          int offset_channel = kernel_h * kernel_w * 2;
          int offsetx_offset = n*offset_channel*height*width + channel_delta_x*height*width + bottom_h*width + bottom_w;
         int offsety_offset = n*offset_channel*height*width + channel_delta_y*height*width + bottom_h*width + bottom_w;

          // *(coord_diff + offsetx_offset) += diff * (-v1 * bh + v2 * bh - v3 * th + v4 * th);
          // *(coord_diff + offsety_offset) += diff * (-v1 * rw - v2 * lw + v3 * rw + v4 * lw);
          caffe_gpu_atomic_add(diff * scale * (-v1 * bh + v2 * bh - v3 * th + v4 * th), coord_diff + offsetx_offset);
          caffe_gpu_atomic_add(diff * scale * (-v1 * rw - v2 * lw + v3 * rw + v4 * lw), coord_diff + offsety_offset);

        //}
    //}// end if
  }//end kernel loop
}










template <typename Dtype>
  void DeformConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* feature_data = bottom[0]->gpu_data();

    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* feature_diff = bottom[0]->mutable_gpu_diff();
    Dtype* offset_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* buffer_data = buffer_.gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(bottom[0]->count(), Dtype(0.), feature_diff);
    caffe_gpu_set(bottom[1]->count(), Dtype(0.), offset_diff);

  DeformConvBackwardAtomic<Dtype><< <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, scale_, num_, channel_, kernel_h_, kernel_w_, height_, width_, out_height_, out_width_,
        feature_data, top_diff, buffer_data, feature_diff, offset_diff, propagate_down[0], propagate_down[1]);
  //  CUDA_POST_KERNEL_CHECK;
  // LOG(ERROR)<<caffe_cpu_asum(bottom[0]->count(), bottom[0]->cpu_diff());
  // CHECK(false);

}
INSTANTIATE_LAYER_GPU_FUNCS(DeformConvLayer);

} // namespace caffe
