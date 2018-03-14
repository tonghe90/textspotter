#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cosinangle_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"



namespace caffe {


template <typename Dtype>
__global__ void cos_forward(const int nthreads, const int width, const int height,
    const int channels, const Dtype* pred_data, const Dtype* gt_data,
    Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  // (n, c, h, w) is an element in the pooled output
    int w = index % width;
    int h = (index / width) % height;
    int c = 0;
    int n = index / width / height;

    float pred_val = float(pred_data[((n * channels + c) * height + h) * width + w]);
    float gt_val = float(gt_data[((n * channels + c) * height + h) * width + w]);
    if (gt_val > -5){
        loss_data[index] = 1 - cos(pred_val - gt_val);
    }
    else{
        loss_data[index] = 0;
    }


   }



 }



template <typename Dtype>
void CosinangleLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        const Dtype* pred_data = bottom[0]->gpu_data();
        const Dtype* gt_data = bottom[1]->gpu_data();
        int count = bottom[0]->count();

        if (top.size() < 2){
            Dtype* loss_data = bottom[0]->mutable_gpu_diff();

            caffe_set(count, Dtype(0), bottom[0]->mutable_cpu_diff());
            cos_forward<Dtype><<<CAFFE_GET_BLOCKS(count),
                    CAFFE_CUDA_NUM_THREADS>>>(count, width_, height_, channels_,
                    pred_data, gt_data, loss_data);
            


            Dtype loss;
            caffe_gpu_asum(count, loss_data, &loss);

            top[0]->mutable_cpu_data()[0] = loss / (int(pos_count_) == 0 ? Dtype(1) : Dtype(pos_count_));
        } else {
            Dtype* loss_data = top[1]->mutable_gpu_data();
            caffe_set(count, Dtype(0), bottom[0]->mutable_cpu_diff());
            cos_forward<Dtype><<<CAFFE_GET_BLOCKS(count),
                    CAFFE_CUDA_NUM_THREADS>>>(count, width_, height_, channels_,
                    pred_data, gt_data, loss_data);
            
            Dtype loss;
            caffe_gpu_asum(count, loss_data, &loss);

            top[0]->mutable_cpu_data()[0] = loss / (int(pos_count_) == 0 ? Dtype(1) : Dtype(pos_count_));

        }



  }



template <typename Dtype>
__global__ void cos_backward(const int nthreads, const int width, const int height,
    const int channels, const Dtype* pred_data, const Dtype* gt_data,
    Dtype* pred_diff, Dtype loss_weight) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = 0;
    int n = index / width / height;

    float gt_val = float(gt_data[((n * channels + c) * height + h) * width + w]);
    float pred_val = float(pred_data[((n * channels + c) * height + h) * width + w]);

    if (gt_val > -5){
        pred_diff[((n * channels + 0) * height + h) * width + w] = loss_weight * sin(pred_val-gt_val);
    }
    else{
        pred_diff[((n * channels + 0) * height + h) * width + w] = 0.f;
    }


  }




}

template <typename Dtype>
void CosinangleLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
       LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
    }

    if (propagate_down[0]) {
        const Dtype* pred_data = bottom[0]->gpu_data();
        const Dtype* gt_data = bottom[1]->gpu_data();
        int count = bottom[0]->count();
        caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
        Dtype* pred_diff = bottom[0]->mutable_gpu_diff();

        cos_backward<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(count, width_, height_, channels_,
            pred_data, gt_data, pred_diff, top[0]->cpu_diff()[0]);

        caffe_gpu_scal(bottom[0]->count(), Dtype(1.0)/(int(pos_count_) == 0 ? Dtype(1) : Dtype(pos_count_)), bottom[0]->mutable_gpu_diff());



    }

  }

INSTANTIATE_LAYER_GPU_FUNCS(CosinangleLossLayer);

}



