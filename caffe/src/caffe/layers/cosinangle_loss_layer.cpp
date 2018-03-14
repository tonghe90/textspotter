#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/cosinangle_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

////////////////////////////////////////////////////////////////////////////////////////
//bottom0: N*1*H*W pred #relative distance to groundtruth: xt, xb, xl, xr
//bottom1: N*1*H*W gt
//top0: loss
////////////////////////////////////////////////////////////////////////////////////////
namespace caffe {


template <typename Dtype>
void CosinangleLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
LossLayer<Dtype>::LayerSetUp(bottom, top);
}


template <typename Dtype>
void CosinangleLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::Reshape(bottom, top);

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  width_ = bottom[0]->width();
  height_ = bottom[0]->height();
  count_ = bottom[0]->count();
  CHECK_EQ(count_, bottom[1]->count())
      << "Inputs must have the same dimension.";
  CHECK_EQ(num_, bottom[1]->num())
      << "Inputs must have the same num.";
  CHECK_EQ(channels_, bottom[1]->channels())
      << "Inputs must have the same channels.";
  CHECK_EQ(1, bottom[1]->channels())
      << "The channels must eq 1.";
  CHECK_EQ(width_, bottom[1]->width())
      << "Inputs must have the same width.";
  CHECK_EQ(height_, bottom[1]->height())
      << "Inputs must have the same height.";


    pos_count_=0;
    const Dtype* gt_data = bottom[1]->cpu_data();
  for (int n = 0; n < num_; n++){
      for (int h = 0; h < height_; h++){
          for (int w = 0; w < width_; w++){
              float gt_val = float(gt_data[((n * channels_ + 0) * height_ + h) * width_ + w]);

              if (gt_val > -5){
                  pos_count_ ++;
              }

            }

        }

    }

    



  }



template <typename Dtype>
void CosinangleLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* pred_data = bottom[0]->cpu_data();
    const Dtype* gt_data = bottom[1]->cpu_data();
    
    Dtype loss = 0;
    for (int n = 0; n < num_; n++){
      for (int h = 0; h < height_; h++){
          for (int w = 0; w < width_; w++){
              float pred_val = float(pred_data[((n * channels_ + 0) * height_ + h) * width_ + w]);
              float gt_val = float(gt_data[((n * channels_ + 0) * height_ + h) * width_ + w]);
              //std::cout<< "pred:" <<pred_val<<std::endl;
              //std::cout<< "gt:" << gt_val << std::endl;
              if (gt_val > -5){
                  loss += 1 - cos(pred_val - gt_val);

                  
              }


            }

        }

    }
    Dtype loss_weight = top[0]->cpu_diff()[0];
    top[0]->mutable_cpu_data()[0] = loss_weight * loss / (pos_count_ == 0 ? 1 : pos_count_);

  }



template <typename Dtype>
void CosinangleLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
        const Dtype* pred_data = bottom[0]->cpu_data();
        const Dtype* gt_data = bottom[1]->cpu_data();

        Dtype* pred_diff = bottom[0]->mutable_cpu_diff();
        caffe_set(bottom[0]->count(), Dtype(0), pred_diff);
        pred_diff = bottom[0]->mutable_cpu_diff();
        Dtype loss_weight = top[0]->cpu_diff()[0];
        //std::cout<< "!!!!!loss_weight:"<<loss_weight;
        for (int n = 0; n < num_; n++){
            for (int h = 0; h < height_; h++){
                for (int w = 0; w < width_; w++){


                    float gt_val = float(gt_data[((n * channels_ + 0) * height_ + h) * width_ + w]);
                    float pred_val = float(pred_data[((n * channels_ + 0) * height_ + h) * width_ + w]);
                    //std::cout<< "!!!!!pred_diff:" <<sin(pred_val-gt_val)<<std::endl;
                    if (gt_val > -5){
                        pred_diff[((n * channels_ + 0) * height_ + h) * width_ + w] = loss_weight * sin(pred_val-gt_val);

                    }

                }
            }
        }

        caffe_scal(bottom[0]->count(), pos_count_==0?1:Dtype(1.0)/Dtype(pos_count_), bottom[0]->mutable_cpu_diff());

    }

  }// end of bachward_cpu
#ifdef CPU_ONLY
STUB_GPU(CosinangleLossLayer);
#endif


INSTANTIATE_CLASS(CosinangleLossLayer);
REGISTER_LAYER_CLASS(CosinangleLoss);
}