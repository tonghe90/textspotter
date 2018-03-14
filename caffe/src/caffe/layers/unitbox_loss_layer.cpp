#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/unitbox_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

////////////////////////////////////////////////////////////////////////////////////////
//bottom0: N*4*H*W pred #relative distance to groundtruth: xt, xb, xl, xr
//bottom1: N*4*H*W gt
//top0: loss
////////////////////////////////////////////////////////////////////////////////////////
namespace caffe {



template <typename Dtype>
void UnitboxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    
}


template <typename Dtype>
void UnitboxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Inputs must have the same dimension.";
     num_ = bottom[0]->num();
     width_ = bottom[0]->width();
     height_ = bottom[0]->height();
     channels_ = bottom[0]->channels();
     pos_count_ = 0;
     const Dtype* gt_data = bottom[1]->cpu_data();
     for (int n = 0; n < num_; n++){
        for (int h = 0; h < height_; h++){
            for (int w = 0; w < width_; w++){
                float gt_xt = float(gt_data[((n * channels_ + 0) * height_ + h) * width_ + w]);
                float gt_xb = float(gt_data[((n * channels_ + 1) * height_ + h) * width_ + w]);
                float gt_xl = float(gt_data[((n * channels_ + 2) * height_ + h) * width_ + w]);
                float gt_xr = float(gt_data[((n * channels_ + 3) * height_ + h) * width_ + w]);


                float gt_x = (gt_xt + gt_xb) * (gt_xl + gt_xr); // X_bar

                if (gt_x > 0 && gt_xt>0){
                    pos_count_++;
                }
            }
        }
     }
    //  if (top.size() > 1){
    //     top[1]->Reshape(num_, 1, height_, width_);
    //     caffe_set(top[1]->count(), Dtype(-10.0), top[1]->mutable_cpu_data());
    // }
    //LOG(INFO) << "pos_count:" << pos_count_;

  }



template <typename Dtype>
void UnitboxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //int count = bottom[0]->count();
    const Dtype* pred_data = bottom[0]->cpu_data();
    const Dtype* gt_data = bottom[1]->cpu_data();
    // Dtype* loss_data = 0;
    // if (top.size() > 1){
    //      loss_data = top[1]->mutable_cpu_data();
    // }
    //int count=0;
//    int num = bottom[0]->num();
//    int width = bottom[0]->width();
//    int height = bottom[0]->height();
//    int channels = bottom[0]->channels();
    Dtype loss = 0;
    for (int n = 0; n < num_; n++){
        for (int h = 0; h < height_; h++){
            for (int w = 0; w < width_; w++){

                float gt_xt = float(gt_data[((n * channels_ + 0) * height_ + h) * width_ + w]);
                float gt_xb = float(gt_data[((n * channels_ + 1) * height_ + h) * width_ + w]);
                float gt_xl = float(gt_data[((n * channels_ + 2) * height_ + h) * width_ + w]);
                float gt_xr = float(gt_data[((n * channels_ + 3) * height_ + h) * width_ + w]);
                

                float gt_x = (gt_xt + gt_xb) * (gt_xl + gt_xr); // X_bar

                if (gt_x <= 0 || gt_xt<=0 || gt_xb<=0 || gt_xl<=0 || gt_xr<=0){
                    continue;
                }
                
                
                float pred_xt = std::max(float(pred_data[((n * channels_ + 0) * height_ + h) * width_ + w]), float(1e-8));
                float pred_xb = std::max(float(pred_data[((n * channels_ + 1) * height_ + h) * width_ + w]), float(1e-8));
                float pred_xl = std::max(float(pred_data[((n * channels_ + 2) * height_ + h) * width_ + w]), float(1e-8));
                float pred_xr = std::max(float(pred_data[((n * channels_ + 3) * height_ + h) * width_ + w]), float(1e-8));
                float pred_x = (pred_xt + pred_xb) * (pred_xl + pred_xr); // X

                //LOG(INFO)<< "gt_xt:" << gt_xt;
                //LOG(INFO)<< "gt_xb:" << gt_xb;
                //LOG(INFO)<< "gt_xl:" << gt_xl;
                //LOG(INFO)<< "gt_xr:" << gt_xr;
                //LOG(INFO)<< "pred_xt:" << pred_xt;
                //LOG(INFO)<< "pred_xb:" << pred_xb;
                //LOG(INFO)<< "pred_xl:" << pred_xl;
                //LOG(INFO)<< "pred_xr:" << pred_xr;
                float intersect_h = std::max(std::min(pred_xt, gt_xt) + std::min(pred_xb, gt_xb), float(1e-8));
                float intersect_w = std::max(std::min(pred_xl, gt_xl) + std::min(pred_xr, gt_xr), float(1e-8));
                float intersect_area = intersect_h * intersect_w;
                
                float united_area = pred_x + gt_x - intersect_area;
                //LOG(INFO)<< "intersect:" << intersect_area;
                //LOG(INFO)<< "united:" << united_area;
                float iou = std::max(intersect_area  / united_area, float(1e-8));
                
                CHECK_GT(iou, 0) << "iou must > 0";
                CHECK_LE(iou, 1) << "iou must <= 1";
                loss += -log(iou);
                // if (top.size() > 1){
                //     loss_data[((n * channels_ + 0) * height_ + h) * width_ + w] = -log(iou);
                // }
                //count++;
            }

        }

    }

    top[0]->mutable_cpu_data()[0] = loss / (pos_count_ == 0 ? 1 : pos_count_);


  }



template <typename Dtype>
void UnitboxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (propagate_down[1]) {
        LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }

    if (propagate_down[0]) {

        const Dtype* pred_data = bottom[0]->cpu_data();
        const Dtype* gt_data = bottom[1]->cpu_data();
//        int num = bottom[0]->num();
//        int width = bottom[0]->width();
//        int height = bottom[0]->height();
//        int channels = bottom[0]->channels();
        Dtype* pred_diff = bottom[0]->mutable_cpu_diff();
        caffe_set(bottom[0]->count(), Dtype(0), pred_diff);
        pred_diff = bottom[0]->mutable_cpu_diff();
        Dtype loss_weight = top[0]->cpu_diff()[0];
        //int count=0;
        for (int n = 0; n < num_; n++){
            for (int h = 0; h < height_; h++){
                for (int w = 0; w < width_; w++){
                    float gt_xt = float(gt_data[((n * channels_ + 0) * height_ + h) * width_ + w]);
                    float gt_xb = float(gt_data[((n * channels_ + 1) * height_ + h) * width_ + w]);
                    float gt_xl = float(gt_data[((n * channels_ + 2) * height_ + h) * width_ + w]);
                    float gt_xr = float(gt_data[((n * channels_ + 3) * height_ + h) * width_ + w]);

                    float gt_x = (gt_xt + gt_xb) * (gt_xl + gt_xr); // X_bar

                    if (gt_x <= 0 || gt_xt<=0 || gt_xb<=0 || gt_xl<=0 || gt_xr<=0){
                        continue;
                    }
                    float pred_xt = std::max(float(pred_data[((n * channels_ + 0) * height_ + h) * width_ + w]), float(1e-8));
                    float pred_xb = std::max(float(pred_data[((n * channels_ + 1) * height_ + h) * width_ + w]), float(1e-8));
                    float pred_xl = std::max(float(pred_data[((n * channels_ + 2) * height_ + h) * width_ + w]), float(1e-8));
                    float pred_xr = std::max(float(pred_data[((n * channels_ + 3) * height_ + h) * width_ + w]), float(1e-8));

                    float pred_x = (pred_xt + pred_xb) * (pred_xl + pred_xr); // X


                    float intersect_h = std::max(std::min(pred_xt, gt_xt) + std::min(pred_xb, gt_xb), float(1e-8));
                    float intersect_w = std::max(std::min(pred_xl, gt_xl) + std::min(pred_xr, gt_xr), float(1e-8));
                    
                    float intersect_area = intersect_h * intersect_w;
                    
                    float delta_X_xt = pred_xl + pred_xr; // d(X)/d(xt) = xl+xr
                    float delta_X_xb = pred_xl + pred_xr; // d(X)/d(xb) = xl+xr

                    float delta_X_xl = pred_xt + pred_xb; // d(X)/d(xl) = xt+xb
                    float delta_X_xr = pred_xt + pred_xb; // d(X)/d(xr) = xt+xb

                    float delta_I_xt = pred_xt < gt_xt ? intersect_w : 0.f;
                    float delta_I_xb = pred_xb < gt_xb ? intersect_w : 0.f;

                    float delta_I_xl = pred_xl < gt_xl ? intersect_h : 0.f;
                    float delta_I_xr = pred_xr < gt_xr ? intersect_h : 0.f;

                    float united_area = pred_x + gt_x - intersect_area;

                    float alpha = loss_weight * 1.0 / united_area;
                    float beta  = loss_weight * (united_area + intersect_area) / (united_area * intersect_area);
//                    LOG(INFO) << "intersect_area:" << intersect_area;
//                    LOG(INFO) << "united_area:" << united_area;
//
//                    LOG(INFO) << "alpha:" << alpha;
//                    LOG(INFO) << "beta:" << beta;
//                    LOG(INFO) << "delta_X_xt:" << delta_X_xt;
//                    LOG(INFO) << "delta_I_xt:" << delta_I_xt;
                    pred_diff[((n * channels_ + 0) * height_ + h) * width_ + w] = alpha * delta_X_xt - beta * delta_I_xt; //d(Loss)/d(xt)
                    pred_diff[((n * channels_ + 1) * height_ + h) * width_ + w] = alpha * delta_X_xb - beta * delta_I_xb; //d(Loss)/d(xb)
                    pred_diff[((n * channels_ + 2) * height_ + h) * width_ + w] = alpha * delta_X_xl - beta * delta_I_xl; //d(Loss)/d(xl)
                    pred_diff[((n * channels_ + 3) * height_ + h) * width_ + w] = alpha * delta_X_xr - beta * delta_I_xr; //d(Loss)/d(xl)

                    //count++;

                }
            }
        }
        caffe_scal(bottom[0]->count(), pos_count_==0?1:Dtype(1.0)/Dtype(pos_count_), bottom[0]->mutable_cpu_diff());


    } // end of if

  }// end of bachward_cpu
//#ifdef CPU_ONLY
//STUB_GPU(UnitboxLossLayer);
//#endif


INSTANTIATE_CLASS(UnitboxLossLayer);
REGISTER_LAYER_CLASS(UnitboxLoss);
}