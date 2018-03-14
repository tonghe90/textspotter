#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/attention_lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void AttLstmLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_0";
  (*names)[1] = "c_0";
}

template <typename Dtype>
void AttLstmLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_" + format_int(this->T_);
  (*names)[1] = "c_T";
}

template <typename Dtype>
void AttLstmLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  const int num_blobs = 2;
  shapes->resize(num_blobs);
  for (int i = 0; i < num_blobs; ++i) {
    (*shapes)[i].Clear();
    (*shapes)[i].add_dim(1);  // a single timestep
    (*shapes)[i].add_dim(this->N_);
    (*shapes)[i].add_dim(num_output);
  }
}

template <typename Dtype>
void AttLstmLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h";
  (*names)[1] = "mask";
}


template <typename Dtype>
void AttLstmLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  const FillerParameter& weight_filler =
      this->layer_param_.recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.recurrent_param().bias_filler();

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter hidden_param;
  hidden_param.set_type("InnerProduct");
  hidden_param.mutable_inner_product_param()->set_num_output(num_output * 4);
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  hidden_param.mutable_inner_product_param()->set_axis(2);
  hidden_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
  biased_hidden_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  LayerParameter slice_param;
  slice_param.set_type("Slice");
  slice_param.mutable_slice_param()->set_axis(0);

  
  LayerParameter attention_param;  
  attention_param.set_type("InnerProduct");  
  attention_param.mutable_inner_product_param()->set_num_output(num_output);  
  attention_param.mutable_inner_product_param()->set_bias_term(false);  
  attention_param.mutable_inner_product_param()->set_axis(2);  
  attention_param.mutable_inner_product_param()->  
      mutable_weight_filler()->CopyFrom(weight_filler);  
  
  LayerParameter biased_attention_param(attention_param);  
  biased_attention_param.mutable_inner_product_param()->set_bias_term(true);  
  biased_attention_param.mutable_inner_product_param()->  
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter transpose_param;  
  transpose_param.set_type("Transpose");  

  LayerParameter bias_layer_param;  
  bias_layer_param.set_type("Bias"); 


  LayerParameter softmax_param;  
  softmax_param.set_type("Softmax");  
  softmax_param.mutable_softmax_param()->set_axis(-1);  


  LayerParameter reshape_layer_param;  
  reshape_layer_param.set_type("Reshape"); 

  LayerParameter scale_param;  
  scale_param.set_type("Scale");  

  LayerParameter axis_sum_param;  
  axis_sum_param.set_type("Sum");  

  LayerParameter split_param;
  split_param.set_type("Split");






  vector<BlobShape> input_shapes;
  RecurrentInputShapes(&input_shapes);
  CHECK_EQ(2, input_shapes.size());
  LayerParameter* input_layer_param = net_param->add_layer();
  input_layer_param->set_type("Input");
  InputParameter* input_param = input_layer_param->mutable_input_param();

  input_layer_param->add_top("c_0");
  input_param->add_shape()->CopyFrom(input_shapes[0]);

  input_layer_param->add_top("h_0");
  input_param->add_shape()->CopyFrom(input_shapes[1]);

  LayerParameter* cont_slice_param = net_param->add_layer();
  cont_slice_param->CopyFrom(slice_param);
  cont_slice_param->set_name("cont_slice");
  cont_slice_param->add_bottom("cont");
  cont_slice_param->mutable_slice_param()->set_axis(0);


  LayerParameter* x_slice_param = net_param->add_layer();  
  x_slice_param->CopyFrom(slice_param);  
  x_slice_param->set_name("x_slice");  
  x_slice_param->add_bottom("x"); 

  LayerParameter output_concat_layer;  
  output_concat_layer.set_name("h_concat");  
  output_concat_layer.set_type("Concat");  
  output_concat_layer.add_top("h");  
  output_concat_layer.mutable_concat_param()->set_axis(0);  
  

  LayerParameter output_m_layer;  
  output_m_layer.set_name("m_concat");  
  output_m_layer.set_type("Concat");  
  output_m_layer.add_top("mask");  
  output_m_layer.mutable_concat_param()->set_axis(0); 

  
  // e(t,j) = Q^T * tanh(W*S(t-1) + V*e(j) + b)
  // a(t,j) = exp(e(j,t)) / sum_j(exp(e(j,t)))
  for (int t = 1; t <= this->T_; ++t) {  
      string tm1s = format_int(t - 1);
      string ts = format_int(t);

      cont_slice_param->add_top("cont_" + ts);  
      x_slice_param->add_top("x_" + ts); 


      // Add layers to flush the hidden state when beginning a new
      // sequence, as indicated by cont_t.
      //     h_conted_{t-1} := cont_t * h_{t-1}
      //
      // Normally, cont_t is binary (i.e., 0 or 1), so:
      //     h_conted_{t-1} := h_{t-1} if cont_t == 1
      //                       0   otherwise
      {
        LayerParameter* cont_h_param = net_param->add_layer();
        cont_h_param->CopyFrom(scale_param);
        cont_h_param->mutable_scale_param()->set_axis(0);
        cont_h_param->set_name("h_conted_" + tm1s);
        cont_h_param->add_bottom("h_" + tm1s);
        cont_h_param->add_bottom("cont_" + ts);
        cont_h_param->add_top("h_conted_" + tm1s);

      }

      // W * S(t-1) [1*N*out_num]
      {  
        //difference 0
        LayerParameter* att_m_param = net_param->add_layer();  
        att_m_param->CopyFrom(biased_attention_param);  
        att_m_param->set_name("st_1"+tm1s);
        att_m_param->add_param()->set_name("W_s");
        att_m_param->add_param()->set_name("b_s");
        att_m_param->set_name("att_m_" + tm1s);  
        att_m_param->add_bottom("h_conted_" + tm1s);  
        att_m_param->add_top("m_" + tm1s); //h(t-1) ==> m(t-1) [1*N*out_num]
        att_m_param->add_propagate_down(true);
      }    

      // V * e(j)
      {
        
          // 1*N*C*Et ==> 1*N*Et*C
          LayerParameter* transpose_x_param = net_param->add_layer();
          transpose_x_param->CopyFrom(transpose_param);
          transpose_x_param->set_name("transpose_x_a_" + ts);
          transpose_x_param->mutable_transpose_param()->add_dim(0);
          transpose_x_param->mutable_transpose_param()->add_dim(1);
          transpose_x_param->mutable_transpose_param()->add_dim(3);
          transpose_x_param->mutable_transpose_param()->add_dim(2);
          transpose_x_param->add_bottom("x_" + ts);
          transpose_x_param->add_top("x_t_" + ts);
          transpose_x_param->add_propagate_down(true);

          // difference_1
          //1*N*Et*C ==> 1*N*Et*out_num
          LayerParameter* att_x_param = net_param->add_layer();
          att_x_param->CopyFrom(biased_attention_param);
          att_x_param->set_name("f_encoder_" + ts);
          att_x_param->mutable_inner_product_param()->set_axis(3); 
          att_x_param->add_param()->set_name("W_f_ej");
          att_x_param->add_param()->set_name("b_f_ej");
          att_x_param->add_bottom("x_t_" + ts);  
          att_x_param->add_top("m_x_" + ts);
          att_x_param->add_propagate_down(true);  


          //1*N*Et*out_num ==> Et*1*N*out_num
          LayerParameter* transpose_x_a_p_param = net_param->add_layer();  
          transpose_x_a_p_param->CopyFrom(transpose_param);
          transpose_x_a_p_param->set_name("transpose_x_a_t_" + ts);
          transpose_x_a_p_param->mutable_transpose_param()->add_dim(2);
          transpose_x_a_p_param->mutable_transpose_param()->add_dim(0);
          transpose_x_a_p_param->mutable_transpose_param()->add_dim(1);
          transpose_x_a_p_param->mutable_transpose_param()->add_dim(3);
          transpose_x_a_p_param->add_bottom("m_x_" + ts);
          transpose_x_a_p_param->add_top("m_x_t_" + ts);

          transpose_x_a_p_param->add_propagate_down(true); 

      }


      // W*S(t-1) + V*e(j) + b
      // bottom[0]: m_x_a_t_1: Et*1*N*out_num
      // bottom[1]: m_t-1: 1*N*out_num
      // top[0]: m_input_t_1: Et*1*N*out_num
      {  
          LayerParameter* m_sum_layer = net_param->add_layer();  
          m_sum_layer->CopyFrom(bias_layer_param);
          m_sum_layer->mutable_bias_param()->set_axis(1);  
          m_sum_layer->set_name("mask_input_" + ts);  
          m_sum_layer->add_bottom("m_x_t_" + ts);  
          m_sum_layer->add_bottom("m_" + tm1s);  
          m_sum_layer->add_top("m_input_" + ts); 
          m_sum_layer->add_propagate_down(true); 
          m_sum_layer->add_propagate_down(true);  
      }  

      // Et*1*N*out_num ==> Et*1*N*1
      
      {
          // difference 2
          LayerParameter* att_x_ap_param = net_param->add_layer();  
        att_x_ap_param->CopyFrom(attention_param);  
        att_x_ap_param->set_name("att_x_ap_" + ts);  
        att_x_ap_param->mutable_inner_product_param()->set_axis(3);  
        att_x_ap_param->mutable_inner_product_param()->set_num_output(1);  
        att_x_ap_param->add_param()->set_name("W_att_ej"+ts);
        //att_x_ap_param->add_param()->set_name("b_att_ej"+ts);
        att_x_ap_param->add_bottom("m_input_" + ts);  
        att_x_ap_param->add_top("m_x_ap_" + ts);  
        att_x_ap_param->add_propagate_down(true); 


      }

    // Et*1*N*1 ==> 1*N*Et*1
      {
          LayerParameter* transpose_m_param = net_param->add_layer();  
          transpose_m_param->CopyFrom(transpose_param);
          transpose_m_param->set_name("transpose_m_" + ts);
          transpose_m_param->mutable_transpose_param()->add_dim(1);
          transpose_m_param->mutable_transpose_param()->add_dim(2);
          transpose_m_param->mutable_transpose_param()->add_dim(0);
          transpose_m_param->mutable_transpose_param()->add_dim(3);
          transpose_m_param->add_bottom("m_x_ap_" + ts);
          transpose_m_param->add_top("m_f_" + ts);
          transpose_m_param->add_propagate_down(true); 


        LayerParameter* softmax_m_param = net_param->add_layer();  
        softmax_m_param->CopyFrom(softmax_param);  
        softmax_m_param->mutable_softmax_param()->set_axis(2);  
        softmax_m_param->set_name("softmax_m_" + ts);  
        softmax_m_param->add_bottom("m_f_" + ts);  
        softmax_m_param->add_top("mask_" + ts);  
        softmax_m_param->add_propagate_down(true); 
        
      }

      // 1*N*Et*1 ==> 1*N*Et
      {  
        LayerParameter* reshape_m_param = net_param->add_layer();  
        reshape_m_param->CopyFrom(reshape_layer_param);  
        BlobShape* shape = reshape_m_param->mutable_reshape_param()->mutable_shape();  
        shape->Clear();  
        shape->add_dim(0);  
        shape->add_dim(0);  
        shape->add_dim(0);  
        reshape_m_param->set_name("reshape_m_" + ts);  
        reshape_m_param->add_bottom("mask_" + ts);  
        reshape_m_param->add_top("mask_reshape_" + ts);
         reshape_m_param->add_propagate_down(true);   
      }  




      // permute ej from 1*N*C*Et ==> 1*N*Et*C
      {  
        LayerParameter* transpose_x_s_param = net_param->add_layer();
          transpose_x_s_param->CopyFrom(transpose_param);
          transpose_x_s_param->set_name("transpose_x_s_" + ts);
          transpose_x_s_param->mutable_transpose_param()->add_dim(0);
          transpose_x_s_param->mutable_transpose_param()->add_dim(1);
          transpose_x_s_param->mutable_transpose_param()->add_dim(3);
          transpose_x_s_param->mutable_transpose_param()->add_dim(2);
          transpose_x_s_param->add_bottom("x_" + ts);
          transpose_x_s_param->add_top("x_p_" + ts);
          transpose_x_s_param->add_propagate_down(true);   
      }  


      //top 1*N*Et*C
      //bottom[0]: x_p_  1*N*Et*C
      //bottom[1]: ej_reshape_ 1*N*Et
      {  
        LayerParameter* scale_x_param = net_param->add_layer();  
        scale_x_param->CopyFrom(scale_param);  
        scale_x_param->mutable_scale_param()->set_axis(0);
        scale_x_param->set_name("scale_x_" + ts);  
        scale_x_param->add_bottom("x_p_" + ts);  
        scale_x_param->add_bottom("mask_reshape_" + ts);  
        scale_x_param->add_top("x_mask_" + ts);  
        scale_x_param->add_propagate_down(true);
        scale_x_param->add_propagate_down(true);
      }  

    
      {  
          // 1*N*Et*C ==> 1*N*1*C
        LayerParameter* sum_att_param = net_param->add_layer();  
        sum_att_param->CopyFrom(axis_sum_param);  
        sum_att_param->mutable_sum_param()->set_axis(2);
        sum_att_param->set_name("sum_x_" + ts);  
        sum_att_param->add_bottom("x_mask_" + ts);  
        sum_att_param->add_top("sum_x_" + ts);  
        sum_att_param->add_propagate_down(true);



        // 1*N*1*C ==> 1*N*C
       
        LayerParameter* reshape_sum_param = net_param->add_layer();  
        reshape_sum_param->CopyFrom(reshape_layer_param);  
        BlobShape* shape = reshape_sum_param->mutable_reshape_param()->mutable_shape();  
        shape->Clear();  
        shape->add_dim(0);  
        shape->add_dim(0);  
        shape->add_dim(-1);  
        reshape_sum_param->set_name("att_fea_" + ts);  
        reshape_sum_param->add_bottom("sum_x_" + ts);  
        reshape_sum_param->add_top("att_x_" + ts);  
        reshape_sum_param->add_propagate_down(true);


          // //C*1*N ==> 1*N*C
          // LayerParameter* transpose_x_input_param = net_param->add_layer();
          // transpose_x_input_param->CopyFrom(transpose_param);
          // transpose_x_input_param->set_name("transpose_x_input_" + ts);
          // transpose_x_input_param->mutable_transpose_param()->add_dim(1);
          // transpose_x_input_param->mutable_transpose_param()->add_dim(2);
          // transpose_x_input_param->mutable_transpose_param()->add_dim(0);
          // transpose_x_input_param->add_bottom("att_x_" + ts);
          // transpose_x_input_param->add_top("input_x_" + ts);


      }  

      // Add layer to transform all timesteps of x to the hidden state dimension.
      //     W_xc_x = W_xc * x + b_c

    {
        LayerParameter* x_transform_param = net_param->add_layer();
        x_transform_param->CopyFrom(biased_hidden_param);
        x_transform_param->mutable_inner_product_param()->set_axis(2);
        x_transform_param->set_name("x_transform_" + ts);
        x_transform_param->add_param()->set_name("W_xc_input");
        x_transform_param->add_param()->set_name("b_c_input");
        x_transform_param->add_bottom("att_x_" + ts);
        x_transform_param->add_top("W_xc_x_"+ts);
        x_transform_param->add_propagate_down(true);

    }

    
      // Add layer to compute
      //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
      {
        LayerParameter* w_param = net_param->add_layer();
        w_param->CopyFrom(hidden_param);
        w_param->set_name("transform_" + ts);
        w_param->add_param()->set_name("W_hc");
        w_param->add_bottom("h_conted_" + tm1s);
        w_param->add_top("W_hc_h_" + tm1s);
        w_param->mutable_inner_product_param()->set_axis(2);
        w_param->add_propagate_down(true);
      }


     // Add the outputs of the linear transformations to compute the gate input.
     //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
     //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
     {
        LayerParameter* input_sum_layer = net_param->add_layer();
        input_sum_layer->CopyFrom(sum_param);
        input_sum_layer->set_name("gate_input_" + ts);
        input_sum_layer->add_bottom("W_hc_h_" + tm1s);
        input_sum_layer->add_bottom("W_xc_x_" + ts);
        input_sum_layer->add_top("gate_input_" + ts);
        input_sum_layer->add_propagate_down(true);
         input_sum_layer->add_propagate_down(true);
     }



    // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
    // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
    // Outputs: c_t, h_t
    //     [ i_t' ]
    //     [ f_t' ] := gate_input_t
    //     [ o_t' ]
    //     [ g_t' ]
    //         i_t := \sigmoid[i_t']
    //         f_t := \sigmoid[f_t']
    //         o_t := \sigmoid[o_t']
    //         g_t := \tanh[g_t']
    //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    //         h_t := o_t .* \tanh[c_t]
    {
      LayerParameter* lstm_unit_param = net_param->add_layer();
      lstm_unit_param->set_type("LSTMUnit");
      lstm_unit_param->add_bottom("c_" + tm1s);
      lstm_unit_param->add_bottom("gate_input_" + ts);
      lstm_unit_param->add_bottom("cont_" + ts);
      lstm_unit_param->add_top("c_" + ts);
      lstm_unit_param->add_top("h_" + ts);
      lstm_unit_param->set_name("unit_" + ts);
    }
    output_concat_layer.add_bottom("h_" + ts);
    output_m_layer.add_bottom("mask_" + ts);  


  }// end of for




  LayerParameter* c_T_copy_param = net_param->add_layer();  
  c_T_copy_param->CopyFrom(split_param);  
  c_T_copy_param->add_bottom("c_" + format_int(this->T_));  
  c_T_copy_param->add_top("c_T");  
  
  net_param->add_layer()->CopyFrom(output_concat_layer);  
  net_param->add_layer()->CopyFrom(output_m_layer);  
  net_param->set_force_backward(true);

}



INSTANTIATE_CLASS(AttLstmLayer);
REGISTER_LAYER_CLASS(AttLstm);

}  // namespace caffe
