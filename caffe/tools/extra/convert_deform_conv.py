import sys
sys.path.insert(0, "/data1/liuxuebo/caffe/python/")
sys.path.insert(0, "/data1/liuxuebo/caffe/python/src/")
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb

proto_in='/data1/liuxuebo/caffe/examples/text_detect/train_pvanet.pt'
proto_out='/data1/liuxuebo/caffe/examples/text_detect/train_pvanet_deform_lr01.pt'
replace_after_layer='concat_8x'
replace=False
net = caffe_pb2.NetParameter()
out_net = caffe_pb2.NetParameter()
with open(proto_in, 'r') as f:
    pb.text_format.Merge(f.read(), net)
for layer in net.layer:
    if layer.name==replace_after_layer:
        replace=True
    if layer.type=='Convolution' and replace:
        conv_offset=out_net.layer.add()
        conv_offset.name=layer.name+'_offset'
        conv_offset.type=layer.type
        # if len(layer.convolution_param.kernel_size)==0:
        #     kernel_h=layer.convolution_param.kernel_h
        #     kernel_w=layer.convolution_param.kernel_w
        # else: 
        #     kernel_h=layer.convolution_param.kernel_size[0]
        #     kernel_w=layer.convolution_param.kernel_size[0]
        # if len(layer.convolution_param.pad)==0:
        #     pad_h=layer.convolution_param.pad_h
        #     pad_w=layer.convolution_param.pad_w
        # else:
        #     pad_h=layer.convolution_param.pad[0]
        #     pad_w=layer.convolution_param.pad[0]
        conv_offset.bottom.append(layer.bottom[0])
        conv_offset.top.append(conv_offset.name)
        conv_offset.convolution_param.kernel_h=1
        conv_offset.convolution_param.kernel_w=1
        conv_offset.param.add()
        conv_offset.param.add()
        conv_offset.param[0].lr_mult=0.1
        conv_offset.param[1].lr_mult=0.1
        # conv_offset.convolution_param.pad_w=pad_w
        # conv_offset.convolution_param.pad_h=pad_h
        # conv_offset.convolution_param.stride.append(1)
        # conv_offset.convolution_param.bias_term=layer.convolution_param.bias_term
        # if layer.convolution_param.HasField('weight_filler'):
        #     conv_offset.convolution_param.weight_filler.MergeFrom(layer.convolution_param.weight_filler)
        # if layer.convolution_param.HasField('bias_filler'):
        #     conv_offset.convolution_param.bias_filler.MergeFrom(layer.convolution_param.bias_filler)
        # conv_offset.convolution_param.weight_filler.MergeFrom(layer.convolution_param.weight_filler)
        # conv_offset.convolution_param.weight_filler.MergeFrom(layer.convolution_param.weight_filler)
        conv_offset.convolution_param.num_output=2
        deform_conv=out_net.layer.add()
        deform_conv.name=layer.name+'_deform'
        deform_conv.type='DeformConv'
        deform_conv.bottom.append(layer.bottom[0])
        deform_conv.bottom.append(conv_offset.top[0])
        deform_conv.top.append(deform_conv.name)
        deform_conv.deformconv_param.kernel_h=1
        deform_conv.deformconv_param.kernel_w=1
        layer.bottom[0]=deform_conv.top[0]
    new_layer=out_net.layer.add()
    new_layer.CopyFrom(layer)
with open(proto_out, 'w') as f:
    f.write(pb.text_format.MessageToString(out_net))