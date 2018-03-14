import caffe
from caffe import layers as L
from caffe import params as P

def conv_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01), bias_filler=dict(type='constant', value=0))
    conv_relu = L.ReLU(conv, in_place=True)
    return conv, conv_relu
    
def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def conv_bn_scale(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0.2))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)

    return conv, conv_bn, conv_scale


def eltwize_relu(bottom1, bottom2):
    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True)

    return residual_eltwise, residual_eltwise_relu


def residual_branch(bottom, base_output=64):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = \
        conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1)  # base_output x n x n
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1)  # base_output x n x n
    branch2c, branch2c_bn, branch2c_scale = \
        conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1)  # 4*base_output x n x n

    residual, residual_relu = \
        eltwize_relu(bottom, branch2c)  # 4*base_output x n x n

    return branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, branch2b_bn, branch2b_scale, branch2b_relu, \
           branch2c, branch2c_bn, branch2c_scale, residual, residual_relu


def residual_branch_shortcut(bottom, stride=2, base_output=64):
    """

    :param stride: stride
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    branch1, branch1_bn, branch1_scale = \
        conv_bn_scale(bottom, num_output=4 * base_output, kernel_size=1, stride=stride)

    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = \
        conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1, stride=stride)
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1)
    branch2c, branch2c_bn, branch2c_scale = \
        conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1)

    residual, residual_relu = \
        eltwize_relu(branch1, branch2c)  # 4*base_output x n x n

    return branch1, branch1_bn, branch1_scale, branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, \
           branch2b_bn, branch2b_scale, branch2b_relu, branch2c, branch2c_bn, branch2c_scale, residual, residual_relu


def Upsample_with_conv(bottom, bottom2x, num_output):
    upsample = L.Deconvolution(bottom, convolution_param=dict(num_output=num_output*2, kernel_size=4, pad=1, \
                                stride=2, group=num_output*2, bias_term=False,  weight_filler=dict(type="bilinear")), \
                               param=dict(lr_mult=0, decay_mult=0))
    conv_conv = L.Convolution(bottom2x, num_output=num_output*2, kernel_size=1, pad=0, \
                              weight_filler=dict(type="xavier"), bias_filler=dict(type="constant", value=0), \
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    conv_conv_relu = L.ReLU(conv_conv, in_place=True)
    concat = L.Concat(upsample, conv_conv, axis=1)
    conv1x1 = L.Convolution(concat, num_output=num_output, kernel_size=1, pad=0, \
                            weight_filler=dict(type="xavier"), bias_filler=dict(type="constant", value=0), \
                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    conv1x1_relu = L.ReLU(conv1x1, in_place=True)
    conv3x3 = L.Convolution(conv1x1, num_output=num_output, kernel_size=3, pad=1, \
                            weight_filler=dict(type="xavier"), bias_filler=dict(type="constant", value=0), \
                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    conv3x3_relu = L.ReLU(conv3x3, in_place=True)
    return upsample, conv_conv, conv_conv_relu, concat, conv1x1, conv1x1_relu, conv3x3, conv3x3_relu

def loss_from_feature_and_label(feature_map, labels):
    """After produce the net prototxt, you should add silence layer
    and iou lambda param.
    """
    label1, label2, label3, gt_bbox, ignore_bbox = L.Slice(labels, slice_point=[1, 6, 14, 15], ntop=5, name='slice_labels')
    silence = L.Silence(label3, ntop=0)
    conv_final, conv_final_relu = conv_relu(feature_map, num_output=32, kernel_size=3, pad=1) 
    score_4s = L.Convolution(conv_final, num_output=2, kernel_size=1, pad=0, \
                             weight_filler=dict(type="xavier"), bias_filler=dict(type="constant", value=0), \
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    
    score_4s_softmax = L.Softmax(score_4s)
    new_label1 = L.OHEM(score_4s_softmax, label1, min_hard_random_ratio=0.2, max_hard_random_ratio=0.7, \
                        neg_num=1024, max_iteration=50000)
    loss_4s = L.SoftmaxWithLoss(score_4s, new_label1, propagate_down=[True, False], loss_weight=1, \
                                loss_param=dict(ignore_label=255, normalize=True))
    conv_feature_prior , conv_feature_prior_relu = conv_relu(conv_final, num_output=32, kernel_size=3, pad=1)
    conv_maps, conv_maps_relu = conv_relu(conv_feature_prior, num_output=4, kernel_size=1, pad=0)
    conv_orient = L.Convolution(conv_feature_prior, num_output=1, kernel_size=1, pad=0, \
                                weight_filler=dict(type="xavier", std=0.01), \
                                bias_filler=dict(type="constant", value=0), \
                                param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    concat_bbox_orient = L.Concat(conv_maps, conv_orient)
    trans_label1 = L.Transpose(label1, dim=[0, 2, 3, 1])
    trans_label2 = L.Transpose(label2, dim=[0, 2, 3, 1])
    trans_pre_bbox = L.Transpose(concat_bbox_orient, dim=[0, 2, 3, 1])
    reg_loss_forwad = L.IouLoss(trans_pre_bbox, trans_label2, trans_label1, \
                                only_forward=True, loss_weight=0, propagate_down=[False, False, False])
    trans_reg_loss_forward = L.Transpose(reg_loss_forwad, dim=[0, 3, 1, 2])
    ohem_reg = L.OHEM(trans_reg_loss_forward, label1, pos_num=256, reg=True, max_iteration=50000, \
                     min_pos_hard_random_ratio=0.2, max_pos_hard_random_ratio=0.7)
    trans_reg_label1 = L.Transpose(ohem_reg, dim=[0, 2, 3, 1])
    reg_loss = L.IouLoss(trans_pre_bbox, trans_label2, trans_reg_label1, only_forward=False, \
                        loss_weight=1, propagate_down=[True, False, False])
    return label1, label2, label3, gt_bbox, ignore_bbox, silence, conv_final, conv_final_relu, \
            score_4s, score_4s_softmax, new_label1, loss_4s, conv_feature_prior , \
            conv_feature_prior_relu, conv_maps, conv_maps_relu, conv_orient, concat_bbox_orient, \
            trans_label1, trans_label2, trans_pre_bbox, reg_loss_forwad, trans_reg_loss_forward, \
            ohem_reg, trans_reg_label1, reg_loss
            
branch_shortcut_string = 'n.res(stage)a_branch1, n.res(stage)a_branch1_bn, n.res(stage)a_branch1_scale, \
        n.res(stage)a_branch2a, n.res(stage)a_branch2a_bn, n.res(stage)a_branch2a_scale, n.res(stage)a_branch2a_relu, \
        n.res(stage)a_branch2b, n.res(stage)a_branch2b_bn, n.res(stage)a_branch2b_scale, n.res(stage)a_branch2b_relu, \
        n.res(stage)a_branch2c, n.res(stage)a_branch2c_bn, n.res(stage)a_branch2c_scale, n.res(stage)a, n.res(stage)a_relu = \
            residual_branch_shortcut((bottom), stride=(stride), base_output=(num))'

branch_string = 'n.res(stage)b(order)_branch2a, n.res(stage)b(order)_branch2a_bn, n.res(stage)b(order)_branch2a_scale, \
        n.res(stage)b(order)_branch2a_relu, n.res(stage)b(order)_branch2b, n.res(stage)b(order)_branch2b_bn, \
        n.res(stage)b(order)_branch2b_scale, n.res(stage)b(order)_branch2b_relu, n.res(stage)b(order)_branch2c, \
        n.res(stage)b(order)_branch2c_bn, n.res(stage)b(order)_branch2c_scale, n.res(stage)b(order), \
        n.res(stage)b(order)_relu = residual_branch((bottom), base_output=(num))'

upsample_string = 'n.upsample(stage), n.conv_conv(stage), n.conv_conv(stage)_relu, n.concat(stage), \
        n.conv(stage)_1x1, n.conv(stage)_1x1_relu, n.conv(stage)_3x3, n.conv(stage)_3x3_relu \
        = Upsample_with_conv((bottom), (bottom2x), (num_output))'

loss_string = 'n.label1, n.label2, n.label3, n.gt_bbox, n.ignore_bbox, n.silence, n.conv_final, n.conv_final_relu, \
            n.score_4s, n.score_4s_softmax, n.new_label1, n.loss_4s, n.conv_feature_prior , \
            n.conv_feature_prior_relu, n.conv_maps, n.conv_maps_relu, n.conv_orient, n.concat_bbox_orient, \
            n.trans_label1, n.trans_label2, n.trans_pre_bbox, n.reg_loss_forwad, n.trans_reg_loss_forward, \
            n.ohem_reg, n.trans_reg_label1, n.reg_loss = loss_from_feature_and_label((feature_map), (labels))'

class ResNet(object):
    def __init__(self, train_path, test_path):
        self.train_data = train_path
        self.test_data = test_path

    def resnet_layers_proto(self, batch_size, phase='TRAIN', stages=(3, 4, 23, 3)):
        """

        :param batch_size: the batch_size of train and test phase
        :param phase: TRAIN or TEST
        :param stages: the num of layers = 2 + 3*sum(stages), layers would better be chosen from [50, 101, 152]
                       {every stage is composed of 1 residual_branch_shortcut module and stage[i]-1 residual_branch
                       modules, each module consists of 3 conv layers}
                        (3, 4, 6, 3) for 50 layers; (3, 4, 23, 3) for 101 layers; (3, 8, 36, 3) for 152 layers
        """
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False

        n.data, n.label = L.UnitboxData(root_folder='', source=source_data, batch_size=batch_size, shuffle=True, \
                            is_color=True, min_scale=0.6, max_scale=2.0, min_size=640, max_size=2560, \
                            min_ratio=0.8, max_ratio=1.2, label_resize=4, crop_size=640, type="UnitBoxData", \
                            mean_value=[122, 122, 122], mirror=mirror, min_rotate=-10, max_rotate=10, ntop=2)
        n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = \
            conv_bn_scale_relu(n.data, num_output=64, kernel_size=7, stride=2, pad=3)  # 64x112x112
        n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x56x56

        for num in xrange(len(stages)):  # num = 0, 1, 2, 3
            for i in xrange(stages[num]):
                if i == 0:
                    stage_string = branch_shortcut_string
                    bottom_string = ['n.pool1', 'n.res2b%s' % str(stages[0] - 1), 'n.res3b%s' % str(stages[1] - 1),
                                     'n.res4b%s' % str(stages[2] - 1)][num]
                else:
                    stage_string = branch_string
                    if i == 1:
                        bottom_string = 'n.res%sa' % str(num + 2)
                    else:
                        bottom_string = 'n.res%sb%s' % (str(num + 2), str(i - 1))
                exec (stage_string.replace('(stage)', str(num + 2)).replace('(bottom)', bottom_string).
                      replace('(num)', str(2 ** num * 64)).replace('(order)', str(i)).
                      replace('(stride)', str(int(num > 0) + 1)))
                
        exec ('n.conv5_3x3, n.conv5_3x3_relu = conv_relu((bottom), num_output=(num), kernel_size=3, stride=1, pad=1)'.
              replace('(bottom)', 'n.res5b%s' % str(stages[-1] -1 )).replace('(num)', str(2**len(stages)*64)))
        
        for num in xrange(len(stages), 1, -1):  # num = 4, 3, 2 
            bottom_string = 'n.conv%s_3x3' % str(num + 1)
            bottom2x_string = ['n.pool1', 'n.res2b%s' % str(stages[0] - 1), 'n.res3b%s' % str(stages[1] - 1),
                                     'n.res4b%s' % str(stages[2] - 1)][num -1]
            exec (upsample_string.replace('(stage)', str(num)).replace('(bottom)', bottom_string).
                 replace('(bottom2x)', bottom2x_string).replace('(num_output)', str(2**(num-1)*64)))
        exec (loss_string.replace('(feature_map)', 'n.conv2_3x3').replace('(labels)', 'n.label'))
        return n.to_proto()