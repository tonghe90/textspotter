#import cfg
import sys
sys.path.insert(0, '../caffe/python')
import caffe
import yaml
from tool import set_blob_data, set_blob_diff, gaussian1d, polyes_to_boxes2d_rotate, box2d_to_poly, rotate, rotate_rect, non_max_suppression_fast
import numpy as np
from math import atan, atan2
import cv2
import math

MAX_LEN = 25
class gen_gts_layer(caffe.Layer):
    """
    bottom[0]: gt_label [N,1,sz,sz]


    top[0]: rois N * 9
    top[1]: cont T * N
    top[2]: input_lobel T * N
    top[3]: output_label T * N
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.sel_num = layer_params['sel_num']
        ### 1 for training, 0 for testing
        self.phase = layer_params.get('phase', 1)


    def reshape(self, bottom, top):
        set_blob_data(top[0], np.zeros((1, 9), dtype=np.float32))
        set_blob_data(top[1], np.zeros((MAX_LEN, 1), dtype=np.float32))
        set_blob_data(top[2], np.zeros((MAX_LEN, 1), dtype=np.float32))
        set_blob_data(top[3], np.zeros((MAX_LEN, 1), dtype=np.float32))
        if len(top)>4:
            set_blob_data(top[4], np.zeros((1, 5), dtype=np.float32))
    def backward(self, top, propagate_down, bottom):
        pass

    def forward(self, bottom, top):
        batch_size = bottom[0].data.shape[0]
        # gt_boxes = []
        cont = []
        input_label = []
        output_label = []
        gt_boxes = np.zeros((0, 9))

        for n in range(batch_size):
            gt_label = bottom[0].data[n, 0]
            tmp = np.sum(gt_label, axis=1)
            gt_num = len(np.where(tmp != 0)[0])
            if gt_num == 0:
                continue

            roi_n = gt_label[:gt_num, :8] * 4
            roi_n = np.hstack((np.ones((gt_num, 1)) * n, roi_n))

            gt_boxes = np.vstack((gt_boxes, roi_n))

            for k in range(gt_num):
                label_len = int(gt_label[k, 9])
                if label_len > MAX_LEN-1:
                    cont.append([0]*MAX_LEN)
                    input_label.append([0]*MAX_LEN)
                    output_label.append([-1]*MAX_LEN)
                    continue
                pad = MAX_LEN - label_len - 1
                cont_tmp = [0] + [1] * label_len + [0] * pad
                cont.append(cont_tmp)
                input_tmp = [0] + list(gt_label[k, 10:10+label_len]) + [-1] * pad
                input_label.append(input_tmp)

                output_tmp = list(gt_label[k, 10:10+label_len]) + [0] + [-1] * pad
                output_label.append(output_tmp)


        if len(gt_boxes) == 0:
            set_blob_data(top[0], np.zeros((1, 9), dtype=np.float32))
            set_blob_data(top[1], np.zeros((MAX_LEN, 1), dtype=np.float32))
            set_blob_data(top[2], np.zeros((MAX_LEN, 1), dtype=np.float32))
            set_blob_data(top[3], -np.ones((MAX_LEN, 1), dtype=np.float32))
            if len(top) > 4:
                set_blob_data(top[4], np.zeros((1, 5), dtype=np.float32))


            return


        gt_boxes = np.array(gt_boxes).reshape(-1, 9)
        cont = np.array(cont, dtype=np.float32).reshape(-1, MAX_LEN).transpose(1,0)
        input_label = np.array(input_label, dtype=np.float32).reshape(-1, MAX_LEN).transpose(1,0)
        output_label = np.array(output_label,dtype=np.float32).reshape(-1, MAX_LEN).transpose(1,0)


        gt_len = gt_boxes.shape[0]

        sel_num = min(gt_len, self.sel_num)
        sel_ids = np.int32(np.random.choice(np.arange(gt_len), sel_num, replace = False))
        gt_boxes = gt_boxes[sel_ids]
        cont = cont[:, sel_ids]
        input_label = input_label[:, sel_ids]
        output_label = output_label[:, sel_ids]

        if self.phase == 0:
           cont = np.ones((25, int(cont.shape[1])), dtype=np.float32)
           cont[0,:] = 0


        set_blob_data(top[0], gt_boxes)
        set_blob_data(top[1], cont)
        set_blob_data(top[2], input_label)
        set_blob_data(top[3], output_label)

        if len(top)>4:
            rects = np.zeros((gt_boxes.shape[0], 5), dtype=np.float32)
            for nn in range(gt_boxes.shape[0]):
                rects[nn, 0] = gt_boxes[nn, 0]
                rects[nn, 1] = np.min(gt_boxes[nn, [1, 3, 5, 7]])
                rects[nn, 2] = np.min(gt_boxes[nn, [2, 4, 6, 8]])
                rects[nn, 3] = np.max(gt_boxes[nn, [1, 3, 5, 7]])
                rects[nn, 4] = np.max(gt_boxes[nn, [2, 4, 6, 8]])
            set_blob_data(top[4], rects)



class sample_points_layer(caffe.Layer):
    """
    bottom[0]: rois [N,9]

    top[0]: sample_points N * ver_num * hor_num * 2
    top[1]: sample_points_id N * ver_num * hor_num * 1
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.ver_num = layer_params['ver_num']
        self.hor_num = layer_params['hor_num']

    def reshape(self, bottom, top):
        set_blob_data(top[0], np.zeros((1, self.ver_num, self.hor_num, 2), dtype=np.float32))
        set_blob_data(top[1], np.zeros((1, self.ver_num, self.hor_num, 1), dtype=np.float32))

    def backward(self, top, propagate_down, bottom):
        pass

    def forward(self, bottom, top):
        roi_num = bottom[0].data.shape[0]
        rois = bottom[0].data
        sample_points = np.zeros((roi_num, self.ver_num, self.hor_num, 2))
        sample_id = np.zeros((roi_num, self.ver_num, self.hor_num, 1))
        for k in range(roi_num):
            bb = rois[k, 1:]
            edge1 = ((bb[0] - bb[2]) ** 2 + (bb[1] - bb[3]) ** 2)
            edge2 = ((bb[0] - bb[6]) ** 2 + (bb[1] - bb[7]) ** 2)
            #angle1 = atan(abs(bb[1] - bb[3]) / abs(bb[0] - bb[2]))
            #angle2 = atan(abs(bb[1] - bb[7]) / abs(bb[0] - bb[6]))
            point1 = np.zeros(2)
            point2 = np.zeros(2)
            box = bb
            if edge1 < edge2:
                for i in range(self.ver_num):

                    point1[0] = box[0] + (box[2] - box[0]) / float(self.ver_num + 1) * float(i + 1)
                    point1[1] = box[1] + (box[3] - box[1]) / float(self.ver_num + 1) * float(i + 1)
                    point2[0] = box[6] + (box[4] - box[6]) / float(self.ver_num + 1) * float(i + 1)
                    point2[1] = box[7] + (box[5] - box[7]) / float(self.ver_num + 1) * float(i + 1)
                    if point1[0] < point2[0]:
                        sample_points[k, i, :, 0] = np.linspace(point1[0], point2[0], self.hor_num)
                        sample_points[k, i, :, 1] = np.linspace(point1[1], point2[1], self.hor_num)
                    else:
                        sample_points[k, i, :, 0] = np.linspace(point2[0], point1[0], self.hor_num)
                        sample_points[k, i, :, 1] = np.linspace(point2[1], point1[1], self.hor_num)



            else:
                for i in range(self.ver_num):

                    point1[0] = box[0] + (box[6] - box[0]) / float(self.ver_num + 1) * float(i + 1)
                    point1[1] = box[1] + (box[7] - box[1]) / float(self.ver_num + 1) * float(i + 1)
                    point2[0] = box[2] + (box[4] - box[2]) / float(self.ver_num + 1) * float(i + 1)
                    point2[1] = box[3] + (box[5] - box[3]) / float(self.ver_num + 1) * float(i + 1)
                    if point1[0] < point2[0]:
                        sample_points[k, i, :, 0] = np.linspace(point1[0], point2[0], self.hor_num)
                        sample_points[k, i, :, 1] = np.linspace(point1[1], point2[1], self.hor_num)
                    else:
                        sample_points[k, i, :, 0] = np.linspace(point2[0], point1[0], self.hor_num)
                        sample_points[k, i, :, 1] = np.linspace(point2[1], point1[1], self.hor_num)

            sample_id[k] = rois[k, 0]
        set_blob_data(top[0], sample_points)
        set_blob_data(top[1], sample_id)



class sample_points_angle_layer(caffe.Layer):
    """
    bottom[0]: rois [N,9]

    top[0]: sample_points N * ver_num * hor_num * 2
    top[1]: sample_points_id N * ver_num * hor_num * 1
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.ver_num = layer_params['ver_num']
        self.hor_num = layer_params['hor_num']

    def reshape(self, bottom, top):
        set_blob_data(top[0], np.zeros((1, self.ver_num, self.hor_num, 2), dtype=np.float32))
        set_blob_data(top[1], np.zeros((1, self.ver_num, self.hor_num, 1), dtype=np.float32))
        if len(top) > 2:
            set_blob_data(top[2], np.zeros((1, self.hor_num, 2), dtype=np.float32))
            set_blob_data(top[3], np.zeros((1, self.hor_num, 1), dtype=np.float32))


    def backward(self, top, propagate_down, bottom):
        pass

    def forward(self, bottom, top):
        roi_num = bottom[0].data.shape[0]
        assert(roi_num>0)
        rois = bottom[0].data
        sample_points = np.zeros((roi_num, self.ver_num, self.hor_num, 2))
        sample_id = np.zeros((roi_num, self.ver_num, self.hor_num, 1))
        for k in range(roi_num):
            bb = rois[k, 1:]
            edge1 = ((bb[0] - bb[2]) ** 2 + (bb[1] - bb[3]) ** 2)
            edge2 = ((bb[0] - bb[6]) ** 2 + (bb[1] - bb[7]) ** 2)
            angle1 = abs(atan((bb[1] - bb[3]) / (bb[0] - bb[2] + 0.001)))
            angle2 = abs(atan((bb[1] - bb[7]) / (bb[0] - bb[6] + 0.001)))
            point1 = np.zeros(2)
            point2 = np.zeros(2)
            box = bb
            #print angle1,angle2
            if angle1 > angle2:

                for i in range(self.ver_num):
                    point1[0] = box[0] + (box[2] - box[0]) / float(self.ver_num + 1) * float(i + 1)
                    point1[1] = box[1] + (box[3] - box[1]) / float(self.ver_num + 1) * float(i + 1)
                    point2[0] = box[6] + (box[4] - box[6]) / float(self.ver_num + 1) * float(i + 1)
                    point2[1] = box[7] + (box[5] - box[7]) / float(self.ver_num + 1) * float(i + 1)
                    if point1[0] < point2[0]:
                        sample_points[k, i, :, 0] = np.linspace(point1[0], point2[0], self.hor_num)
                        sample_points[k, i, :, 1] = np.linspace(point1[1], point2[1], self.hor_num)
                    else:
                        sample_points[k, i, :, 0] = np.linspace(point2[0], point1[0], self.hor_num)
                        sample_points[k, i, :, 1] = np.linspace(point2[1], point1[1], self.hor_num)
            else:
                for i in range(self.ver_num):

                    point1[0] = box[0] + (box[6] - box[0]) / float(self.ver_num + 1) * float(i + 1)
                    point1[1] = box[1] + (box[7] - box[1]) / float(self.ver_num + 1) * float(i + 1)
                    point2[0] = box[2] + (box[4] - box[2]) / float(self.ver_num + 1) * float(i + 1)
                    point2[1] = box[3] + (box[5] - box[3]) / float(self.ver_num + 1) * float(i + 1)
                    if point1[0] < point2[0]:
                        sample_points[k, i, :, 0] = np.linspace(point1[0], point2[0], self.hor_num)
                        sample_points[k, i, :, 1] = np.linspace(point1[1], point2[1], self.hor_num)
                    else:
                        sample_points[k, i, :, 0] = np.linspace(point2[0], point1[0], self.hor_num)
                        sample_points[k, i, :, 1] = np.linspace(point2[1], point1[1], self.hor_num)


            sample_id[k] = rois[k, 0]
            set_blob_data(top[0], sample_points)
            set_blob_data(top[1], sample_id)
            if len(top) > 2:
                set_blob_data(top[2], np.mean(sample_points, axis=1))
                set_blob_data(top[3], np.mean(sample_id, axis=1))





class ps_embedding_layer(caffe.Layer):
    """
    bottom[0]: att_weights T*N*Et
    top[0]: ps_embedding T*N*C'(C'=Et)
    """

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        set_blob_data(top[0], np.zeros_like(bottom[0].data, dtype=np.float32))

    def backward(self, top, propagate_down, bottom):
        pass


    def forward(self, bottom, top):
        ps_embedding_blob = np.zeros_like(bottom[0].data, dtype=np.float32)
        for t in range(bottom[0].data.shape[0]):
            for n in range(bottom[0].data.shape[1]):
                id_x = np.argmax(bottom[0].data[t,n])
                ps_embedding_blob[t,n,id_x] = 1

        set_blob_data(top[0], ps_embedding_blob)





class text_accuracy_layer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        set_blob_data(top[0], np.zeros((bottom[0].data.shape[0], bottom[0].data.shape[1]), dtype=np.float32))

    def backward(self, top, propagate_down, bottom):
        pass


    def forward(self, bottom, top):
        soft_max_blob = bottom[0].data
        arg_max = np.argmax(soft_max_blob, axis=2)
        set_blob_data(top[0], arg_max)


class char_distribution_layer(caffe.Layer):

    """
    bottom[0]: cts[N*64*2]
    bottom[1]: char_boxes N * (max_len*8)
    bottom[2]: target_seq max_len*N

    top[0]: distribution N*max_len*64
    top[1]: weights  N*max_len*64
    """

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        pts_num = bottom[0].data.shape[1]
        set_blob_data(top[0], np.zeros((1, MAX_LEN, pts_num)))
        set_blob_data(top[1], np.zeros((1, MAX_LEN, pts_num)))

    def backward(self, top, propagate_down, bottom):
        pass


    def forward(self, bottom, top):
        num = bottom[1].data.shape[0]
        char_boxes = bottom[1].data
        seq = bottom[2].data
        pts = bottom[0].data

        pts_num = bottom[0].data.shape[1]
        distribution = np.zeros((num, MAX_LEN, pts_num), dtype=np.float32)
        weights = np.zeros((num, MAX_LEN, pts_num), dtype=np.float32)

        for n in range(num):
            char_len = np.argmax(np.where(seq[:,n] > 0)[0])

            char_box = char_boxes[n, :8*char_len].reshape(-1,8)
            for k in range(char_len):
                cx = np.mean(char_box[k, ::2])

                width = (abs(char_box[k, 0] - char_box[k,4]) + abs(char_box[k,2] - char_box[k,6])) * 0.5

                prob = gaussian1d((pts[n,:,0]-cx)/width*0.5, 0, 1)
                #print 'pts:', prob
                #print 'cx:', cx
                #print 'width:', width

                distribution[n, k] = prob
                weights[n,k] = 1


        set_blob_data(top[0], distribution)
        set_blob_data(top[1], weights)




class att_points_layer(caffe.Layer):

    """
    bottom[0]: cts[N*64*2]
    bottom[1]: char_boxes N * (max_len*8)
    bottom[2]: weights max_len * N * 64
    bottom[3]: target_seq max_len*N

    top[0]: char_gt max_len*N*2
    top[1]: char_gt_weights_in  max_len_*N*2
    top[2]: char_gt_weights_out max_len_*N*2
    top[3]: pred max_len_*N*2
    """

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        num = bottom[0].data.shape[0]
        layer_params = yaml.load(self.param_str)
        self.loss_weights = layer_params['weights']
        set_blob_data(top[0], np.zeros((MAX_LEN, num, 2)))
        set_blob_data(top[1], np.zeros((MAX_LEN, num, 2)))
        set_blob_data(top[2], np.zeros((MAX_LEN, num, 2)))
        set_blob_data(top[3], np.zeros((MAX_LEN, num, 2)))

    def backward(self, top, propagate_down, bottom):
        pass



    def forward(self, bottom, top):

        num = bottom[0].data.shape[0]
        seq = bottom[3].data
        pts = bottom[0].data

        att_weights = bottom[2].data
        assert (seq.shape[0] == MAX_LEN)
        assert (seq.shape[1] == num)
        char_gt = np.zeros((MAX_LEN, num, 2), dtype=np.float32)
        char_gt_inweights = np.zeros((MAX_LEN, num, 2), dtype=np.float32)
        char_gt_outweights = np.zeros((MAX_LEN, num, 2), dtype=np.float32)
        char_boxes = bottom[1].data
        pred_char_cx = np.zeros((MAX_LEN, num, 2), dtype=np.float32)

        if not (char_boxes > 0).any():
            set_blob_data(top[0], np.zeros((MAX_LEN, num, 2)))
            set_blob_data(top[1], np.zeros((MAX_LEN, num, 2)))
            set_blob_data(top[2], np.zeros((MAX_LEN, num, 2)))
            set_blob_data(top[3], np.zeros((MAX_LEN, num, 2)))
            if (len(top) > 4):
                set_blob_data(top[4], np.ones((MAX_LEN, num)))
            return



        mean_widths = np.ones((MAX_LEN, num), dtype=np.float32)

        for n in range(num):
            char_len = np.argmax(np.where(seq[:, n] > 0)[0]) + 1
            char_box = char_boxes[n, :8 * char_len].reshape(-1, 8)
            # xmaxs = np.max(char_box[:, ::2])
            # xmins = np.min(char_box[:, ::2])
            # mean_width = np.mean(xmaxs - xmins)
            # if mean_width == 0:
            #     mean_width = 1
            mean_width = np.mean(np.abs(char_box[:,0]-char_box[:,4]))
            if mean_width == 0:
                mean_width = 1
            mean_widths[:, n] = 1.0/mean_width
            for t in range(char_len):
                char_gt_inweights[t, n, 0] = 1 * self.loss_weights
                char_gt_outweights[t, n, 0] = 1 * self.loss_weights

                char_x_min = np.min(char_box[t][::2])
                char_x_max = np.max(char_box[t][::2])
                char_gt[t, n] = np.mean(char_box[t,::2]) / mean_width

                #print 'att:', np.sum(att_weights[t,n])

                ave_pt_x = np.sum(pts[n,:,0] * att_weights[t, n, :, 0])
                pred_char_cx[t, n, 0] = ave_pt_x / mean_width
                # print 'pred:', ave_pt_x
                # print 'gt:', np.mean(char_box[t,::2])
                # print 'width:', mean_width

                if ave_pt_x > char_x_min and ave_pt_x < char_x_max:
                    # char_gt_inweights[t, n, 0] = 0.3 * self.loss_weights
                    # char_gt_outweights[t, n, 0] = 0.3 * self.loss_weights
                    char_gt_inweights[t, n, 0] = 0.1 * self.loss_weights
                    char_gt_outweights[t, n, 0] = 1


        set_blob_data(top[0], char_gt)
        set_blob_data(top[1], char_gt_inweights)
        set_blob_data(top[2], char_gt_outweights)
        set_blob_data(top[3], pred_char_cx)
        if (len(top) > 4):
            set_blob_data(top[4], mean_widths)






class gen_gts_my_layer(caffe.Layer):
    """
    bottom[0]: gt_boxes N*9
    bottom[1]: char_boxes N * (max_len*8)
    bottom[2]: conts T*N
    bottom[3]: input_seq T*N
    bottom[4]: target_seq T*N

    top[0]: rois N * 9
    top[1]: char_boxes N * (max_len*8)
    top[2]: cont T * N
    top[3]: input_lobel T * N
    top[4]: output_label T * N
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.sel_num = layer_params['sel_num']


    def reshape(self, bottom, top):
        set_blob_data(top[0], np.zeros((1, 9), dtype=np.float32))
        set_blob_data(top[1], np.zeros((1, MAX_LEN*8), dtype=np.float32))
        set_blob_data(top[2], np.zeros((MAX_LEN, 1), dtype=np.float32))
        set_blob_data(top[3], np.zeros((MAX_LEN, 1), dtype=np.float32))
        set_blob_data(top[4], np.zeros((MAX_LEN, 1), dtype=np.float32))

    def backward(self, top, propagate_down, bottom):
        pass

    def forward(self, bottom, top):
        gt_boxes = bottom[0].data
        gt_num = gt_boxes.shape[0]

        if not (gt_boxes > 0).any():
            gt_num = 0

        if gt_num > 0:
            sel_num = min(gt_num, self.sel_num)
            sel_ids = np.int32(np.random.choice(np.arange(gt_num), sel_num, replace=False))
            gt_boxes = gt_boxes[sel_ids]
            char_boxes = bottom[1].data[sel_ids]
            cont = bottom[2].data[:, sel_ids]
            input_label = bottom[3].data[:, sel_ids]
            output_label = bottom[4].data[:, sel_ids]

            set_blob_data(top[0], gt_boxes)
            set_blob_data(top[1], char_boxes)
            set_blob_data(top[2], cont)
            set_blob_data(top[3], input_label)
            set_blob_data(top[4], output_label)
        else:
            set_blob_data(top[0], np.zeros((1,9), dtype=np.float32))
            set_blob_data(top[1], np.zeros((1, MAX_LEN * 8), dtype=np.float32))
            set_blob_data(top[2], np.zeros((MAX_LEN,1), dtype=np.float32))
            set_blob_data(top[3], np.zeros((MAX_LEN,1), dtype=np.float32))
            set_blob_data(top[4], -np.ones((MAX_LEN,1), dtype=np.float32))


class gen_char_cts_label(caffe.Layer):
    """
    bottom[0]: cts T*N*2
    bottom[1]: char_boxes N * (max_len*8)
    bottom[2]: sample_target T*N

    top[0]: char_label T*N

    """

    def setup(self, bottom, top):
        pass


    def reshape(self, bottom, top):
        set_blob_data(top[0], np.zeros((MAX_LEN, 1), dtype=np.float32))


    def backward(self, top, propagate_down, bottom):
        pass

    def forward(self, bottom, top):
        num = bottom[0].data.shape[1]
        assert num == bottom[1].data.shape[0]
        assert num == bottom[2].data.shape[1]
        char_label = -1*np.ones((MAX_LEN, num), dtype=np.float32)
        pred_pts_x = bottom[0].data

        char_boxes = bottom[1].data
        char_gts = bottom[2].data

        if not (char_boxes > 0).any():
            set_blob_data(top[0], char_label)
            return

        for n in range(num):
            char_len = np.argmax(np.where(char_gts[:, n] > 0)[0])+1
            char_box = char_boxes[n, :8 * char_len].reshape(-1, 8)
            char_box_cxs = np.mean(char_box[:,::2], axis=1)
            for t in range(char_len):
                pt_x = pred_pts_x[t, n, 0]
                tmp = int(np.argmin(np.abs(char_box_cxs-pt_x)))
                char_label[t, n] = char_gts[tmp, n]


        set_blob_data(top[0], char_label)


import matplotlib.pyplot as plt

class UnitRotateBoxCharLayer(caffe.Layer):
    """
    bottom[0]: data (n*c*h*w)
    bottom[1]: gt_boxes (N*9) batch_id(1) + four points(8)
    bottom[2]: charboxes (N*(max_len*8))
    bottom[3]: seq_gt T*N
    bottom[4]: ignore_mask n*1*h*w

    top[0]: maps n*4*h*w
    top[1]: map_angle  n*1*h*w
    top[2]: mask n*1*h*w
    top[3]: mask_char n*1*h*w
    param: scale, clip ratio
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.scale = layer_params.get('scale', 0.25)
        self.clip_ratio = layer_params.get('clip_ratio', 0.15)
        self.rf = layer_params.get('scale_factor', 1200)


    def reshape(self, bottom, top):

        n, c, h, w = bottom[0].data.shape
        h0 = int(h * self.scale)
        w0 = int(w * self.scale)
        set_blob_data(top[0], np.zeros((n, 4, h0, w0), dtype=np.float32))
        set_blob_data(top[1], np.zeros((n, 1, h0, w0), dtype=np.float32))
        set_blob_data(top[2], np.zeros((n, 1, h0, w0), dtype=np.float32))
        set_blob_data(top[3], np.zeros((n, 1, h0, w0), dtype=np.float32))


    def backward(self, top, propagate_down, bottom):
        pass

    def forward(self, bottom, top):

        polys = bottom[1].data[:, 1:]
        batch_ids = bottom[1].data[:,0]
        poly_num = polys.shape[0]
        im_h, im_w = bottom[0].data.shape[2:]
        out_h, out_w = int(im_h * self.scale), int(im_w * self.scale)

        boxes2d = polyes_to_boxes2d_rotate(polys)
        boxes2d_org = boxes2d.copy()
        boxes2d[:, :4] = boxes2d[:, :4] * self.scale

        ignore_masks = bottom[4].data
        batch_num = bottom[0].data.shape[0]
        assert (batch_num == ignore_masks.shape[0])
        assert (ignore_masks.shape[1] == 1)


        masks = np.zeros((batch_num, 1, out_h, out_w), dtype=np.float32)
        masks_char = -np.ones((batch_num, 1, out_h, out_w), dtype=np.float32)
        for n in range(batch_num):
            id_y, id_x = np.where(ignore_masks[n,0] < 0)
            id_y, id_x = np.int32(id_y*self.scale), np.int32(id_x*self.scale)
            masks[n,0,id_y,id_x] = -1
            masks_char[n,0,id_y,id_x] = -1




        maps = np.zeros((batch_num, 4, out_h, out_w), dtype=np.float32)
        angle_map = np.zeros((batch_num, 1, out_h, out_w), dtype=np.float32)


        seq_gts = bottom[3].data
        charBoxes = bottom[2].data * self.scale
        #masks_char = ignore_masks.copy()


        has_charboxes = 0
        if (charBoxes>0).any():
            has_charboxes = 1


        for n in range(poly_num):
            box2d = boxes2d[n]
            ind = int(batch_ids[n])
            cx, cy, wn, hn = box2d[:4]
            angle = box2d[-1]

            poly = box2d_to_poly(np.array([cx, cy, wn, hn, angle])).reshape(-1, 4, 2)
            poly_org = np.ascontiguousarray(poly, np.int)
            cv2.fillPoly(masks[ind, 0], [poly_org], (-1, -1, -1))

            ### mask_char
            char_len = len(np.where(seq_gts[:, n] > 0)[0])
            char_boxes = charBoxes[n, :8 * char_len].reshape(-1, 4, 2)
            if has_charboxes:

                for kk in range(char_len):
                    box_tmp = char_boxes[[kk]]
                    box_tmp = np.ascontiguousarray(box_tmp, np.int)
                    char_label = int(seq_gts[kk, n])
                    assert (char_label < 69)
                    cv2.fillPoly(masks_char[ind, 0], [box_tmp], (char_label, char_label, char_label))
            # else:
            #
            #     masks_char[...] = -1


            ### fill mask
            short_side = min(wn, hn)
            wn_new = int(round(max(2, wn - short_side * 2 * self.clip_ratio)))
            hn_new = int(round(max(2, hn - short_side * 2 * self.clip_ratio)))
            poly = box2d_to_poly(np.array([cx, cy, wn_new, hn_new, angle])).reshape(-1, 4, 2)
            poly = np.ascontiguousarray(poly, np.int)

            cv2.fillPoly(masks[ind, 0], [poly], (n+1, n+1, n+1))


            ### fill maps
            idy, idx = np.where(masks[ind, 0] == n + 1)
            masks[ind, 0, idy, idx] = 1
            if len(idx) == 0:
                continue
            cx, cy, wn, hn = boxes2d_org[n][:4]
            angle = boxes2d[n][-1]

            xmin = cx - wn / 2
            ymin = cy - hn / 2
            xmax = xmin + wn
            ymax = ymin + hn
            points = np.vstack((idx / self.scale, idy / self.scale))
            ro_points_x, ro_points_y = np.round(rotate(points, (cx, cy), -angle))

            # x_t
            maps[ind, 0, idy, idx] = (ro_points_y - ymin) / float(self.rf)
            # x_b
            maps[ind, 1, idy, idx] = (ymax - ro_points_y) / float(self.rf)
            # x_l
            maps[ind, 2, idy, idx] = (ro_points_x - xmin) / float(self.rf)
            # x_r
            maps[ind, 3, idy, idx] = (xmax - ro_points_x) / float(self.rf)
            # angle
            angle_map[ind, 0, idy, idx] = angle


        for n in range(batch_num):
            id_c, id_h, id_w = np.where(maps[n] <= 0)
            maps[n, id_c, id_h, id_w] = 0

        set_blob_data(top[0], maps)
        set_blob_data(top[1], angle_map)
        set_blob_data(top[2], masks)
        set_blob_data(top[3], masks_char)





class ohem_fcn_layer(caffe.Layer):
    """
    bottom[0]: fcn_softmax 1*2*H*W
    bottom[1]: mask_gt 1*1*H*W

    top[0]: gt_mask_new for fcn

    """
    def setup(self, bottom, top):

        layer_params = yaml.load(self.param_str)
        self.pos_num = layer_params['pos_num']
        self.neg_num = layer_params['neg_num']
        self.hardmining_ratio = float(layer_params.get('hard_ratio', 0.2))

    def reshape(self, bottom, top):
        set_blob_data(top[0], np.zeros_like(bottom[1].data, dtype=np.float32))


    def backward(self, top, propagate_down, bottom):
        pass

    def forward(self, bottom, top):

        fcn_new_mask = np.ones(bottom[1].data.shape, dtype=np.float32) * -1

        mask_gt = bottom[1].data[0,0]
        fcn_id_y_pos_org, fcn_id_x_pos_org = np.where(mask_gt == 1)
        fcn_id_y_neg_org, fcn_id_x_neg_org = np.where(mask_gt == 0)

        hard_pos_num = int(min(int(self.pos_num * self.hardmining_ratio), len(fcn_id_x_pos_org)))
        normal_pos_num = int(max(self.pos_num - hard_pos_num, 0))

        hard_neg_num = int(min(int(self.neg_num * self.hardmining_ratio), len(fcn_id_x_neg_org)))
        normal_neg_num = int(max(self.neg_num - hard_neg_num, 0))

        if hard_pos_num > 0 and normal_neg_num > 0 and hard_pos_num > 0 and normal_pos_num > 0:
            ###pos
            loss_fcn_pos = bottom[0].data[0, 0, fcn_id_y_pos_org, fcn_id_x_pos_org]
            fcn_sort_pos = np.argsort(loss_fcn_pos)[::-1]

            fcn_id_x_pos_org = fcn_id_x_pos_org[fcn_sort_pos]
            fcn_id_y_pos_org = fcn_id_y_pos_org[fcn_sort_pos]

            pos_hard_x = fcn_id_x_pos_org[:hard_pos_num]
            pos_hard_y = fcn_id_y_pos_org[:hard_pos_num]
            fcn_new_mask[0,0,pos_hard_y,pos_hard_x] = 1

            #print hard_pos_num, len(fcn_id_x_pos_org), normal_pos_num
            if len(fcn_id_x_pos_org) > hard_pos_num:
                id_tmp = np.random.choice(np.arange(hard_pos_num, len(fcn_id_x_pos_org)), normal_pos_num, replace=True)
                fcn_new_mask[0,0,fcn_id_y_pos_org[id_tmp], fcn_id_x_pos_org[id_tmp]] = 1


            ###neg
            loss_fcn_neg = bottom[0].data[0,1,fcn_id_y_neg_org, fcn_id_x_neg_org]
            fcn_sort_neg = np.argsort(loss_fcn_neg)[::-1]
            fcn_id_x_neg_org = fcn_id_x_neg_org[fcn_sort_neg]
            fcn_id_y_neg_org = fcn_id_y_neg_org[fcn_sort_neg]

            neg_hard_x = fcn_id_x_neg_org[:hard_neg_num]
            neg_hard_y = fcn_id_y_neg_org[:hard_neg_num]
            fcn_new_mask[0, 0, neg_hard_y, neg_hard_x] = 0
            if len(fcn_id_x_neg_org) > hard_neg_num:
                id_tmp = np.random.choice(np.arange(hard_neg_num, len(fcn_id_x_neg_org)), normal_neg_num, replace=True)
                fcn_new_mask[0, 0, fcn_id_y_neg_org[id_tmp], fcn_id_x_neg_org[id_tmp]] = 0

        set_blob_data(top[0], fcn_new_mask)






class det_nms_layer(caffe.Layer):
    """
    bottom[0]: fcn_softmax 1*2*H*W
    bottom[1]: iou_maps 1*4*H*W
    bottom[2]: angle_map 1*1*H*W

    top[0]: rois
    top[1]: sample_gt_cont
    params: nms_th, fcn_th, rf
    """
    def setup(self, bottom, top):

        layer_params = yaml.load(self.param_str)
        self.nms_th = layer_params['nms_th']
        self.fcn_th = layer_params['fcn_th']
        self.rf = layer_params['rf']

    def reshape(self, bottom, top):
        set_blob_data(top[0], np.zeros((1,10), dtype=np.float32))
        if len(top) > 1:
            set_blob_data(top[1], np.zeros((25,1), dtype=np.float32))

    def backward(self, top, propagate_down, bottom):
        pass

    def forward(self, bottom, top):
        if len(bottom) > 3:
            self.fcn_th = float(bottom[3].data[0,0])
        pre_score = bottom[0].data[0,1,:,:]
        pre_bbox = bottom[1].data[0,:,:,:]
        pre_orient = bottom[2].data[0,0,:,:]
        pre_bbox *= self.rf
        pos_y, pos_x = np.where(pre_score > self.fcn_th)
        bboxes = np.zeros((len(pos_y), 7), dtype='float32')
        orients = np.zeros((len(pos_y)), dtype='float32')
        for i in range(len(pos_y)):
            y = pos_y[i]
            x = pos_x[i]
            t, b, l, r = pre_bbox[:, y, x]
            bboxes[i] = np.asarray((x - l, y - t, x + r, y + b, pre_score[y, x], x, y))
            orients[i] = pre_orient[y, x]
        orient_bboxes = np.zeros((len(bboxes), 11), dtype='float32')
        for i, box in enumerate(bboxes):
            box[0] *= 4
            box[2] *= 4
            box[1] *= 4
            box[3] *= 4
            box[5] *= 4
            box[6] *= 4
            temp_box = rotate_rect(box[0], box[1], box[2], box[3], orients[i], box[5], box[6])
            orient_bboxes[i] = np.array((temp_box[0][0], temp_box[0][1], temp_box[1][0], temp_box[1][1], temp_box[2][0],
                                         temp_box[2][1], temp_box[3][0], temp_box[3][1], box[4], box[-2], box[-1]))
        keep_indices, temp_boxes = non_max_suppression_fast(orient_bboxes, self.nms_th)
        #assert len(keep_indices)==len(temp_boxes)
        #orient_bboxes = temp_boxes[keep_indices]
        orient_bboxes = orient_bboxes[keep_indices]
        if len(orient_bboxes)>0:
            set_blob_data(top[0], np.array(np.hstack((np.zeros((orient_bboxes.shape[0], 1)),orient_bboxes[:,:9])), dtype=np.float32))
            if len(top) > 1:
                cont = np.ones((25, orient_bboxes.shape[0]), dtype=np.float32)
                cont[0,:] = 0
                set_blob_data(top[1], cont)
        else:
            set_blob_data(top[0], np.zeros((1,10), dtype=np.float32))
            if len(top) > 1:
                set_blob_data(top[1], np.zeros((25, 1), dtype=np.float32))




class crop_att_layer(caffe.Layer):
    """
    bottom[0]: att_feature T*N*256
    bottom[1]: att_points T*N*2
    bottom[2]: att_points_id T*N*...
    bottom[3]: sample_gt_output T*N
    bottom[4]: feature maps N'*C*H*W
    bottom[5]: char_mask N'*1*H*W
    bottom[6]: char_boxes N*(max_len*8)

    top[0]: crop_feature_map crop_num*(C+256)*crop_h*crop_w
    top[1]: crop_mask crop_num*1*crop_h*crop_w

    params: ratio, crop_num
    """
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.ratio = layer_params['ratio']
        self.crop_num = layer_params['crop_num']

    def reshape(self, bottom, top):
        N = bottom[0].data.shape[1]
        assert N == bottom[1].data.shape[1]
        assert N == bottom[2].data.shape[1]
        assert N == bottom[3].data.shape[1]
        assert N == bottom[6].data.shape[1]
        max_batch_id = np.max(bottom[2].data)
        assert max_batch_id < bottom[4].data.shape[0]
        assert bottom[4].data.shape[0] == bottom[5].data.shape[0]
        assert bottom[5].data.shape[1] == 1
        assert bottom[5].data.shape[2] == bottom[4].data.shape[2]
        assert bottom[5].data.shape[3] == bottom[4].data.shape[3]

        channels = bottom[0].data.shape[2] + bottom[4].data.shape[1]
        set_blob_data(top[0], np.zeros((self.crop_num, channels, 50, 50), dtype=np.float32))
        set_blob_data(top[1], np.zeros((self.crop_num, 1, 50, 50), dtype=np.float32))



    def forward(self, bottom, top):
        N = bottom[0].data.shape[1]
        sample_gt_output = bottom[3].data
        cts = bottom[1].data
        cts_id = bottom[2].data
        char_boxes = bottom[6].data
        feature_maps = bottom[4].data
        char_mask_gt = bottom[5].data
        att_feature = bottom[0].data

        cnt = 0
        mean_h = 0
        for n in range(N):
            seq_len = len(np.where(sample_gt_output[:, n] > 0)[0])
            char_box = char_boxes[n, :8*seq_len].reshape(-1,8)
            for t in range(seq_len):
                box = char_box[t]
                h = max(box.reshape(4,2)[:,1]) - min(box.reshape(4,2)[:,1])
                assert h > 0
                cnt += 1
                mean_h += h

        mean_h = math.ceil(mean_h / float(cnt) * self.ratio)
        assert mean_h > 0

        crop_size = mean_h
        crop_rois = []
        fea_h, fea_w = bottom[4].data.shape[2:]
        ids = []
        for n in range(N):
            seq_len = len(np.where(sample_gt_output[:, n] > 0)[0])
            for t in range(seq_len):
                center_x, center_y = np.int32(cts[t, n] * self.ratio)
                assert len(np.unique(cts_id[t, n])) == 1
                pt_id = np.unique(cts_id[t, n])[0]
                xmin = center_x - int(crop_size/2)
                xmax = xmin + crop_size
                ymin = center_y - int(crop_size/2)
                ymax = ymin + crop_size

                if xmin >= 0 and ymin >=0 and xmax < fea_w and ymax < fea_h:
                    crop_rois.append([pt_id, xmin, ymin, xmax, ymax])
                    ids.append([t, n])


        crop_rois = np.array(crop_rois)
        crop_num = min(self.crop_num, len(crop_rois))
        self.crop_num = crop_num
        self.crop_rois = crop_rois
        self.ids = ids
        if crop_num > 0:
            c1 = bottom[4].data.shape[1]
            c2 = bottom[0].data.shape[2]
            crop_features = np.zeros((crop_num, c1+c2, crop_size, crop_size), dtype=np.float32)
            crop_masks = -np.ones((crop_num, 1, crop_size, crop_size), dtype=np.float32)
            for n in range(crop_num):
                batch_id, xmin, ymin, xmax, ymax = crop_rois[n]
                crop_features[n, :c1, :, :] = feature_maps[int(batch_id), :, ymin:ymax, xmin:xmax]
                crop_features[n, c1:, :, :] = att_feature[ids[n][0], ids[n][1]].reshape(-1,1,1)
                crop_masks[n, 0] = char_mask_gt[int(batch_id), 0, ymin:ymax, xmin:xmax]
        else:
            c1 = bottom[4].data.shape[1]
            c2 = bottom[0].data.shape[2]
            crop_features = np.zeros((crop_num, c1 + c2, crop_size, crop_size), dtype=np.float32)
            crop_masks = -np.ones((crop_num, 1, crop_size, crop_size), dtype=np.float32)

        set_blob_data(top[0], crop_features)
        set_blob_data(top[1], crop_masks)


    def backward(self, top, propagate_down, bottom):
        set_blob_diff(bottom[0], np.zeros_like(bottom[0].data, dtype=np.float32))
        set_blob_diff(bottom[4], np.zeros_like(bottom[4].data, dtype=np.float32))
        c1 = bottom[4].data.shape[1]
        c2 = bottom[0].data.shape[2]
        if propagate_down[0]:
            if self.crop_num > 0:
                for n in range(self.crop_num):
                    batch_id, xmin, ymin, xmax, ymax = self.crop_rois[n]
                    bottom[4].diff[int(batch_id), :, ymin:ymax, xmin:xmax] = top[0].diff[n, :c1]
                    tmp = np.sum(top[0].diff[n, c1:, :, :], axis=1)
                    tmp = np.sum(tmp, axis=1)
                    bottom[0].diff[self.ids[n][0], self.ids[n][1]] =tmp


















