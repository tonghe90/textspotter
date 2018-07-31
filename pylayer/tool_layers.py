#import cfg
import sys
sys.path.insert(0, '../caffe/python')
import caffe
import yaml
from tool import set_blob_data, rotate_rect, non_max_suppression
import numpy as np
from math import atan


MAX_LEN = 25


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
        keep_indices, temp_boxes = non_max_suppression(orient_bboxes, self.nms_th)
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



MAX_LEN = cfg.MAX_LEN
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

        rects = np.zeros((gt_boxes.shape[0], 5), dtype=np.float32)
        for nn in range(gt_boxes.shape[0]):
            rects[nn, 0] = gt_boxes[nn, 0]
            rects[nn, 1] = np.min(gt_boxes[nn, [1, 3, 5, 7]])
            rects[nn, 2] = np.min(gt_boxes[nn, [2, 4, 6, 8]])
            rects[nn, 3] = np.max(gt_boxes[nn, [1, 3, 5, 7]])
            rects[nn, 4] = np.max(gt_boxes[nn, [2, 4, 6, 8]])


        if self.phase == 0:
           cont = np.ones((25, int(cont.shape[1])), dtype=np.float32)
           cont[0,:] = 0


        set_blob_data(top[0], gt_boxes)
        set_blob_data(top[1], cont)
        set_blob_data(top[2], input_label)
        set_blob_data(top[3], output_label)
        if len(top)>4:
            set_blob_data(top[4], rects)



