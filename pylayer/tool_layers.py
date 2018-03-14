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

