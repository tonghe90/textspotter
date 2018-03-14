import sys
sys.path.insert(0, "/data1/liuxuebo/caffe/python/")
sys.path.insert(0, "/data1/liuxuebo/caffe/python/src/")
import caffe
import os
import cv2
import math
import pyclipper
from shapely.geometry import Polygon
import numpy as np
gpu_id = 2
model_file = '/data1/liuxuebo/text_detect/models/2stage_one_lstm_iter_60000.caffemodel'
net_def_file = '/data1/liuxuebo/text_detect/test_pvanet2stage_one_lstm.pt'
image_dir = '/data1/liuxuebo/data/icdar/ch4_test_images'
# image_dir = '/data1/liuxuebo/data/icdar/Challenge2_Test_Task12_Images'
# image_dir = '/data1/liuxuebo/datauawei_500/train'
save_dir = '/data1/liuxuebo/data/icdar/icdar15_result_test_2stage_one_lstm'
image_list = []
image_resize_length_list = [2240]
# image_resize_length_list = [2560, 2240, 1920, 1600, 1280]
# image_resize_height = 1248
# image_resize_width = 2240
mean_value = 122
min_score = 0.
nms_score = 0.15
out_score=False
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if image_dir != '':
    image_list.extend(list(map(lambda x:os.path.join(image_dir, x), [im_name for im_name in os.listdir(image_dir) if (im_name.endswith('.jpg') and not im_name.startswith('.'))])))

class CaffeModel:
    def __init__(self, net_def_file, model_file):
        self.net_def_file=net_def_file
        self.net=caffe.Net(net_def_file, model_file, caffe.TEST)

    def blob(self, key):
        return self.net.blobs[key].data.copy()

    def forward(self, input_data):
        return self.forward2({"data": input_data[np.newaxis, :]})

    def forward2(self, input_data):
        for k, v in input_data.items():
            self.net.blobs[k].reshape(*v.shape)
            self.net.blobs[k].data[...]=v
        return self.net.forward()

    def net_def_file(self):
        return self.net_def_file

caffe.set_mode_gpu()
caffe.set_device(gpu_id)
caffe_model=CaffeModel(net_def_file, model_file)

def non_max_suppression_fast(boxes, overlapThresh):
    nms_min_score = 0
    east_min_iou = 5
    min_iou = 0.5
    if (len(boxes)==0):
        return [], boxes
    order = np.lexsort((tuple(boxes[:, -2]), tuple(boxes[:, -1])))
    new_boxes = boxes.copy()
    # initialize the list of picked indexes 
    pick = []
    suppressed = np.zeros((len(boxes)), dtype=np.int)
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    polygon = list()
    area = list()
    for box in boxes:
        p = list()
        for i in range(4):
            p.append([box[i*2], box[i*2+1]])
        polygon.append(pyclipper.scale_to_clipper(p))
        area.append((((box[0]-box[2])**2+(box[1]-box[3])**2)**0.5)*(((box[2]-box[4])**2+(box[3]-box[5])**2)**0.5))
    order = boxes[:,8].argsort()[::-1]
    for _i in range(len(boxes)):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        pick.append(i)
        area_i = area[i]
        p_i = polygon[i]
        neighbor_list = []
        for _j in range(_i + 1, len(boxes)):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            area_j = area[j]
            p_j = polygon[j]
            try:
                pc = pyclipper.Pyclipper()
                pc.AddPath(p_i, pyclipper.PT_CLIP, True)
                pc.AddPaths([p_j], pyclipper.PT_SUBJECT, True)
                solution = pc.Execute(pyclipper.CT_INTERSECTION)
                if len(solution) == 0:
                    inter = 0
                else:
                    inter = pyclipper.scale_from_clipper(pyclipper.scale_from_clipper(pyclipper.Area(solution[0])))
            except:
                inter = 0
            iou = inter / (area_i + area_j - inter)
            if iou > min_iou:
                neighbor_list.append(j)
            if (area_i + area_j - inter) > 0 and iou > overlapThresh:
                suppressed[j] = 1
        score_sum = boxes[i][8] - nms_min_score
        for k in range(8):
            new_boxes[i][k] *= score_sum
        for neighbor in neighbor_list:
            for k in range(8):
                new_boxes[i][k] += boxes[neighbor][k] * (boxes[neighbor][8] - nms_min_score)
            score_sum += (boxes[neighbor][8] - nms_min_score)
        for k in range(8):
            new_boxes[i][k] /= score_sum
    return pick, new_boxes


for image in image_list:
    print(image)
    new_boxes = list()
    for k in range(len(image_resize_length_list)):
        image_resize_length = image_resize_length_list[k]
        image_id = image.split('/')[-1].split('.')[0]
        im = cv2.imread(image)
        h,w,c = im.shape
        scale = max(h,w)/float(image_resize_length)
        image_resize_height = int(round(h / scale / 32) * 32)
        image_resize_width = int(round(w / scale / 32) * 32)
        scale_h = float(h)/image_resize_height
        scale_w = float(w)/image_resize_width
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        origin_im = im
        im = cv2.resize(im, (image_resize_width, image_resize_height))
        resized_im = im
        im = np.asarray(im, dtype=np.float32)
        # print(im[0])
        # print((im-mean_value)[0])
        # print(im[0,:,0])
        im = im - mean_value
        # print(im[0,:,0])
        im = np.transpose(im, (2, 0, 1))
        im = im[np.newaxis, :]
        # print(im[0,0,0,:])
        caffe_model.forward2({"data": im})
        pre_bbox=caffe_model.blob('proposal')
        pre_bbox_score=caffe_model.blob('bbox_pre_score')[:, 1, 0, 0]
        # pre_bbox[:,:,-3]=pre_bbox_score
        # stage1_score = pre_bbox[0, :, 8]
        # pre_bbox_score += stage1_score
        # pre_bbox_score /= 2
        keep_indices=np.where(pre_bbox_score>min_score)
        if len(keep_indices[0]) != pre_bbox.shape[1]:
            print(pre_bbox_score)
            print('before:', pre_bbox.shape[1])
            print('after:', len(keep_indices[0]))
        pre_bbox=pre_bbox[:, keep_indices[0], :]
        pre_bbox=pre_bbox[0]
        keep_indices=np.where(pre_bbox[:, 8]>0)
        pre_bbox = pre_bbox[keep_indices]
        for box in pre_bbox:
            box[:8]*=4
            box[0:8:2]*=scale_w
            box[1:8:2]*=scale_h
            new_boxes.append(box)
    new_boxes = np.array(new_boxes)
    keep_indices, temp_boxes = non_max_suppression_fast(new_boxes, nms_score)
    new_boxes = temp_boxes[keep_indices]
    with open(os.path.join(save_dir, 'res_'+image_id+'.txt'), 'w') as fw:
        for i, box in enumerate(new_boxes):
            if sum(box) <= 0:
                continue
            score = box[8]
            box = map(round, box)
            for j in range(4):
                if box[2*j]>=w:
                    box[2*j]=w-1
                if box[2*j+1]>=h:
                    box[2*j+1]=h-1
                if box[2*j]<0:
                    box[2*j]=0
                if box[2*j+1]<0:
                    box[2*j+1]=0
            box = map(int, box)
            if out_score:
                fw.write('{},{},{},{},{},{},{},{},{}\r\n'.format(box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7],score))
            else:
                fw.write('{},{},{},{},{},{},{},{}\r\n'.format(box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7]))
            cv2.line(origin_im, (box[0],box[1]), (box[2],box[3]), (255,0,0), 2)
            cv2.line(origin_im, (box[0],box[1]), (box[6],box[7]), (255,0,0), 2)
            cv2.line(origin_im, (box[4],box[5]), (box[2],box[3]), (255,0,0), 2)
            cv2.line(origin_im, (box[4],box[5]), (box[6],box[7]), (255,0,0), 2)
        cv2.imwrite(os.path.join(save_dir, 'res_'+image_id+'.jpg'), origin_im)





