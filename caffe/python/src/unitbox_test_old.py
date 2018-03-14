# -*- coding: utf-8 -*-
import sys
# caffe_root = '/mnt/lustre/chendagui/caffe_text_detect/'
# sys.path.insert(0, caffe_root+"python/")
sys.path.insert(0, "/mnt/lustre/liuxuebo/caffe_text_detect/python/")
import caffe
import os
import cv2
import math
import pyclipper
import numpy as np
from shapely.geometry import Polygon
from skimage.transform import rotate
from time import clock
from rotated_nms import non_max_suppression_fast

# from utils.nms.cpu_nms import cpu_nms as nms
# %matplotlib inline
# caffe.mpi_init()
start=clock()

gpu_id = 0
four_points = True
output_score = True
recog = False

model_file = '/mnt/lustre/liuxuebo/text_detect/models/mlt_no_ohem_reg_iter_150000.caffemodel'
# model_file = '/mnt/lustre/liuxuebo/text_detect/models/icdar15_deform_lr01_iter_240000.caffemodel'
# model_file = '/mnt/lustre/liuxuebo/text_detect/models/icdar17_scale_pow2_iter_240000_iter_220000.caffemodel'
net_def_file = '/mnt/lustre/liuxuebo/text_detect/test_pvanet.pt'
# image_dir = '/mnt/lustre/liuxuebo/data/express/align/Image/201611_汇通_1w_aligned/images'
image_dir = '/mnt/lustre/pocgroup/dataset/research/ICDAR2017MLT/ch8_validation_images/'
# image_dir = '/mnt/lustre/liuxuebo/data/icdar2017rctw_train_v1.2/val'
# image_dir = '/mnt/lustre/liuxuebo/data/icdar2017rctw_test/icdar2017rctw_test'
# image_dir = '/mnt/lustre/liuxuebo/datauawei_500/train'
save_dir = '/mnt/lustre/pocgroup/dataset/research/ICDAR2017MLT/ch8_validation_images_pred'
save_prefix = 'res_'

# text_detect_root = caffe_root + 'examples/text_detect/'
# model_file = text_detect_root + 'models/icdar15_iter_240000.caffemodel'
# net_def_file = text_detect_root + 'test_pvanet.pt'
# image_dir = caffe_root + '../data/icdar/ch4_test_images'
# save_dir = caffe_root + '../data/icdar/test_result'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_list = save_dir + '/list.txt'
image_list = []
if recog:
    f = open(save_list, 'w+')

# image_resize_length_list = [2560, 1280, 640]
# image_resize_length_list = [2560, 128, 640]
# image_resize_length_list = [2560, 2240, 1920, 1600, 1280]
image_resize_length_list = [2240]
mask_threshold_list = [0.8]
# mask_threshold_list = [0.99, 0.98, 0.995, 0.995, 0.999]
# mask_threshold_list = [0.1, 0.1, 0.1]

mean_value = 122
# image_resize_height_list = [1472,1248,1088,896,736]
# image_resize_width_list = [2560,2240,1920,1600,1280]
# mask_threshold_list = [0.98, 0.97, 0.98, 0.99, 0.99]
base_nms_score = 0.8
nms_score = 0.15

def rotate_rect(x1,y1,x2,y2,degree,center_x,center_y):
    # print('_______')
    # print(center_x, center_y)
    # center_x = float(x1+x2)/2
    # center_y = float(y1+y2)/2
    # print(center_x, center_y)
    points = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    new_points = list()
    for point in points:
        dx = point[0] - center_x
        dy = point[1] - center_y
        new_x = center_x + dx * math.cos(degree) - dy * math.sin(degree)
        new_y = center_y + dx * math.sin(degree) + dy * math.cos(degree)
        new_points.append([(new_x), (new_y)])
    return new_points

if image_dir != '':
    image_list.extend(list(map(lambda x:os.path.join(image_dir, x), [im_name for im_name in os.listdir(image_dir) if ((im_name.endswith('.jpg') or im_name.endswith('.jpeg')) and not im_name.startswith('.'))])))
if len(sys.argv)==3:
    num_thread=int(sys.argv[1])
    thread_id=int(sys.argv[2]) # 0 is the begining
    num_image=int(float(len(image_list))/num_thread)+1
    with open('temp.txt','a') as fw:
        fw.write('{} {} {} {}\n'.format(thread_id,num_image,max(0,thread_id*num_image),min(len(image_list),(thread_id+1)*num_image)))
    image_list = image_list[int(max(0,thread_id*num_image)):int(min(len(image_list),(thread_id+1)*num_image))]
# image_list=image_list[3100:]
# import the necessary packages
import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



# def non_max_suppression_fast(boxes, overlapThresh):
#     min_score = 0
#     east_min_iou = 5
#     min_iou = 0.5
#     if (len(boxes)==0):
#         return [], boxes
#     order = np.lexsort((tuple(boxes[:, -2]), tuple(boxes[:, -1])))
#     # # print(boxes.shape)
#     # S = list()
#     # p = list()
#     # p = boxes[order[0]]
#     # num_p = 1
#     # for _i in range(1,len(boxes)):
#     #     i = order[_i]
#     #     # print(boxes[i][-2:])
#     #     p_p = Polygon([(p[0],p[1]),(p[2],p[3]),(p[4],p[5]),(p[6],p[7])])
#     #     area_p = p_p.area
#     #     p_g = Polygon([(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(boxes[i][4],boxes[i][5]),(boxes[i][6],boxes[i][7])])
#     #     area_g = p_g.area
#     #     inter = p_p.intersection(p_g).area
#     #     iou = inter / (area_p + area_g - inter)
#     #     if iou > east_min_iou:
#     #         score1 = p[8]
#     #         score2 = boxes[i][8]
#     #         p[8] = max(score2, score1)
#     #         num_p+=1
#     #         for k in range(8):
#     #             p[k] = p[k] * score1 + boxes[i][k] * score2
#     #             p[k] /= (score1+score2)
#     #     else:
#     #         # p[8] /= num_p
#     #         S.append(p)
#     #         p = boxes[i]
#     #         num_p = 1
#     # if len(p)>0:
#     #     p[8] /= num_p
#     #     S.append(p)
#     # boxes = np.array(S)
#     new_boxes = boxes.copy()
#     # initialize the list of picked indexes
#     pick = []
#     suppressed = np.zeros((len(boxes)), dtype=np.int)
#     # compute the area of the bounding boxes and sort the bounding
#     # boxes by the bottom-right y-coordinate of the bounding box
#     polygon = list()
#     area = list()
#     for box in boxes:
#         p = Polygon([(box[0],box[1]),(box[2],box[3]),(box[4],box[5]),(box[6],box[7])])
#         polygon.append(p)
#         area.append(p.area)
#     order = boxes[:,8].argsort()[::-1]
#     for _i in range(len(boxes)):
#         i = order[_i]
#         if suppressed[i] == 1:
#             continue
#         pick.append(i)
#         area_i = area[i]
#         p_i = polygon[i]
#         neighbor_list = []
#         for _j in range(_i + 1, len(boxes)):
#             j = order[_j]
#             if suppressed[j] == 1:
#                 continue
#             area_j = area[j]
#             p_j = polygon[j]
#             inter = p_i.intersection(p_j).area
#             iou = inter / (area_i + area_j - inter)
#             if iou > min_iou:
#                 neighbor_list.append(j)
#             if (area_i + area_j - inter) > 0 and iou > overlapThresh:
#                 suppressed[j] = 1
#         score_sum = boxes[i][8] - min_score
#         for k in range(8):
#             new_boxes[i][k] *= score_sum
#         for neighbor in neighbor_list:
#             for k in range(8):
#                 new_boxes[i][k] += boxes[neighbor][k] * (boxes[neighbor][8] - min_score)
#             score_sum += (boxes[neighbor][8] - min_score)
#         for k in range(8):
#             new_boxes[i][k] /= score_sum
#     return pick, new_boxes

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

for image in image_list:
    print(image)
    # if not image.endswith('5675.jpg'):
    #     continuepre_score
    new_boxes = list()
    for k in range(len(image_resize_length_list)):
        # image_resize_height = image_resize_height_list[k]
        # image_resize_width = image_resize_width_list[k]
        image_resize_length = image_resize_length_list[k]
        mask_threshold = mask_threshold_list[k]
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
        start_caffe=clock()
        net_output=caffe_model.forward2({"data": im})
        end_caffe=clock()
        # print('model running time:', end_caffe-start_caffe)
        start_post=clock()
        # pre_score=caffe_model.blob('score_4s')[:,1,:,:]
        # pre_orient=caffe_model.blob('conv_orient')
        # pre_bbox=caffe_model.blob('conv_maps')
        pre_score=caffe_model.blob('pre_score')[:,1,:,:]
        pre_orient=caffe_model.blob('pre_orient')
        pre_bbox=caffe_model.blob('pre_bbox')
        pre_score=pre_score.squeeze()
        pre_orient=pre_orient.squeeze()
        pre_bbox=pre_bbox.squeeze()
        pre_bbox*=10
        # print(pre_orient[0]-math.pi/2)
        pos_y, pos_x=np.where(pre_score>mask_threshold)
        bboxes = np.zeros((len(pos_y), 7),dtype='float32')
        orients = np.zeros((len(pos_y)),dtype='float32')
        for i in range(len(pos_y)):
            y = pos_y[i]
            x = pos_x[i]
            t, b, l, r = pre_bbox[:,y,x]
            bboxes[i] = np.asarray((x-l,y-t,x+r,y+b,pre_score[y,x],x,y))
            orients[i] = pre_orient[y,x]
        # keep_indices = py_cpu_nms(bboxes[:,:5], base_nms_score)
        # bboxes = bboxes[keep_indices]
        orient_bboxes = np.zeros((len(bboxes), 11),dtype='float32')
        for i, box in enumerate(bboxes):
            box[0]*=(scale_w*4)
            box[2]*=(scale_w*4)
            box[1]*=(scale_h*4)
            box[3]*=(scale_h*4)
            box[5]*=(scale_w*4)
            box[6]*=(scale_h*4)
            # dist = ((box[0]-box[2])**2+(box[1]-box[3])**2)**0.5
            # box[4]*=min(200, dist) / max(200, dist)
            if four_points:
                temp_box = rotate_rect(box[0],box[1],box[2],box[3],orients[i],box[5],box[6])
            else:
                temp_box = rotate_rect(box[0],box[1],box[2],box[3],0,pos[i,0],pos[i,1])
            orient_bboxes[i] = np.array((temp_box[0][0],temp_box[0][1],temp_box[1][0],temp_box[1][1],temp_box[2][0],temp_box[2][1],temp_box[3][0],temp_box[3][1],box[4],box[-2],box[-1]))
        keep_indices, temp_boxes = non_max_suppression_fast(orient_bboxes, nms_score)
        orient_bboxes = temp_boxes
        end_post=clock()
        # print('post time:', end_post-start_post)
        for index in keep_indices:
            for i in range(4):
                orient_bboxes[index][2*i] = int(round(orient_bboxes[index][2*i]))
                orient_bboxes[index][2*i] = max(0, orient_bboxes[index][2*i])
                orient_bboxes[index][2*i] = min(w-1, orient_bboxes[index][2*i])
                orient_bboxes[index][2*i+1] = int(round(orient_bboxes[index][2*i+1]))
                orient_bboxes[index][2*i+1] = max(0, orient_bboxes[index][2*i+1])
                orient_bboxes[index][2*i+1] = min(h-1, orient_bboxes[index][2*i+1])
            new_boxes.append(orient_bboxes[index])
        # bboxes = bboxes[keep_indices]
        # orients = orients[keep_indices]
        # for i, box in enumerate(bboxes):
        #     temp_score = box[4] / 4
        #     box[0]*=scale_w
        #     box[2]*=scale_w
        #     box[1]*=scale_h
        #     box[3]*=scale_h
        #     box[5]*=scale_w
        #     box[6]*=scale_h
        #     if four_points:
        #         box = rotate_rect(box[0],box[1],box[2],box[3],orients[i],box[5],box[6])
        #     else:
        #         box = rotate_rect(box[0],box[1],box[2],box[3],0,pos[i,0],pos[i,1])
        #     for point in box:
        #         point[0] = int(round(point[0]))
        #         point[1] = int(round(point[1]))
        #         point[0] = max(0, point[0])
        #         point[0] = min(w-1, point[0])
        #         point[1] = max(0, point[1])
        #         point[1] = min(h-1, point[1])
        #     new_box = np.array((box[0][0],box[0][1],box[1][0],box[1][1],box[2][0],box[2][1],box[3][0],box[3][1],temp_score))
        #     new_boxes.append(new_box)
    new_boxes = np.array(new_boxes)
    keep_indices, temp_boxes = non_max_suppression_fast(new_boxes, nms_score)
    new_boxes = temp_boxes
    with open(os.path.join(save_dir, save_prefix+image_id+'.txt'), 'w') as fw:
        for i in keep_indices:
            temp_box = new_boxes[i]
            score = temp_box[-3]
            box=list()
            for j in range(4):
                box.append((int(round(temp_box[2*j])),int(round(temp_box[2*j+1]))))
            if not recog:
                cv2.line(origin_im, (box[0][0],box[0][1]), (box[1][0],box[1][1]), (255,0,0), 2)
                cv2.line(origin_im, (box[0][0],box[0][1]), (box[3][0],box[3][1]), (255,0,0), 2)
                cv2.line(origin_im, (box[2][0],box[2][1]), (box[1][0],box[1][1]), (255,0,0), 2)
                cv2.line(origin_im, (box[2][0],box[2][1]), (box[3][0],box[3][1]), (255,0,0), 2)
            else:
                x1 = box[0][0]
                y1 = box[0][1]
                x2 = box[1][0]
                y2 = box[1][1]
                x3 = box[2][0]
                y3 = box[2][1]
                x4 = box[3][0]
                y4 = box[3][1]
                width = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                height = math.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)
                pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
                pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst = cv2.warpPerspective(origin_im,M,(int(width), int(height)))
                if height > width * 2:
                    M = cv2.getRotationMatrix2D((width/2,width/2),90,1)
                    dst = cv2.warpAffine(dst,M,(int(height),int(width)))
                save_path = os.path.join(save_dir, save_prefix + image_id + '_' + str(i) + '.jpg')
                cv2.imwrite(save_path, dst)
                f.write(save_path + ' abchjaldj\n')
            if four_points:
                suffix = ''
                if output_score:
                    suffix = ','
                    suffix += str(score)
                if recog:
                    if suffix != ',':
                        suffix += ','
                    suffix += save_path
                fw.write('{},{},{},{},{},{},{},{}{}\r\n'.format(box[0][0],box[0][1],box[1][0],box[1][1],box[2][0],box[2][1],box[3][0],box[3][1],suffix))
            else:
                fw.write('{},{},{},{}\r\n'.format(box[0][0],box[0][1],box[2][0],box[2][1]))
    cv2.imwrite(os.path.join(save_dir, save_prefix+image_id+'.jpg'), origin_im)
if recog:
    f.close()
end=clock()
# print('total time:', end-start)
