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
from rotated_nms import non_max_suppression_fast
gpu_id = 2
model_file = '/data1/liuxuebo/text_detect/models/e2e_best.caffemodel'
net_def_file = '/data1/liuxuebo/text_detect/test_e2e.pt'
image_dir = '/data1/liuxuebo/data/icdar/ch4_test_images'
# image_dir = '/data1/liuxuebo/data/icdar/Challenge2_Test_Task12_Images'
# image_dir = '/data1/liuxuebo/datauawei_500/train'
save_dir = '/data1/liuxuebo/data/icdar/icdar15_result_test_e2e_recog'
recog_dict = '/data1/liuxuebo/recog/dict15.txt'
image_list = []
# dont support multi scale test
image_resize_length_list = [2240]
# image_resize_length_list = [2560, 2240, 1920, 1600, 1280]
# image_resize_height = 1248
# image_resize_width = 2240
mean_value = 122
nms_score = 0.15
out_recog = True
# 1: using the label for per image; 2: using the global label; 3: not compare
compare_option = 1
compare_global_label_path = '/data1/guominghao/ch4_test_vocabulary.txt'
compare_per_image_label_path = '/data1/liuxuebo/data/icdar/ch4_test_vocabularies_per_image'
# threshold for levenshteinDistance, larger for more recognition
threshold = 0.5


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if image_dir != '':
    image_list.extend(list(map(lambda x:os.path.join(image_dir, x), [im_name for im_name in os.listdir(image_dir) if (im_name.endswith('.jpg') and not im_name.startswith('.'))])))

recog_dict_list = list()
with open(recog_dict) as fr:
    for line in fr:
        recog_dict_list.append(line.strip())

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def compare_str(s, dict_):
    if s.isdigit():
        s_out = s
    else:
        distance = list()
        for cell in dict_:
            distance.append(levenshteinDistance(s.upper(), cell.upper()))
        # TODO: multi min distance
        min_dis = distance.index(min(distance))
        if len(s)==0 or float(min(distance)) / (len(s)) > threshold:
            s_out = None
        s_out = dict_[min_dis]
    return(s_out)

compare_global_dict = list()
if (compare_option == 2):
    with open(compare_global_label_path) as fr:
        compare_global_dict = fr.read().strip().split()

def compare_dict(s, name, option):
    if (option == 1):
        dict_ = list()
        dict_name = 'voc_' + name + '.txt'
        with open('{}/{}'.format(compare_per_image_label_path, dict_name)) as fr_tmp:
            dict_ = fr_tmp.read().strip().split()
        s_out = compare_str(s, dict_)
    if (option == 2):
        s_out = compare_str(s, compare_global_dict)
    if (option == 3):
        s_out = s
    return(s_out)

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
    # if image != '/data1/liuxuebo/data/icdar/ch4_test_images/img_2.jpg':
    #     continue
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
        pre_recog=caffe_model.blob('StandardCTC_test')
        # pre_bbox_score=caffe_model.blob('bbox_pre_score')[:, 1, 0, 0]
        # # pre_bbox[:,:,-3]=pre_bbox_score
        # # stage1_score = pre_bbox[0, :, 8]
        # # pre_bbox_score += stage1_score
        # # pre_bbox_score /= 2
        # keep_indices=np.where(pre_bbox_score>min_score)
        # if len(keep_indices[0]) != pre_bbox.shape[1]:
        #     print(pre_bbox_score)
        #     print('before:', pre_bbox.shape[1])
        #     print('after:', len(keep_indices[0]))
        # pre_bbox=pre_bbox[:, keep_indices[0], :]
        pre_bbox=pre_bbox[0]
        keep_indices=np.where(pre_bbox[:, 8]>0)[0]
        pre_bbox = pre_bbox[keep_indices]
        pre_recog = pre_recog[:, keep_indices]
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
            recog_result = ''
            for j in pre_recog[:, i]:
                j=int(j)
                if j==-1:
                    break
                recog_result+=recog_dict_list[j-1]
            recog_result = compare_dict(recog_result, image_id, compare_option)
            if(recog_result == None):
                continue

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
            if out_recog:
                fw.write('{},{},{},{},{},{},{},{},{}\r\n'.format(box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7],recog_result))
            else:
                fw.write('{},{},{},{},{},{},{},{}\r\n'.format(box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7]))
            cv2.line(origin_im, (box[0],box[1]), (box[2],box[3]), (255,0,0), 2)
            cv2.line(origin_im, (box[0],box[1]), (box[6],box[7]), (255,0,0), 2)
            cv2.line(origin_im, (box[4],box[5]), (box[2],box[3]), (255,0,0), 2)
            cv2.line(origin_im, (box[4],box[5]), (box[6],box[7]), (255,0,0), 2)
        cv2.imwrite(os.path.join(save_dir, 'res_'+image_id+'.jpg'), origin_im)





