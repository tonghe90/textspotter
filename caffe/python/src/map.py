import sys
import pylab as pl
from shapely.geometry import Polygon
import os
import cv2

plot = False
save_error = False
# %matplotlib inline

predict_path = '/data1/liuxuebo/data/icdar/icdar17_result_test5'
ground_truth_path = '/data1/liuxuebo/data/icdar2017rctw_train_v1.2/val'
error_save_dir = '/data1/liuxuebo/data/icdar/icdar17_result_test_error'

if not os.path.exists(error_save_dir):
	os.makedirs(error_save_dir)

predict_file_list = map(lambda x: os.path.join(predict_path, x), [im_name for im_name in os.listdir(predict_path) if (im_name.endswith('.txt') and not im_name.startswith('.'))])
ground_truth_file_list = map(lambda x: os.path.join(ground_truth_path, x), [im_name for im_name in os.listdir(ground_truth_path) if (im_name.endswith('.txt') and not im_name.startswith('.'))])
predict_dict = dict()
ground_truth_dict = dict()

iou_threshold = 0.5
def get_info_predict(info_file, info_dict):
	bbox_num = 0
	if(info_file.endswith('txt')):
		with open(info_file) as fr:
			for line in fr:
				im_id = info_file.split('/')[-1].split('.')[0].split('_')[1] + '_' + info_file.split('/')[-1].split('.')[0].split('_')[2]
				x1,y1,x2,y2,x3,y3,x4,y4,score = map(float, line.strip().split(','))
				if not im_id in info_dict:
					info_dict[im_id] = list()
				info_dict[im_id].append([x1,y1,x2,y2,x3,y3,x4,y4,score])
				bbox_num+=1
	return bbox_num

def get_info_ground_truth(info_file, info_dict, predict_dict):
	bbox_num_easy = 0
	bbox_num_hard = 0
	with open(info_file) as fr:
		for line in fr:
			im_id = info_file.split('/')[-1].split('.')[0]
			if not im_id in predict_dict:
				return 0,0
			x1 = float(line.strip().split(',')[0].split('\xef\xbb\xbf')[-1])
			y1,x2,y2,x3,y3,x4,y4,difficulty = map(float, line.strip().split(',')[1:9])
			if not im_id in info_dict:
				info_dict[im_id] = list()
			info_dict[im_id].append([x1,y1,x2,y2,x3,y3,x4,y4,difficulty])
			if(difficulty == 0):
				bbox_num_easy += 1
			if(difficulty == 1):
				bbox_num_hard += 1
	return bbox_num_easy, bbox_num_hard

predict_bbox_num = 0
ground_truth_bbox_num_easy = 0
ground_truth_bbox_num_hard = 0
for predict_file in predict_file_list:
	predict_bbox_num += get_info_predict(predict_file, predict_dict)
for ground_truth_file in ground_truth_file_list:
	num_easy, num_hard = get_info_ground_truth(ground_truth_file, ground_truth_dict, predict_dict)
	ground_truth_bbox_num_easy += num_easy
	ground_truth_bbox_num_hard += num_hard
print('predict_bbox_num', predict_bbox_num)
print('ground_truth_bbox_num', ground_truth_bbox_num_easy+ground_truth_bbox_num_hard)

def compute_iou(predict_bbox, ground_truth_bbox):
	predict_area = Polygon([(predict_bbox[0],predict_bbox[1]), (predict_bbox[2],predict_bbox[3]),(predict_bbox[4],predict_bbox[5]),(predict_bbox[6],predict_bbox[7])])
	ground_truth_area = Polygon([(ground_truth_bbox[0],ground_truth_bbox[1]), (ground_truth_bbox[2],ground_truth_bbox[3]),(ground_truth_bbox[4],ground_truth_bbox[5]),(ground_truth_bbox[6],ground_truth_bbox[7])])
	inter_area = predict_area.intersection(ground_truth_area).area
	return inter_area / (predict_area.area+ground_truth_area.area-inter_area)

# def compare(predict_list, ground_truth_list, score_list, match_list):
#     ground_truth_unuse = [True for i in range(len(ground_truth_list))]
#     predict_unuse = [True for i in range(len(predict_list))]
#     iou_list = list()
#     for i, predict_bbox in enumerate(predict_list):
#         for j, ground_truth_bbox in enumerate(ground_truth_list):
#             iou_list.append((compute_iou(predict_bbox, ground_truth_bbox), i, j))
#     iou_list.sort(key = lambda x:-x[0])
#     for iou, i, j in iou_list:
#         match = 0
#         if predict_unuse[i] and ground_truth_unuse[j]:
#             if iou > iou_threshold:
#                 if ground_truth_list[j][-1] == 0:
#                     match = 1
#                 if(ground_truth_list[i][-1] == 1):
#                     match = 2
#                 predict_unuse[i] = False
#                 ground_truth_unuse[j] = False
#         score_list.append(predict_list[i][-1])
#         match_list.append(match)

def compare(predict_list, ground_truth_list, score_list, match_list, unmatch_list):
	ground_truth_unuse = [True for i in range(len(ground_truth_list))]
	for predict_bbox in predict_list:
		match = 0
		for i in range(len(ground_truth_list)):
			if ground_truth_unuse[i]:
				if compute_iou(predict_bbox, ground_truth_list[i])>iou_threshold:
					if(ground_truth_list[i][-1] == 0):
						match = 1
						ground_truth_unuse[i] = False
					if(ground_truth_list[i][-1] == 1):
						match = 2
					break
		score_list.append(predict_bbox[-1])
		match_list.append(match)
	for i in range(len(ground_truth_unuse)):
		if ground_truth_unuse[i] and ground_truth_list[i][-1] == 0:
			unmatch_list.append(ground_truth_list[i])

score_list = list()
match_list = list()
unmatch_dict = dict()
for key in predict_dict:
	predict_dict[key].sort(key = lambda x:x[-1], reverse = True)
	unmatch_dict[key] = list()
	compare(predict_dict[key], ground_truth_dict[key], score_list, match_list, unmatch_dict[key])
if save_error:
	for key in unmatch_dict:
		im = cv2.imread(os.path.join(ground_truth_path, key + '.jpg'))
		for box in unmatch_dict[key]:
			box = list(map(int, box))
			cv2.line(im, (box[0],box[1]), (box[2],box[3]), (0,255,0), 2)
			cv2.line(im, (box[0],box[1]), (box[6],box[7]), (0,255,0), 2)
			cv2.line(im, (box[4],box[5]), (box[2],box[3]), (0,255,0), 2)
			cv2.line(im, (box[4],box[5]), (box[6],box[7]), (0,255,0), 2)
		cv2.imwrite(os.path.join(error_save_dir, key + '.jpg'), im)
p = list()
r = list()
predict_num = 0
truth_num_p = 0
truth_num_r = 0

score_match_list = list(zip(score_list, match_list))
score_match_list.sort(key = lambda x:x[0], reverse = True)
for item in score_match_list:
	predict_num += 1
	if(item[1] == 2):
		predict_num -= 1
	if(item[1] == 1):
		truth_num_p += 1
	if float(predict_num) == 0:
		p.append(1)
	else:
		p.append(float(truth_num_p)/float(predict_num))
	r.append(float(truth_num_p)/float(ground_truth_bbox_num_easy))
mAP = 0

for i in range(1,len(r)):
	mAP += (p[i-1]+p[i])/2*(r[i]-r[i-1])
print('mAP:{}'.format(mAP))
print('max p:{}'.format(p[0]))
print('max r:{}'.format(r[-1]))
if plot:
	pl.xlim(0,1)
	pl.ylim(0,1)
	pl.plot(r,p)
	pl.show()
