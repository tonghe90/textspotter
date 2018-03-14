import sys
sys.path.append('/home/tonghe/caffe_st/python')
import caffe
import yaml
import random
import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon
from math import sin, cos, pi, exp, atan2

class ProposalRecogLayer(caffe.Layer):

	def setup(self, bottom, top):
		layer_param = yaml.load(self.param_str)
		self.batch_size = bottom[0].data.shape[0]
		self.phase = layer_param['phase']
		self.nms_score = layer_param['nms_score']
		self.scale = layer_param['scale']
		self.threshold = layer_param['threshold']
		self.bbox_scale = layer_param['bbox_scale']
		self.out_height = layer_param['out_height']
		self.keep_num = 0
		if self.phase == 1:
			if len(bottom) != 4:
				raise Exception('Need 4 inputs in training.') # score, bbox_orient, gt_bbox, ignore bbox
			if 'recog' in layer_param:
				self.recog = layer_param['recog']
			else:
				self.recog = False
			self.num_proposal = layer_param['num_proposal']
			self.fg_iou = layer_param['fg_iou']
			self.bg_iou = layer_param['bg_iou']
			self.ignore_iou = layer_param['ignore_iou']
			self.max_w = layer_param['max_w']
			self.select_pos = np.zeros((self.batch_size, self.num_proposal, 2), dtype=np.int32)
			top[0].reshape(self.batch_size, self.num_proposal, 11) # gt bbox
			top[1].reshape(self.batch_size, self.num_proposal, 1) # gt label
			top[2].reshape(self.batch_size, self.num_proposal, 6) # affine transform param
			top[3].reshape(self.batch_size, self.num_proposal) # w
			if self.recog:
				top[4].reshape(self.max_w, self.batch_size * self.num_proposal) # recog label
		else:
			if len(bottom) != 2:
				raise Exception('Need 2 inputs in testing.') # score, bbox_orient
			if self.batch_size != 1:
				raise Exception('Batch size must set to 1 in testing.')
			top[0].reshape(self.batch_size, 1, 11)
			top[1].reshape(self.batch_size, 1, 6)
			top[2].reshape(self.batch_size, 1)
			self.max_w = np.inf

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		l2_distance = lambda x,y:((x[0]-y[0])**2+(x[1]-y[1])**2)**0.5

		def rotate_points(points,degree,center_x,center_y):
			new_points = list()
			for point in points:
				dx = point[0] - center_x
				dy = point[1] - center_y
				new_x = center_x + dx * cos(degree) - dy * sin(degree)
				new_y = center_y + dx * sin(degree) + dy * cos(degree)
				new_points.append([(new_x), (new_y)])
			return new_points

		def dis_point2line(p0, p1, p2): # p2 is the point
			x0, y0 = p0
			x1, y1 = p1
			x2, y2 = p2
			A = y1-y0
			B = x0-x1
			C = x1*y0-x0*y1
			if A**2+B**2==0:
				dis = l2_distance(p0, p2)
			else:
				dis = abs(A*x2+B*y2+C)/((A**2+B**2)**0.5)
			return dis

		def rotate_rect(x1,y1,x2,y2,degree,center_x,center_y):
			points = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
			new_points = list()
			for point in points:
				dx = point[0] - center_x
				dy = point[1] - center_y
				new_x = center_x + dx * cos(degree) - dy * sin(degree)
				new_y = center_y + dx * sin(degree) + dy * cos(degree)
				new_points.append([(new_x), (new_y)])
			return new_points

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

		def non_max_suppression_fast(boxes, overlapThresh):
			min_score = 0
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
				score_sum = boxes[i][8] - min_score
				for k in range(8):
					new_boxes[i][k] *= score_sum
				for neighbor in neighbor_list:
					for k in range(8):
						new_boxes[i][k] += boxes[neighbor][k] * (boxes[neighbor][8] - min_score)
					score_sum += (boxes[neighbor][8] - min_score)
				for k in range(8):
					new_boxes[i][k] /= score_sum
			return pick, new_boxes
			
		N, C, H, W = bottom[0].data.shape
		score = bottom[0].data
		bbox = bottom[1].data
		if self.phase == 1:
			pos_num = 0
			neg_num = 0
			gt_bbox = bottom[2].data # NxCx9
			ignore_bbox = bottom[3].data # NxCx9
			top_bboxes=np.zeros((self.batch_size, self.num_proposal, 11), dtype=np.float32)
			top_labels=np.empty((self.batch_size, self.num_proposal, 1), dtype=np.float32)
			top_labels.fill(255)
			top_param=np.zeros((self.batch_size, self.num_proposal, 6), dtype=np.float32)
			top_w=np.zeros((self.batch_size, self.num_proposal), dtype=np.float32)
			if self.recog:
				top_recog_label=np.zeros((self.batch_size, self.num_proposal, self.max_w), dtype=np.float32)
		for i in range(N):
			if self.phase == 0:
				pre_score = score[i, 1, :, :]
				pre_bbox = bbox[i, :, :, :]
				pos_y, pos_x = np.where(pre_score>self.threshold)
				if self.phase == 1:
					temp_threshold = self.threshold
					while len(pos_y) < self.num_proposal:
						temp_threshold -= 0.1
						if temp_threshold < 0:
							pos_y, pos_x = np.where(np.ones(pre_score.shape)>0)
							break
						pos_y, pos_x = np.where(pre_score>temp_threshold)
				bottom_info = np.zeros((len(pos_y), 7),dtype='float32')
				bboxes = np.zeros((len(pos_y), 7),dtype='float32')
				orients = np.zeros((len(pos_y)),dtype='float32')
				orient_bboxes = np.zeros((len(pos_y), 11),dtype='float32')
				dets = np.zeros((len(pos_y), 5),dtype='float32')
				for j in range(len(pos_y)):
					y = pos_y[j]
					x = pos_x[j]
					t, b, l, r, orient = pre_bbox[:,y,x]
					t *= self.bbox_scale
					b *= self.bbox_scale
					l *= self.bbox_scale
					r *= self.bbox_scale
					dets[j] = np.asarray((x - l, y - t, x + r, y + b, pre_score[y,x]))
					bottom_info[j] = np.asarray((t, b, l, r, orient ,x,y))
					bboxes[j] = np.asarray((x-l,y-t,x+r,y+b,pre_score[y,x],x,y))
					orients[j] = orient
				for j, box in enumerate(bboxes):
					temp_box = rotate_rect(box[0],box[1],box[2],box[3],orients[j],box[5],box[6])
					orient_bboxes[j] = np.array((temp_box[0][0],temp_box[0][1],temp_box[1][0],temp_box[1][1],temp_box[2][0],temp_box[2][1],temp_box[3][0],temp_box[3][1],box[4],box[-2],box[-1]))
			if self.phase == 1:
				gt_bbox_i = gt_bbox[i]
				gt_bbox_i = gt_bbox_i[0, :, :]
				gt_bbox_i = gt_bbox_i[np.where(gt_bbox_i[:,8]==1)]
				keep = range(len(gt_bbox_i))
				if len(keep) > self.num_proposal:
					keep = random.sample(keep, self.num_proposal)
					gt_bbox_i = gt_bbox_i[keep]
				self.keep_num = len(keep)
				gt_orient_bboxes = np.zeros((self.keep_num, 11),dtype='float32')
				gt_bottom_info = np.zeros((self.keep_num, 7),dtype='float32')
				gt_recog_label = np.zeros((self.keep_num, self.max_w),dtype='int32')
				for n, temp_bbox in enumerate(gt_bbox_i):
					if temp_bbox[9]>self.max_w-1:
						gt_recog_label[n][0] = -1
					else:
						for j in range(int(temp_bbox[9])):
							gt_recog_label[n][j] = int(temp_bbox[10+j])
						gt_recog_label[n][int(temp_bbox[9])] = -1
					for j in range(8):
						temp_bbox[j] = int(round(temp_bbox[j]))
					rect = cv2.minAreaRect(np.asarray([[temp_bbox[0],temp_bbox[1]], [temp_bbox[2],temp_bbox[3]], [temp_bbox[4],temp_bbox[5]], [temp_bbox[6],temp_bbox[7]]], dtype=np.int32))
					new_box = cv2.cv.BoxPoints(rect)
					dis = np.zeros((4, 4))
					for k in range(4):
						for j in range(4):
							dis[k][j] = l2_distance([temp_bbox[2*k], temp_bbox[2*k+1]], new_box[j])
					relation = np.zeros(4, dtype=np.int32)
					unuse = np.ones(4)
					for k in range(4):
						j = np.argmin(dis[k]*unuse)
						unuse[j] = np.inf
						relation[k] = j
					new_box = np.asarray(new_box)
					# print('before',new_box)
					new_box = new_box[relation]
					# print('after',new_box)
					gt_center_x = np.sum(new_box[:, 0]) / 4
					gt_center_y = np.sum(new_box[:, 1]) / 4
					# if l2_distance(new_box[0], new_box[1]) * 2 < l2_distance(new_box[0], new_box[3]):
						# new_box = np.array(rotate_points(new_box, pi / 2, gt_center_x, gt_center_y))
						# new_box = new_box[np.array((1,2,3,0))]
					gt_orient = atan2(new_box[1][1] - new_box[0][1], new_box[1][0] - new_box[0][0])
					gt_t = dis_point2line(new_box[0], new_box[1], [gt_center_x, gt_center_y])
					gt_b = dis_point2line(new_box[2], new_box[3], [gt_center_x, gt_center_y])
					gt_l = dis_point2line(new_box[0], new_box[3], [gt_center_x, gt_center_y])
					gt_r = dis_point2line(new_box[1], new_box[2], [gt_center_x, gt_center_y])
					gt_orient_bboxes[n] = np.array((new_box[0][0], new_box[0][1], new_box[1][0], new_box[1][1], new_box[2][0], new_box[2][1], new_box[3][0], new_box[3][1], 1, gt_center_x, gt_center_y))
					gt_bottom_info[n] = np.array((gt_t, gt_b, gt_l, gt_r, gt_orient, gt_center_x, gt_center_y))
				top_recog_label[i][:self.keep_num] = gt_recog_label
				orient_bboxes = gt_orient_bboxes
				bottom_info = gt_bottom_info
			# testing
			if self.phase == 0:
				keep_indices, temp_boxes = non_max_suppression_fast(orient_bboxes, self.nms_score)
				orient_bboxes = temp_boxes
				bottom_info = bottom_info[keep_indices]
				# keep_indices = range(len(orient_bboxes))
				top_length = max(len(keep_indices), 1)
				top[0].reshape(N, top_length, 11)
				top[1].reshape(N, top_length, 6)
				top[2].reshape(N, top_length)
				top_param=np.zeros((self.batch_size, len(keep_indices), 6), dtype=np.float32)
				top_w=np.zeros((self.batch_size, len(keep_indices)), dtype=np.float32)
				j=0
				for index in keep_indices:
					# for k in range(4):
						# orient_bboxes[index][2*k] = int(round(orient_bboxes[index][2*k]))
						# orient_bboxes[index][2*k] = max(0, orient_bboxes[index][2*k])
						# orient_bboxes[index][2*k] = min(W, orient_bboxes[index][2*k])
						# orient_bboxes[index][2*k+1] = int(round(orient_bboxes[index][2*k+1]))
						# orient_bboxes[index][2*k+1] = max(0, orient_bboxes[index][2*k+1])
						# orient_bboxes[index][2*k+1] = min(H, orient_bboxes[index][2*k+1])
					top[0].data[i, j, :] = np.array(orient_bboxes[index])
					j+=1
			# training
			else:
				top_bboxes[i,:self.keep_num,:] = orient_bboxes[:, :11]
				top_labels[i][:self.keep_num] = 1
				top_labels[i][np.where(gt_recog_label[:, 0]==-1)] = 255
				# print(top_labels[i])
				# print(gt_recog_label)
			for j, info in enumerate(bottom_info):
				t, b, l, r, orient,x,y = info * self.scale
				# t*=1.5
				# b*=1.5
				# l*=1.5
				# r*=1.5
				orient /= self.scale
				if ((t+b)>(l+r)*2):
					t,b,l,r=l,r,b,t
					orient+=pi/2
				orient = -orient
				dx = -l
				dy = -t
				new_x = x + dx * cos(-orient) - dy * sin(-orient)
				new_y = y + dx * sin(-orient) + dy * cos(-orient)
				tx = -new_x
				ty = -new_y
				if t+b < 0.01:
					scale = 1
					top_labels[i][j] = 255
				else:
					scale = self.out_height / (t+b)
				top_w[i][j] = min(self.max_w, round(scale * (l+r)))
				top_param[i][j] = np.array([cos(orient)/scale, sin(orient)/scale, -tx, \
					-sin(orient)/scale, cos(orient)/scale, -ty])
				if len(bottom) == 5:
					# if top_labels[i,j] != 1:
					# 	continue
					feature = bottom[4].data[i]
					M = np.float32([[scale*cos(orient), -scale*sin(orient), scale*(tx*cos(orient)-ty*sin(orient))], \
						[scale*sin(orient), scale*cos(orient), scale*(tx*sin(orient)+ty*cos(orient))]])
					for c in range(bottom[4].shape[1]):
						top_feature[i*self.num_proposal+j, c] = cv2.warpAffine(feature[c], M, (self.out_width, self.out_height))
		if self.phase == 1:
			# print('pos num:', pos_num)
			# print('neg num:', neg_num)
			top[0].data[...]=top_bboxes
			top[1].data[...]=top_labels
			top[2].data[...]=top_param
			top[3].data[...]=top_w
			# print(top[1].data[0])
			# print(top[3].data[0])
			# print(top_recog_label[0])
			top[4].data[...]=top_recog_label.reshape((self.batch_size * self.num_proposal, self.max_w)).transpose()
		else:
			if len(keep_indices) > 0:
				top[1].data[...]=top_param
				top[2].data[...]=top_w
				
	def backward(self, top, propagate_down, bottom):
		pass
