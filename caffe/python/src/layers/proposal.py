import caffe
import yaml
import random
import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon
from math import sin, cos, pi, exp

class ProposalLayer(caffe.Layer):

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
			if len(bottom) != 4 and len(bottom) != 5:
				raise Exception('Need 4 or 5 inputs in training.') # score, bbox_orient, gt_bbox, ignore bbox
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
			if len(bottom) == 5:
				self.out_width = layer_param['out_width']
				top[4].reshape(self.batch_size * self.num_proposal, bottom[4].data.shape[1], self.out_height, self.out_width) # feature
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

		def iou(bboxes1, bboxes2):
			area1_list = list()
			area2_list = list()
			iou_list = list()
			for bbox1 in bboxes1:
				p1 = Polygon([(bbox1[0],bbox1[1]),(bbox1[2],bbox1[3]),(bbox1[4],bbox1[5]),(bbox1[6],bbox1[7])])
				area1_list.append([p1, p1.area])
			for bbox2 in bboxes2:
				p2 = Polygon([(bbox2[0],bbox2[1]),(bbox2[2],bbox2[3]),(bbox2[4],bbox2[5]),(bbox2[6],bbox2[7])])
				area2_list.append([p2, p2.area])
			for p1, p1_area in area1_list:
				temp_iou = list()
				for p2, p2_area in area2_list:
					inter = p1.intersection(p2).area
					temp_iou.append(inter / (p1_area + p2_area - inter))
				if len(temp_iou) > 0:
					iou_list.append(max(temp_iou))
				else:
					iou_list.append(0)
			return np.array(iou_list)

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
			if len(bottom) == 5:
				top_feature = np.zeros((self.batch_size * self.num_proposal, bottom[4].data.shape[1], self.out_height, self.out_width), dtype=np.float32)
		for i in range(N):
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
				keep = py_cpu_nms(dets, self.nms_score)
				if len(keep) > self.num_proposal:
					keep = random.sample(keep, self.num_proposal)
				self.keep_num = len(keep)
				# if len(keep) < self.num_proposal:
				# 	temp = list()
				# 	keep_set = set(keep)
				# 	for j in range(len(dets)):
				# 		if not j in keep_set:
				# 			temp.append(j)
				# 	keep.extend(random.sample(temp, self.num_proposal - len(keep)))
				orient_bboxes = orient_bboxes[keep]
				bottom_info = bottom_info[keep]
				self.select_pos[i, :self.keep_num, 0] = pos_x[keep]
				self.select_pos[i, :self.keep_num, 1] = pos_y[keep]
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
				gt_bbox_i = gt_bbox[i]
				gt_bbox_i = gt_bbox_i[0, :, :9]
				gt_bbox_i = gt_bbox_i[np.where(gt_bbox_i[:,-1]==1)]
				ignore_bbox_i = ignore_bbox[i]
				ignore_bbox_i = ignore_bbox_i[0, :, :9]
				ignore_bbox_i = ignore_bbox_i[np.where(ignore_bbox_i[:,-1]==1)]
				top_bboxes[i,:self.keep_num,:] = orient_bboxes[:, :11]
				ignore_bbox_iou_list = iou(orient_bboxes, ignore_bbox_i)
				ignore_indices = np.where(ignore_bbox_iou_list > self.ignore_iou)
				# orient_bboxes = orient_bboxes[keep_indices]
				gt_bbox_iou_list = iou(orient_bboxes, gt_bbox_i)
				# print('gt_bbox_iou_list',gt_bbox_iou_list)
				fg_indices = np.where(gt_bbox_iou_list > self.fg_iou)
				bg_indices = np.where(gt_bbox_iou_list < self.bg_iou)
				# print('fg_indices',fg_indices)
				# fg_indices = np.intersect1d(keep_indices[0], fg_indices[0], assume_unique=True)
				# bg_indices = np.intersect1d(keep_indices[0], bg_indices[0], assume_unique=True)
				# print('fg:',len(fg_indices[0]))
				# print('bg:',len(bg_indices[0]))
				pos_num += len(fg_indices[0])
				neg_num += len(bg_indices[0])
				top_labels[i][fg_indices] = 1
				top_labels[i][bg_indices] = 0
				top_labels[i][ignore_indices] = 255
				# print('top_labels',top_labels[i])
				# print(self.keep_num)
			for j, info in enumerate(bottom_info):
				t, b, l, r, orient,x,y = info * self.scale
				orient /= self.scale
				if ((t+b)>(l+r)):
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
			if len(bottom) == 5:
				top[4].data[...]=top_feature
		else:
			if len(keep_indices) > 0:
				top[1].data[...]=top_param
				top[2].data[...]=top_w

	def backward(self, top, propagate_down, bottom):
		if propagate_down[1]:
			bottom_diff = np.zeros(bottom[1].data.shape, dtype=np.float32)
			top_diff = top[2].diff
			bottom_data = bottom[1].data
			for i in range(self.batch_size):
				for j in range(self.num_proposal):
					if j >= self.keep_num:
						break
					temp_top_diff = top_diff[i][j]
					x, y = self.select_pos[i, j]
					t, b, l, r, th = bottom_data[i,:,y,x]
					rotated = False
					if ((t+b)>(l+r)):
						rotated = True
						t,b,l,r=l,r,b,t
						th+=pi/2
					t *= self.bbox_scale
					b *= self.bbox_scale
					l *= self.bbox_scale
					r *= self.bbox_scale
					t *= self.scale
					b *= self.scale
					l *= self.scale
					r *= self.scale
					diff_t = 0
					diff_b = 0
					diff_l = 0
					diff_r = 0
					diff_orient = 0
					dst_h = self.out_height
					# compute diff_orient
					# diff_t += temp_top_diff[0] * (-dst_h*cos(th)/max((b + t)**2, 1e-5))
					# diff_b += temp_top_diff[0] * (-dst_h*cos(th)/max((b + t)**2, 1e-5))
					# diff_orient += temp_top_diff[0] * (-dst_h*sin(th)/max((b + t), 1e-5))
					# diff_t += temp_top_diff[4] * (-dst_h*cos(th)/max((b + t)**2, 1e-5))
					# diff_b += temp_top_diff[4] * (-dst_h*cos(th)/max((b + t)**2, 1e-5))
					# diff_orient += temp_top_diff[4] * (-dst_h*sin(th)/max((b + t), 1e-5))
					# diff_t += temp_top_diff[3] * (-dst_h*sin(th)/max((b + t)**2, 1e-5))
					# diff_b += temp_top_diff[3] * (-dst_h*sin(th)/max((b + t)**2, 1e-5))
					# diff_orient += temp_top_diff[3] * (dst_h*cos(th)/max((b + t), 1e-5))
					# diff_t += temp_top_diff[1] * (dst_h*sin(th)/max((b + t)**2, 1e-5))
					# diff_b += temp_top_diff[1] * (dst_h*sin(th)/max((b + t)**2, 1e-5))
					# diff_orient += temp_top_diff[1] * (-dst_h*cos(th)/max((b + t), 1e-5))
					# diff_t += temp_top_diff[2] * (-dst_h*sin(th)/max((b + t), 1e-5) - dst_h*((l - x)*cos(th) - (t - y)*sin(th))/max((b + t)**2, 1e-5))
					# diff_b += temp_top_diff[2] * (-dst_h*((l - x)*cos(th) - (t - y)*sin(th))/max((b + t)**2, 1e-5))
					# diff_orient += temp_top_diff[2] * (dst_h*(-(l - x)*sin(th) - (t - y)*cos(th))/max((b + t), 1e-5))
					# diff_l += temp_top_diff[2] * dst_h*cos(th)/max((b + t), 1e-5)
					# diff_t += temp_top_diff[5] * (dst_h*cos(th)/max((b + t), 1e-5) - dst_h*((l - x)*sin(th) + (t - y)*cos(th))/max((b + t)**2, 1e-5))
					# diff_b += temp_top_diff[5] * (-dst_h*((l - x)*sin(th) + (t - y)*cos(th))/max((b + t)**2, 1e-5))
					# diff_orient += temp_top_diff[5] * (dst_h*((l - x)*cos(th) - (t - y)*sin(th))/max((b + t), 1e-5))
					# diff_l += temp_top_diff[5] * (dst_h*sin(th)/max((b + t), 1e-5))
					diff_t += temp_top_diff[0] * cos(th)/dst_h
					diff_b += temp_top_diff[0] * cos(th)/dst_h
					diff_orient += temp_top_diff[0] * (-(b + t)*sin(th)/dst_h)
					diff_t += temp_top_diff[4] * cos(th)/dst_h
					diff_b += temp_top_diff[4] * cos(th)/dst_h
					diff_orient += temp_top_diff[4] * (-(b + t)*sin(th)/dst_h)
					diff_t += temp_top_diff[1] * (-sin(th)/dst_h)
					diff_b += temp_top_diff[1] * (-sin(th)/dst_h)
					diff_orient += temp_top_diff[1] * (-(b + t)*cos(th)/dst_h)
					diff_t += temp_top_diff[3] * (sin(th)/dst_h)
					diff_b += temp_top_diff[3] * (sin(th)/dst_h)
					diff_orient += temp_top_diff[3] * ((b + t)*cos(th)/dst_h)
					diff_t += temp_top_diff[2]*sin(th)
					diff_l += temp_top_diff[2]*(-cos(th))
					diff_orient += temp_top_diff[2]*(l*sin(th) + t*cos(th))
					diff_t += temp_top_diff[5]*(-cos(th))
					diff_l += temp_top_diff[5]*(-sin(th))
					diff_orient += temp_top_diff[5]*(-l*cos(th) + t*sin(th))
					if rotated:
						diff_l, diff_r, diff_t, diff_b = diff_t, diff_b, diff_r, diff_l
					# print(diff_t, diff_b, diff_l, diff_r)
					bottom_diff[i,0,y,x] = self.bbox_scale * self.scale * diff_t
					bottom_diff[i,1,y,x] = self.bbox_scale * self.scale * diff_b
					bottom_diff[i,2,y,x] = self.bbox_scale * self.scale * diff_l
					bottom_diff[i,3,y,x] = self.bbox_scale * self.scale * diff_r
					bottom_diff[i,4,y,x] = diff_orient
			bottom[1].diff[...] = bottom_diff
