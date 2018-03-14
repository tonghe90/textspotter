import caffe
import yaml
import cv2
import os
import numpy as np

class DrawAffineLayer(caffe.Layer):

	def setup(self, bottom, top):
		layer_param = yaml.load(self.param_str)
		self.save_dir = layer_param['save_dir']
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		# if len(bottom) != 2:
		# 	raise Exception('Need 2 inputs in training.')
		if bottom[0].data.shape[1] != 3 and bottom[0].data.shape[1] != 1:
			raise Exception('bottom must have 1 or 3 channels.')

	def reshape(self, bottom, top):
		pass

	def backward(self, top, propagate_down, bottom):
		pass

	def forward(self, bottom, top):
		bottom_data = bottom[0].data
		if len(bottom) == 2:
			bottom_label = bottom[1].data
			for i in range(bottom_label.shape[0]):
				for j in range(bottom_label.shape[1]):
					if bottom_label[i, j] == 1:
						prefix = 'fg'
					elif bottom_label[i, j] == 0:
						prefix = 'bg'
					else:
						prefix = 'ignore'
					index = i * bottom_label.shape[1] + j
					img = bottom_data[index]
					img = np.transpose(img, (1, 2, 0))
					img += 122
					cv2.imwrite(os.path.join(self.save_dir, '{}_{}.jpg'.format(prefix, str(index))), img)
		else:
			for index in range(bottom_data.shape[0]):
				img = bottom_data[index]
				img = np.transpose(img, (1, 2, 0))
				img += 122
				cv2.imwrite(os.path.join(self.save_dir, '{}_{}.jpg'.format('fg', str(index))), img)