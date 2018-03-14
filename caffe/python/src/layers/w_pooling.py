import caffe
import yaml
import cv2
import os
import numpy as np

class WPoolingLayer(caffe.Layer):

	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception('Need 2 inputs in training.')
		layer_param = yaml.load(self.param_str)
		self.pooling_method = layer_param['pooling_method']
		if self.pooling_method != 'ave' and self.pooling_method != 'last':
			raise Exception('pooling method must be ave or last.')
		N, C, H, W = bottom[0].data.shape
		T, N2 = bottom[1].data.shape
		if N != N2 or T != W:
			raise Exception('bottom\'s shape doesn\'t match')

	def reshape(self, bottom, top):
		N, C, H, W = bottom[0].data.shape
		top[0].reshape(N, C, H, 1)

	def forward(self, bottom, top):
		feature = bottom[0].data
		cont = bottom[1].data
		top_data = top[0].data
		N, C, H, W = feature.shape
		for n in range(N):
			length = int(-cont[0][n])
			if self.pooling_method=='ave':
				if length > 0:
					top_data[n, :, :, 0] = np.sum(feature[n, :, :, :length], axis=-1) / length
			elif self.pooling_method=='last':
				top_data[n, :, :, 0] = feature[n, :, :, length-1]

	def backward(self, top, propagate_down, bottom):
		if propagate_down[0]:
			bottom_diff = bottom[0].diff
			top_diff = top[0].diff
			feature = bottom[0].data
			cont = bottom[1].data
			top_data = top[0].data
			N, C, H, W = feature.shape
			for n in range(N):
				length = int(-cont[0][n])
				if self.pooling_method=='ave':
					for l in range(length):
						if length > 0:
							bottom_diff[n, :, :, l] = top_diff[n, :, :, 0] / length
				elif self.pooling_method=='last':
					bottom_diff[n, :, :, length-1] = top_diff[n, :, :, 0]
