﻿import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models

class RPN(nn.module):
	''' Region Proposal Network as a component of the Faster-rcnn net
	'''
	def __init__(self, RPN):
		super.__init__(self, RPN)
		self.feature_net = models.vgg16() # The covnets that outputs the feature map
		self.
		

	@staticmethod
	def generate_anchor(img, stride = 16, scales = [128, 256, 512], ratios = [0.5, 1, 2]):
		'''	Generating the 4 parameters (x1, y1, x2, y2) of each anchor.
			And concatenate them along the 0-th dimension.
			---------------------------------
			img: input image matrix (N, C, H, W)- 4 dimensions                 x1, y1--------+
			stride: the stride that the sliding window woudl move                 |          |
			scales: the width of anchor (when it is a square)                     |          |
			ratios: under certain scale, the ratio between width and height       +-------x2, y2
			---------------------------------
			output: a tensor (H*W, 4) a series of anchor parameters, for the memory efficiency
		'''
		_num_anchors = img.shape()[-2] * img.shape()[-1]
		x_ctr, y_ctr = torch.meshgrid([img.shape()[2], img.shape()[3]])
		# keep the array into one dimension, (2 dimension in all)
		x_ctr = x_ctr.reshape((-1, 1)).expand(_num_anchors * len(scales) * len(ratios), 1)
		y_ctr = y_ctr.reshape((-1, 1)).expand(_num_anchors * len(scales) * len(ratios), 1)
		
		# prepare to generate different shape
		base = torch.ones((_num_anchors, 1))
		h_seq = []
		w_seq = []

		# concatenate each anchor parameters (height and width) together
		for scale in scales:
			for ratio in ratios:
				h_seq.append(
					torch.round(torch.sqrt(torch.mul(base, scale * scale * ratio)))
					)
				w_seq.append(
					torch.round(torch.sqrt(torch.mul(base, scale * scale / ratio)))
					)
		h_half = torch.round(torch.mul(torch.cat(h_seq, 0).reshape((-1, 1)), 0.5))
		w_half = torch.round(torch.mul(torch.cat(w_seq, 0).reshape((-1, 1)), 0.5))

		anchors = [
			x_ctr - h_half,
			y_ctr - w_half,
			x_ctr + h_half,
			y_ctr + w_half
			]
		anchors = torch.cat(anchors, 1).contiguous()

		return anchors

	def forward(self, x):
		pass

class FasterRCNN(nn.module):
	def __init__(self):
		super.__init__(self, FasterRCNN)

	def forward(self, x):
		pass