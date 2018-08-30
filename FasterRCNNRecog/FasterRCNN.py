import numpy as np

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
	def generate_anchor(img, scales = [128, 256, 512], ratios = [0.5, 1, 2]):
		'''	Generating the 4 parameters (center_x, center_y, height, width) of each anchor.
			And concatenate them along the 0-th dimension.
			---------------------------------
			img: input image matrix (only two dimension)
			scales: the width of anchor (when it is a square)
			ratios: under certain scale, the ratio between width and height
			---------------------------------
			output: a tensor (N, 4) a series of anchor parameters
		'''
		c_x, c_y = torch.meshgrid([img.shape()[0], img.shape()[1]])
		

	def forward(self, x):
		pass

class FasterRCNN(nn.module):
	def __init__(self):
		super.__init__(self, FasterRCNN)

	def forward(self, x):
		pass