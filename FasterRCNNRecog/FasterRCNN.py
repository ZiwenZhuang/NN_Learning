import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models

from modules import Conv2d, FC

class RPN(nn.module):
	''' Region Proposal Network as a component of the Faster-rcnn net
	'''
	def __init__(self, RPN):
		super.__init__(self, RPN)
		self.feature_net = models.vgg16() # The covnets that outputs the feature map
		# Considering the output of vgg is 512 channel, take the input of Conv layer as 512 channel
		self.conv0 = Conv2d(512, 512, 3, same_padding=True) 

	def rpn_score(self, x):
		''' The method takes the output or feature map (by self.conv0)
		and output the score for the roi pooling layer.
		'''


	def forward(self, x):
		assert isinstance(x, torch.Tensor) | isinstance(x, torch.Variable)
		pass

class FasterRCNN(nn.module):
	def __init__(self):
		super.__init__(self, FasterRCNN)
		self.rpn_layer = RPN() # The whole region proposal layer

	def forward(self, x):
		# input x has to be (N, C, H, W) image batch
		features, rois = self.rpn_layer(x)
		pass