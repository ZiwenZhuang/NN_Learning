import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models

from modules import Conv2d, FC, ROIPool

class RPN(nn.module):
	''' Region Proposal Network as a component of the Faster-rcnn net
		Using .training field to define the forward() behavior
	'''
	def __init__(self):
		super.__init__(self, RPN)
		self.feature_net = models.vgg16() # The covnets that outputs the feature map
		# Considering the output of vgg is 512 channel, take the input of Conv layer as 512 channel
		self.conv0 = Conv2d(512, 512, 3, same_padding=True) 

	def rpn_score(self, x):
		''' It outputs score for each part of the feature map (from the result of self.conv0),
			for the input of making proposal
		'''
		pass

	def rpn_bbox(self, x):
		''' Proposing bounding boxes for selecting the proposals.
		'''
		pass

	def forward(self, x):
		assert isinstance(x, torch.Tensor) | isinstance(x, torch.Variable)
		# one output item: feature map
		features = self.feature_net(x)

		# start for another output item: ROI (region of interest)
		rpn_conv = self.conv0(features)

		rpn_prob = self.rpn_score(rpn_conv)
		rpn_bbox_pred = self.rpn_bbox(rpn_conv)

		# another output item: roi using the proposal layer
		rois = self.proposal_layer(...)

		# check if in the training mode to build loss
		if self.training:
			...

		return features, rois

	def build_loss(self):
		''' This method is model specific, and the interface is designed only for the internal
		method.
		'''
		pass

	def train(self):
		''' You should call this method to train the network.
		'''
		pass

class FasterRCNN(nn.module):
	def __init__(self):
		super.__init__(self, FasterRCNN)
		self.rpn = RPN() # The whole region proposal layer
		self.roi_pooling = ROIPool() # ROI pooling layer

	def forward(self, x):
		# input x has to be (N, C, H, W) image batch
		features, rois = self.rpn_layer(x)


		pass