import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models

from modules import Conv2d, FC, ROIPool
import utils

class RPN(nn.module):
	''' Region Proposal Network as a component of the Faster-rcnn net
	'''
	def __init__(self, configs = {
								"anchor_scales": [8, 16, 32], \
								"anchor_ratios": [0.5, 1, 2], \
								"lambda": 10, \
								}):
		''' Using configs field to store all the configurations as well as hyper-parameters
			"lambda": this is the hyper-parameter during calculating the loss
		'''
		super.__init__(self, RPN)
		self.configs = configs

		# The covnets that outputs the feature map
		self.feature_net = models.vgg16()

		# Considering the output of vgg is 512 channel, take the input of Conv layer as 512 channel
		self.conv0 = Conv2d(512, 512, 3, same_padding=True)

		num_anchors = len(self.configs["anchor_scales"]) * len(self.configs["anchor_ratios"])
		self.score_conv = Conv2d(512, num_anchors * 2, 1, relu= False, same_padding= False) # using softmax later, so no ReLU
		self.bbox_conv = Conv2d(512, num_anchors * 4, 1, relu= False, same_padding= False)

		# Recording the loss
		self.loss = None

	def rpn_score(self, features):
		''' It outputs score for each part of the feature map (from the result of self.conv0),
			for the input of making proposal
		'''
		score = self.score_conv(features)
		# Considering Softmax is not a learnable layer, we don't need to "see" it later.
		return nn.Softmax(1)(score)

	def rpn_bbox(self, features):
		''' Proposing bounding boxes for selecting the proposals.
		'''
		return self.bbox_conv(features)

	def proposal_layer(self, rpn_conv, rpn_bbox):
		''' For the simplicity, the implementation is moved to another file.
		'''
		return utils.proposal_layer(rpn_conv, rpn_bbox)

	def forward(self, x):
		assert isinstance(x, torch.Tensor) | isinstance(x, torch.Variable)
		# one output item: feature map
		features = self.feature_net(x)

		# start for another output item: ROI (region of interest)
		rpn_conv = self.conv0(features)

		rpn_prob = self.rpn_score(rpn_conv)
		rpn_bbox_pred = self.rpn_bbox(rpn_conv)
		#	Until now the tensors are always (N, C, H, W), where C is always the number of output
		# for each pixel in the feature map.

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