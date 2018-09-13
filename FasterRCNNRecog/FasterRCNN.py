import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models

from modules import Conv2d, FC, ROIPool
import utils
from proposal_layer import proposal_layer as proposal_py
from proposal_layer import anchor_targets_layer as anchor_targets_py

class RPN(nn.module):
	''' Region Proposal Network as a component of the Faster-rcnn net
	'''
	def __init__(self, configs = {
								"anchor_scales": [8, 16, 32], \
								"anchor_ratios": [0.5, 1, 2], \
								"lambda": 10, \
								"rpn_min_size": 16,\
								"rpn_max_ratio": 3, \
								"nms_thresh": 0.7, \
								"pre_nms_topN": 6000, \
								"post_nms_topN": 300, \
								"IoU_high_thresh": 0.7, \
								"IoU_low_thresh": 0.3, \
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

		_num_anchors = len(self.configs["anchor_scales"]) * len(self.configs["anchor_ratios"])
		self.score_conv = Conv2d(512, _num_anchors * 2, 1, relu= False, same_padding= False) # using softmax later, so no ReLU
		self.bbox_conv = Conv2d(512, _num_anchors * 4, 1, relu= False, same_padding= False)

		# Recording the loss
		self.loss = None

	def rpn_score(self, features):
		''' It outputs score for each part of the feature map (from the result of self.conv0),
			for the input of making proposal
		'''
		score = self.score_conv(features)
		# Considering Softmax is not a learnable layer, we don't need to "see" it later.
		return nn.Softmax(1)(score)
		# The output here is (N, A*2, H, W). Along the 1-th axis, the scores are (bg, fg, bg, fg, ...)

	def rpn_bbox(self, features):
		''' Proposing bounding boxes for selecting the proposals.
		'''
		return self.bbox_conv(features)

	def proposal_layer(self, rpn_prob, rpn_bbox_pred):
		''' For the simplicity, the detail implementation is moved to another file.
		'''
		rpn_prob = rpn_prob.data.cpu().numpy()
		rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
		output = proposal_py(np.transpose(rpn_prob, (0, 2, 3, 1)), \
			np.transpose(rpn_bbox_pred, (0, 2, 3, 1)), \
			self.configs)
		return torch.from_numpy(output)

	def anchor_targets_layer(self, rpn_prob, gt_boxes):
		''' For the simpliciry, the datail implementation is moved to the proposal_layer file.
		'''
		rpn_prob = rpn_prob.data.cpu().numpy()
		if isinstance(gt_boxes, nn.Tensor):
			gt_boxes = gt_boxes.data.cpu().numpy()
		labels, bbox_targets = anchor_targets_py(np.transpose(rpn_prob, (0, 2, 3, 1)), \
									gt_boxes, self.configs)
		return torch.from_numpy(labels), torch.from_numpy(bbox_targets)


	def forward(self, x, gt_boxes= None):
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
		rois = self.proposal_layer(rpn_prob, rpn_bbox_pred)

		# check if in the training mode to build loss
		if self.training:
			assert not gt_boxes is None
			self.anchor_targets_layer(rpn_prob, gt_boxes)
			

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