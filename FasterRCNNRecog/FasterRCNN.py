import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms

from modules import Conv2d, FC, ROIPool, VGG16
import utils
from proposal_layer import proposal_layer as proposal_py
from proposal_layer import anchor_targets_layer as anchor_targets_py
from proposal_layer import proposal_targets as proposal_targets_py
from proposal_layer import generate_anchor
from bbox_transform import corner2centered

class RPN(nn.module):
	''' Region Proposal Network as a component of the Faster-rcnn net
	'''
	def __init__(self, configs = {
								"vgg_rate": 16, \
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
								"loss_lambda": 10, \
								}):
		''' Using configs field to store all the configurations as well as hyper-parameters
			"lambda": this is the hyper-parameter during calculating the loss
		'''
		super.__init__(self, RPN)
		self.configs = configs

		# The covnets that outputs the feature map
		self.feature_net = modules.VGG16()

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
		# Considering the proposal_py function is processed in feature map size (smaller in number),
		# it is needed to multiply the roi result.
		output = output * self.configs["vgg_rate"]
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
		# This rois are marked in image size (not the feature map size)
		rois = self.proposal_layer(rpn_prob, rpn_bbox_pred)

		# check if in the training mode to build loss
		if self.training:
			assert not gt_boxes is None
			rpn_labels, rpn_bbox_targets = self.anchor_targets_layer(rpn_prob, gt_boxes)
			self.loss = self.build_loss(rpn_prob, rpn_bbox_pred,\
									rpn_labels, rpn_bbox_targets)

		return features, rois

	def build_loss(self, prob, bbox_pred, labels, bbox_targets):
		''' This method is model specific, and the interface is designed only for the internal
		method.
			It stores the loss directly to the self.loss and also returns
		'''
		smooth_loss = nn.SmoothL1Loss()
		cls_loss = smooth_loss(prob[labels != -1], labels[labels != -1])

		box_loss = smooth_loss(bbox_pred[labels == 1], bbox_targets[labels == 1])
		box_loss = box_loss.sum() # based on page 3 at 'fast rcnn' paper

		self.loss = cls_loss + configs["lambda"] * box_loss
		return self.loss

class FasterRCNN(nn.module):
	def __init__(self, classes = None):
		super.__init__(self, FasterRCNN)
		if classes is not None:
			self.classes = classes
			self.num_classes = len(classes)

		# Sorry, due to the function structure, I have set the rate between the input
		# image and the output feature map statically. Normally this is not any kind
		# of changing parameters, so don't worry about that.
		self.img2feat_rate = 16

		self.rpn = RPN() # The whole region proposal layer
		self.roi_pooling = ROIPool() # ROI pooling layer
		self.fcs = nn.Sequential([
				FC(512*7*7, 4096),\
				nn.Dropout(),\
				FC(4096, 4096),\
				nn.Dropout(),\
			])
		self.score_fc = nn.Sequential([
				FC(4096, self.num_classes, relu= False),\
				nn.Softmax()
			])
		self.bbox_offset_fc = FC(4096, self.num_classes * 4, relu= False)

	def forward(self, x, gt_bbox=None, gt_labels=None):
		'''	Inputs:
			-------
			x: the input image (N, C, H, W) 4d Tensor, N = 1
			gt_bbox: necessary in training mode, (G, 4) 2d Tensor where 
		'''

		# input x has to be (N, C, H, W) image batch, usually N=1
		features, rois = self.rpn_layer(x, gt_bbox)
		# set the rois which are in the feature map scale.
		# or set the rois using gt_bounding boxes.
		if self.training:
			assert gt_bbox is not None and gt_labels is not None
			feat_rois, offsets_targets = self.proposal_targets(rois, gt_bbox, self.img2feat_rate)
		else:
			feat_rois = (rois / self.img2feat_rate).astype(int)
		# Now pooled is a (G, C, feature_size) 4-dimension tensor
		pooled = self.roi_pooling(features[0], feat_rois[0])

		# treat the all the proposals as a batch and feed to the rest of the network
		pooled_fc = self.fcs(pooled)
		pd_scores = self.score_fc(pooled_fc) # (G, D) Here, D means the number of classes
		pd_offsets = self.bbox_offset_fc(pooled_fc) # (G * 4) defined as [x1, y1, x2, y2, ...]

		# still the output from the network is bounding box deltas that need further
		# transformation to output bounding boxes coordinates.
		if self.training:
			self.loss = self.build_loss(pd_scores, \
										pd_offsets, \
										gt_labels, \
										offsets_targets)

		pd_bbox = self.interpret_offsets(rois, pd_offsets)
		return pd_scores, pd_bbox

	def proposal_targets(self, pd_rois, gt_bbox, img2feat_ratio):
		'''	for the simplicity, the implementation is put to another file.
		'''
		pd_rois = pd_rois.data.cpu().numpy()
		gt_bbox = gt_bbox.data.cpu().numpy()
		rois, offsets_targets = proposal_targets_py(pd_rois, gt_bbox, img2feat_ratio)
		rois = torch.from_numpy(rois)
		offsets_targets = torch.from_numpy(offsets_targets)
		return rois, offsets_targets

	def build_loss(self, pd_scores, pd_offsets, gt_labels, offsets_targets):
		'''	Considering the input for the rest of the network (except from the RPN part)
		has been changed in the training mode, the pd_scores and pd_bboxes are supposed
		to be aligned to the targets.
		'''
		# using cross entropy loss for the classification
		CEL_criterion = nn.CrossEntropyLoss()
		# using smooth L1 loss for the bounding box prediction
		# I will calculate the mean and sum manually
		SL1_criterion = nn.SmoothL1Loss(reduction = "none")

		cel = CEL_criterion(pd_scores, gt_labels)

		sl1 = SL1_criterion(pd_offsets, offsets_targets)
		sl1 = sl1.mean(dim = 0) # for each element along the N's items
		sl1 = sl1.sum() # The exact sum

		return cel + sl1

def train(data_path, store_path = "./FasterRCNNRecog/FasterRCNN_Learnt.pth"):
	'''	By training the network, it will print the traing epoch, and returns the learnt network
	which was set to testing mode (.training = False). It will save the entire network data to
	given file (override) as well.
	'''
	data_detections = dset.CocoDetection(root = data_path["train_img"],
										annFile = data_path["train_instances"],
										transform = transforms.ToTensor())
	
	pass