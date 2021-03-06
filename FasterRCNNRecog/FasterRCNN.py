﻿import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms

from .modules import Conv2d, FC, ROIPool, VGG16
from . import utils
from .proposal_layer import proposal_layer as proposal_py
from .proposal_layer import anchor_targets_layer as anchor_targets_py
from .proposal_layer import proposal_targets as proposal_targets_py
from .proposal_layer import generate_anchor
from .bbox_transform import corner2centered

class RPN(nn.Module):
	''' Region Proposal Network as a component of the Faster-rcnn net
	'''
	def __init__(self, configs = {
								"vgg_rate": 16, \
								"anchor_scales": [4, 8, 16], \
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
		super(RPN, self).__init__()
		self.configs = configs

		# The covnets that outputs the feature map
		self.feature_net = VGG16()

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
		return output

	def anchor_targets_layer(self, rpn_prob, gt_boxes):
		''' For the simpliciry, the datail implementation is moved to the proposal_layer file.
		'''
		rpn_prob = rpn_prob.data.cpu().numpy()
		if isinstance(gt_boxes, torch.Tensor):
			gt_boxes = gt_boxes.data.cpu().numpy()
		labels, bbox_targets = anchor_targets_py(np.transpose(rpn_prob, (0, 2, 3, 1)), \
									gt_boxes, self.configs)
		return torch.from_numpy(labels), torch.from_numpy(bbox_targets)

	def forward(self, x, gt_boxes= None):
		assert isinstance(x, torch.Tensor)
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

		# The returned rois is a numpy array
		# features is a Tensor
		return features, rois

	def build_loss(self, prob, bbox_pred, labels, bbox_targets):
		''' This method is model specific, and the interface is designed only for the internal
		method.
			It stores the loss directly to the self.loss and also returns
		'''
		cross_entropy = nn.CrossEntropyLoss()
		prob = prob.permute([0, 2, 3, 1]).contiguous().view(-1, 2)
		labels_l = labels.long()
		cls_loss = cross_entropy(prob[labels != -1], labels_l[labels != -1])

		smooth_loss = nn.SmoothL1Loss()
		bbox_pred = bbox_pred.permute([0, 2, 3, 1]).contiguous().view(-1, 4)
		bbox_targets_f = bbox_targets.float()
		box_loss = smooth_loss(bbox_pred[labels == 1], bbox_targets_f[labels == 1])
		box_loss = box_loss.sum() # based on page 3 at 'fast rcnn' paper

		self.loss = cls_loss + self.configs["lambda"] * box_loss
		return self.loss

class FasterRCNN(nn.Module):
	def __init__(self, num_classes = None):
		super(FasterRCNN, self).__init__()
		if num_classes is not None:
			self.num_classes = num_classes

		# Sorry, due to the function structure, I have set the rate between the input
		# image and the output feature map statically. Normally this is not any kind
		# of changing parameters, so don't worry about that.
		self.img2feat_rate = 16

		self.rpn = RPN() # The whole region proposal layer
		self.roi_pooling = ROIPool() # ROI pooling layer
		self.fcs = nn.Sequential(
				FC(512*7*7, 4096),\
				nn.Dropout(),\
				FC(4096, 4096),\
				nn.Dropout() )
		self.score_fc = nn.Sequential(
				FC(4096, self.num_classes, relu= False),\
				nn.Softmax() )
		self.bbox_offset_fc = FC(4096, 4, relu= False)

	def forward(self, x, gt_bbox=None, gt_labels=None):
		'''	Inputs:
			-------
			x: the input image (N, C, H, W) 4d Tensor, N = 1
			gt_bbox: necessary in training mode, (G, 4) 2d Tensor [x1, y1, x2, y2]
			gt_labels: necessary in training mode, (G,) 1d is ok, all are indeces
		'''

		# input x has to be (N, C, H, W) image batch, usually N=1
		features, rois = self.rpn(x, gt_bbox)
		# set the rois which are in the feature map scale.
		# or set the rois using gt_bounding boxes.
		if self.training:
			assert gt_bbox is not None and gt_labels is not None
			feat_rois, offsets_targets = self.proposal_targets(rois, gt_bbox, \
											self.img2feat_rate, features.shape)
		else:
			feat_rois = (torch.from_numpy(rois) / self.img2feat_rate).astype(int)
		# Now pooled is a (G, C, feature_size) 4-dimension tensor
		pooled = self.roi_pooling(features[0], feat_rois[0])

		# treat the all the proposals as a batch and feed to the rest of the network
		pooled_fc = self.fcs(pooled.reshape((-1, 512 * 7 * 7)))
		pd_scores = self.score_fc(pooled_fc) # (G, D) Here, D means the number of classes
		pd_offsets = self.bbox_offset_fc(pooled_fc) # (G * 4) defined as [x1, y1, x2, y2, ...]

		# still the output from the network is bounding box deltas that need further
		# transformation to output bounding boxes coordinates.
		if self.training:
			self.loss = self.build_loss(pd_scores, \
										pd_offsets, \
										gt_labels, \
										offsets_targets)

		pd_bbox = self.interpret_offsets(feat_rois, pd_offsets)
		# return two tensors
		return pd_scores, pd_bbox * self.img2feat_rate

	def proposal_targets(self, pd_rois, gt_bbox, img2feat_ratio, features_shape):
		'''	for the simplicity, the implementation is put to another file.
		'''
		pd_rois = pd_rois # This is already a numpy array
		if isinstance(gt_bbox, torch.Tensor):
			gt_bbox = gt_bbox.data.cpu().numpy()
		rois, offsets_targets = proposal_targets_py(pd_rois, gt_bbox, \
										img2feat_ratio, features_shape)
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

		sl1 = SL1_criterion(pd_offsets, offsets_targets.float())
		sl1 = sl1.mean(dim = 0) # for each element along the N's items
		sl1 = sl1.sum() # The exact sum

		return cel + sl1

	def interpret_offsets(self, rois, pd_offsets):
		'''	Considering the offsets are all in ratio, it doesn't matter whether you provide
		the rois in feature map scale or in image scale.
			This function returns the bounding coordinates in image scale [x1, x2, y1, y2]
		'''
		if isinstance(rois, np.ndarray): rois = torch.from_numpy(rois)
		pd_offsets = pd_offsets.float()
		rois = rois[0].float()
		H = rois[:, 2] - rois[:, 0]
		W = rois[:, 3] - rois[:, 1]

		pd_bbox = torch.cat(( \
				(torch.mul(pd_offsets[:, 0], H) + rois[:, 0]).unsqueeze(1), \
				(torch.mul(pd_offsets[:, 1], W) + rois[:, 1]).unsqueeze(1), \
				(torch.mul(pd_offsets[:, 2], H) + rois[:, 2]).unsqueeze(1), \
				(torch.mul(pd_offsets[:, 3], W) + rois[:, 3]).unsqueeze(1), \
			), 1)

		return pd_bbox

	def savetofile(self, filepath):
		''' Save the parameters of this model to the file, by using state_dict().
			So be sure to use the loadfromfile method to recover the parameters.
		'''
		torch.save(self.state_dict(), filepath)

def train(data_path, store_path = "./FasterRCNNRecog/FasterRCNN_Learnt.pth"):
	'''	By training the network, it will print the traing epoch, and returns the learnt network
	which was set to testing mode (.training = False). It will save the entire network data to
	given file (override) as well.
	'''

	# 1. configuring the data
	#	read files and setup the targets

	data_detections = dset.CocoDetection(root = data_path["train_img"],
										annFile = data_path["train_instances"],
										transform = transforms.ToTensor())
	# set hyper-parameters
	config = {	"learning_rate": 0.001,
				"momentum": 0.9,
				"weight_decay": 0.0005}

	# initialize the network
	net = FasterRCNN(91)
	net.train() # set the network to training mode
	# Considering I still did not understand why longcw used a pretrained part inside VGG16,
	# I set the entire network trainable.
	optimizer = torch.optim.SGD(net.parameters(), lr= config["learning_rate"], momentum= config["momentum"], weight_decay= config["weight_decay"])

	# entering epoch and then prepare image and data for the network
	iterations = 0
	epoch = 0
	epoch_size = 10
	total_loss = 0
	for img, targets in data_detections:
		if len(targets) == 0: continue

		# get one batch of input (only one image)
		img_batch = img.unsqueeze(0)

		# set up the labels (N items in all)
		gt_labels = [i["category_id"] for i in targets] # a list of int
		gt_labels = torch.LongTensor(gt_labels)
		gt_bbox = [i["bbox"] for i in targets] # a list of 4-element floats
		gt_bbox = np.array(gt_bbox)
		x1s, y1s, x2s, y2s = utils.coco2corner(gt_bbox) # change the corner format
		gt_bbox = np.concatenate([np.expand_dims(x1s, 1), np.expand_dims(y1s, 1),\
								np.expand_dims(x2s, 1), np.expand_dims(y2s, 1)], axis = 1)

		# forward (train the two parts together)
		pd_scores, pd_bbox = net(img_batch, gt_bbox, gt_labels)
		total_loss += net.loss + net.rpn.loss

		# Check if the iterations can be seen as an epoch.
		if iterations % epoch_size == 0:
			# the mean of total_loss
			total_loss = total_loss / epoch_size
			# backward
			optimizer.zero_grad()
			total_loss.backward()
			utils.clip_gradient(net, 10.)
			optimizer.step()
			# display details to show progress
			epoch += 1
			print("In epoch loss: " + str(total_loss))
			total_loss = 0

		iterations += 1

	# store the network into file
	print("Saving learnt model into file: " + store_path)
	net.savetofile(store_path)
	print("Done saving!")

	return net
