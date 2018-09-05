# This module contains several tools that helps changing the network module
import numpy as np

import torch
import torch.nn as nn

DEBUG = True

def set_trainable(model, require_grad = True):
	for param in model.parameters():
		param.requires_grad = require_grad

def generate_anchor_TensorVersion(img_info, stride = 1, scales = [8, 16, 32], ratios = [0.5, 1, 2]):
	'''	Generating the 4 parameters (x1, y1, x2, y2) of each anchor.
		And concatenate them along the 0-th dimension.
		---------------------------------
		img: input image matrix (N, C, H, W)- 4 dimensions                 x1, y1--------+
		stride: the stride that the sliding window would move                 |          |
		scales: the width of anchor (when it is a square)                     |          |
		ratios: under certain scale, the ratio between width and height       +-------x2, y2
		---------------------------------
		output: a tensor (H*W/stride/stride, 4) a series of anchor parameters, for the memory
			efficiency
		---------------------------------
		This is just for Documnetation, not tested yet. DO NOT use it if you are not the author.
	'''
	_num_anchors = (img_info[-2] // stride) * (img_info[-1] // stride)
	x_ctr, y_ctr = torch.meshgrid([img_info[2] // stride, img_info[3] // stride])
	# keep the array into one dimension, (2 dimension in all)
	x_ctr = x_ctr.reshape((-1, 1)).expand(_num_anchors * len(scales) * len(ratios), 1)
	y_ctr = y_ctr.reshape((-1, 1)).expand(_num_anchors * len(scales) * len(ratios), 1)
	# let the number be aligned to the pixel coordinates
	x_ctr = torch.mul(x_ctr, stride) + (stride // 2)
	y_ctr = torch.mul(y_ctr, stride) + (stride // 2)
		
	# prepare to generate different shape
	base = torch.ones((_num_anchors, 1))
	h_seq = []
	w_seq = []

	# concatenate each anchor parameters (height and width) together
	for scale in scales:
		for ratio in ratios:
			h_seq.append(
				torch.round(torch.sqrt(torch.mul( base, int(float(scale * scale) * float(ratio)) )))
				)
			w_seq.append(
				torch.round(torch.sqrt(torch.mul( base, int(float(scale * scale) * float(ratio)) )))
				)
	h_half = torch.round(torch.mul(torch.cat(h_seq, 0).reshape((-1, 1)), 0.5))
	w_half = torch.round(torch.mul(torch.cat(w_seq, 0).reshape((-1, 1)), 0.5))

	# calculate the 2 pair of coordinates
	anchors = [
		x_ctr - h_half,
		y_ctr - w_half,
		x_ctr + h_half,
		y_ctr + w_half
		]
	anchors = torch.cat(anchors, 1).contiguous()

	return anchors

def generate_anchor(img_info, stride = 1, scales = [8, 16, 32], ratios = [0.5, 1, 2]):
	'''	Generating the 4 parameters (x1, y1, x2, y2) of each anchor.
		And concatenate them along the 0-th dimension.
		---------------------------------
		img_info: the [height, width] of the image                         x1, y1--------+
		stride: the stride that the sliding window would move                 |          |
		scales: the width of anchor (when it is a square)                     |          |
		ratios: under certain scale, the ratio between width and height       +-------x2, y2
		---------------------------------
		output: a numpy matrix (_num_anchors, H, W, 4) a series of anchor parameters.
	'''
	_num_anchors = (img_info[0] // stride) * (img_info[1] // stride)
	x_ctr = np.arange(img_info[0] // stride)
	y_ctr = np.arange(img_info[1] // stride)
	x_ctr, y_ctr = np.meshgrid(x_ctr, y_ctr)
	# let the number be aligned to the pixel coordinates
	x_ctr = (x_ctr * stride) + (stride // 2)
	y_ctr = (y_ctr * stride) + (stride // 2)
	
	# prepare to generate different shape
	base = np.ones([img_info[0] // stride, img_info[1] // stride])
	base = scale * scale
	
	# preparing anchors
	anchors = []
	# generate anchors (still in matrix) and put into a list
	for scale in scales:
		for ratio in ratios:
			x_half = base * ratio / 2
			y_half = base / ratio / 2
			anchor = np.concatenate([
				np.expand_dims(x_ctr - x_half, axis=0),
				np.expand_dims(y_ctr - y_half, axis=0),
				np.expand_dims(x_ctr + x_half, axis=0),
				np.expand_dims(y_ctr + y_half, axis=0)
				], axis= 0)
			anchors.append(np.expand_dims(anchor, axis= 0))
	# Till Now the dimensions should still be (_num_anchors, 4, H, W)
	anchors = np.concatenate(anchors, axis= 0)

	anchors = np.transpose(anchors, (0, 3, 1, 2)) # (_num_anchors, 4, H, W)
	return anchors
	

def proposal_layer(rpn_score, rpn_bbox, img_info, anchor_scales, anchor_ratios, nms_thresh= 0.7):
	'''	Extract useful proposals that are both high in score and big in area.
		---------------------
		Parameters
		----------
		rpn_score: is torch.Tensor (N, C, H, W) 4-dimension
		rpn_bbox: the network predicted slight shift for the "ground true" bounding box relative to
			the anchor at the specific point. (N, C, H, W) 4-dimension
		img_info: the [height, width] of the input image (I means the input image of the whole network)
		anchor_scales: the anchor size if put onto the input images
		anchor_ratios: the ratio between the height and the width of the anchor
		nms_thresh: Non-Maximum Suppression threshold
		----------------------
		Returns
		----------
		rpn_rois: a batch of bunch of rois that are elected from the network proposals.
			(N, H * W * A, 5) where H and W are the size of input feature map (rpn_score, rpn_bbox)
			and A is the number of anchors for each point in the feature map
	'''
	#	Considering the output proposal for each image is different, you should provide batch with
	# only 1 item, or it is difficult for this method to return the same amount of proposals for
	# the whole batch.
	
	assert rpn_score.size()[0] == 1 & rpn_rpn_bbox.size()[0] == 1, \
		"Only single item batches are supported"
	
	anchors = generate_anchor(list(rpn_bbox.size())[2:], scales= anchor_scales, ratios= anchor_ratios)
	rpn_score_size = list(rpn_score.size())
	rpn_score = rpn_score.permute(0, 2, 3, 1)
	rpn_bbox = rpn_bbox.permute(0, 2, 3, 1)

