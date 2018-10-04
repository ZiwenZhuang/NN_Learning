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

def box_IoU(box0, box1):
	'''	Given two bounding boxes, the function returns the IoU of these two.
		__________________
		Input: two batches of coordinates of the bounding boxes (x1, y1, x2, y2) for each box
			where the size are either (N, 4) or (4,) or there will be trouble
		------------------
		Output: a list of numbers between 0 ~ 1 even there are only one set of coordinates in
			both box0 and box1
		------------------
		Requirements: You should ensure that x1 <= x2 and y1 <= y2
			or there will be problems which are even undetectable.
	'''
	# Prepare to broadcasting any of the array, if needed
	if box0.shape == (4,):
		box0 = box0.reshape((-1, 4))
	if box1.shape == (4,):
		box1 = box1.reshape((-1, 4))
	np.broadcast_arrays(box0, box1)

	# Find the intersected area
	inter_x1 = np.maximum(box0[:, 0], box1[:, 0])
	inter_y1 = np.maximum(box0[:, 1], box1[:, 1])
	inter_x2 = np.minimum(box0[:, 2], box1[:, 2])
	inter_y2 = np.minimum(box0[:, 3], box1[:, 3])

	# Calculate the intersection area
	inter_area = np.abs(inter_x2 - inter_x1 + 1) * np.abs(inter_y2 - inter_y1 + 1)

	# Calculate the area for each bounding boxes
	b0_area = (box0[:, 2] - box0[:, 0]) * (box0[:, 3] - box0[:, 1] + 1)
	b1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1] + 1)

	# Calculate the IoU
	return (inter_area) / (b0_area + b1_area - inter_area)

def cal_overlaps(anchors, gt_bboxes):
	'''	Calcualte IoU between each anchor and each ground truth bounding box.
		-----------------------
		Parameters:
		-----------
		anchors: a (N, 4) numpy array, which denotes the coordinates of the anchors,
			in the form of ([x1, y1, x2, y2], [x1, y1, x2, y2], ...)
		gt_bboxes: a (G, 4) numpy array, which denots the coordinates of the ground
			truth bounding boxes, in the form of ([x1, y1, x2, y2], [x1, y1, x2, y2], ...)
		-----------------------
		Returns:
		--------
		overlaps: a (N, G) numpy array, which denotes the IoU between each anchor 
			and each ground truth bounding box. And the order is still the same as
			the input sequence.
	'''
	output = []

	# for each ground truth bounding box, calculate all the IoU with all the N anchors
	for i in range(gt_bboxes.shape[0]):
		a_column = box_IoU(anchors, gt_bboxes[i])
		output.append(np.expand_dims(a_column, 1))

	# put all the IoUs into a single matrix
	return np.ascontiguousarray(np.concatenate(output, axis=1))

def coco2corner(coco_bbox):
	'''	This function returns the corners coordinates of a bounding box,
	given bbox field from COCOapi is not aligned like that.
		By the way, you can input a 2darray (N, 4), and it will still
	returns 4 lists of number with length N.
	'''
	if len(coco_bbox.shape) == 2:
		coco_bbox.transpose()
	x1 = coco_bbox[0]
	y1 = coco_bbox[1]
	x2 = coco_bbox[0] + coco_bbox[2]
	y2 = coco_bbox[1] + coco_bbox[3]
	return x1, y1, x2, y2

def clip_gradient(model, clip_norm):
	'''	Preventing the gradient goes too big that the parameters shoot to nowhere.
	This function is learnt from longcw/faster-rcnn
	'''
	totalnorm = 0
	for p in model.parameters():
		if p.requires_grad:
			modulenorm = p.grad.data.norm()
			totalnorm += modulenorm ** 2
	totalnorm = np.sqrt(totalnorm)

	norm = clip_norm / max(totalnorm, clip_norm)
	for p in model.parameters():
		if p.requires_grad:
			p.grad.mul_(norm)