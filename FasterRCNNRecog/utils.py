﻿# This module contains several tools that helps changing the network module
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
		Input: two set of coordinates of the bounding boxes (x1, y1, x2, y2) for each box
		------------------
		Output: a number between 0 ~ 1
		------------------
		Requirements: You should ensure that x1 <= x2 and y1 <= y2
			or there will be problems which are even undetectable.
	'''
	# Find the intersected area
	inter_x1 = np.max(box0[0], box1[0])
	inter_y1 = np.max(box0[1], box1[1])
	inter_x2 = np.min(box0[2], box1[2])
	inter_y2 = np.min(box0[3], box1[3])

	# Calculate the intersection area
	inter_area = np.abs(inter_x2 - inter_x1 + 1) * np.abs(inter_y2 - inter_y1 + 1)

	# Calculate the area for each bounding boxes
	b0_area = (box0[2] - box0[0]) * (box0[3] - box0[1] + 1)
	b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1] + 1)

	# Calculate the IoU
	return (inter_area) / (b0_area + b1_area - inter_area)