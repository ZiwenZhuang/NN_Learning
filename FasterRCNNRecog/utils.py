# This module contains several tools that helps changing the network module
import numpy as mp
import torch
import torch.nn as nn

def set_trainable(model, require_grad = True):
	for param in model.parameters():
		param.requires_grad = require_grad

def generate_anchor(img, stride = 16, scales = [128, 256, 512], ratios = [0.5, 1, 2]):
	'''	Generating the 4 parameters (x1, y1, x2, y2) of each anchor.
		And concatenate them along the 0-th dimension.
		---------------------------------
		img: input image matrix (N, C, H, W)- 4 dimensions                 x1, y1--------+
		stride: the stride that the sliding window would move                 |          |
		scales: the width of anchor (when it is a square)                     |          |
		ratios: under certain scale, the ratio between width and height       +-------x2, y2
		---------------------------------
		output: a tensor (H*W/stride/stride, 4) a series of anchor parameters, for the memory efficiency
	'''
	_num_anchors = (img.shape()[-2] // stride) * (img.shape()[-1] // stride)
	x_ctr, y_ctr = torch.meshgrid([img.shape()[2] // stride, img.shape()[3] // stride])
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
				torch.round(torch.sqrt(torch.mul(base, scale * scale * ratio)))
				)
			w_seq.append(
				torch.round(torch.sqrt(torch.mul(base, scale * scale / ratio)))
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