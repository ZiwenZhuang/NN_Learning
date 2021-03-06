﻿# The file is used as part of the region proposal network, which makes the whole network two
# parts ;P
# And the code is based on https://github.com/longcw/faster_rcnn_pytorch/blob/master/faster_rcnn/rpn_msr/proposal_layer.py
# If you want to see the original code, please clone the whole repo for a closer view.

import numpy as np
from .utils import box_IoU, cal_overlaps
from .bbox_transform import bbox_transform_inv, bbox_transform

def generate_anchor(img_info, stride = 1, scales = [8, 16, 32], ratios = [0.5, 1, 2]):
	'''	Generating the 4 parameters (x1, y1, x2, y2) of each anchor.
	And concatenate them along the 0-th dimension.
		---------------------------------
		Parameters:
		--------------------------
		img_info: the [height, width] of the image                         x1, y1--------+>
		stride: the stride that the sliding window would move                 |          |
		scales: the width of anchor (when it is a square)                     |          |
		ratios: under certain scale, the ratio between width and height       +-------x2, y2
		---------------------------------                                     v
		output: a numpy matrix (H * W * _num_anchors, 4) a series of anchor parameters.
			along the last axis is like ([x1, y1, x2, y2],
										 [x1, y1, x2, y2], ...)
	'''
	_num_anchors = (img_info[0] // stride) * (img_info[1] // stride)
	x_ctr = np.arange(img_info[0] // stride)
	y_ctr = np.arange(img_info[1] // stride)
	x_ctr, y_ctr = np.meshgrid(y_ctr, x_ctr)
	# let the number be aligned to the pixel coordinates
	x_ctr = (x_ctr * stride) + (stride // 2)
	y_ctr = (y_ctr * stride) + (stride // 2)
	
	# prepare to generate different shape
	base = np.ones([img_info[0] // stride, img_info[1] // stride])
	
	# preparing anchors
	anchors = []
	# generate anchors (still in matrix) and put into a list
	for scale in scales:
		base = base * scale * scale
		for ratio in ratios:
			x_half = base * ratio / 2
			y_half = base / ratio / 2
			anchor = np.concatenate([
				np.expand_dims(x_ctr - x_half, axis=0),
				np.expand_dims(y_ctr - y_half, axis=0),
				np.expand_dims(x_ctr + x_half, axis=0),
				np.expand_dims(y_ctr + y_half, axis=0)
				], axis=0)
			anchors.append(np.expand_dims(anchor, axis= 0))
	# Till Now the dimensions should still be (_num_anchors, 4, H, W)
	anchors = np.concatenate(anchors, axis= 0)

	# (1, _num_anchors*4, H, W)
	anchors = anchors.reshape((1, -1, img_info[0] // stride, img_info[1] // stride))

	# (1, H, W, _num_anchors*4)
	anchors = np.transpose(anchors, axes=(0, 2, 3, 1))
	
	# (H * W * _num_anchors, 4)
	anchors = anchors.reshape((-1, 4))
	return anchors

def apply_delta(anchors, rpn_bbox):
	'''	Considering the definition of bounding boxes, the application of predicted bbox delta
		will be a little troublesome.
		But the overall strategy is still the same (rpn_bbox represents the ratio from anchor 
	parameters to the ground true bounding boxes.)
	'''
	# Still the predictions from the neural network are ratio to the anchors
	# But the predicted bounding box are notated in 2 pairs of coordinates at the corners.
	pd_bbox = bbox_transform_inv(anchors, rpn_bbox)
	return pd_bbox

def clip_boxes(boxes, im_shape):
	'''	Clip boxes to image boundaries
		boxes: (N, A*4)	
		imshape: [height, width]
	'''
	if boxes.shape[0] == 0:
		return boxes

	# series of x1
	boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[0] - 1), 0)
	# series of y1
	boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[1] - 1), 0)
	# series of x2
	boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[0]), 1)
	# series of y2
	boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[1]), 1)

	return boxes

def _filter_boxes(boxes, min_size, max_ratio):
	''' Generate indexes (along 0-th dimension) that satisfies the size as ratio scales.
		max_ratio: only one value (greater than 1), which will be used as reciprocal.
	'''
	dx = boxes[:, 2] - boxes[:, 0] + 1
	dy = boxes[:, 3] - boxes[:, 1] + 1
	rat0 = dy / dx
	rat1 = dx / dy
	
	keep = np.where((dx >= min_size) & (dy >= min_size) & (rat0 <= max_ratio) & (rat1 <= max_ratio))[0]
	return keep

def nms(predictions, thresh):
	'''	Given the 2-d numpy array (N, 5) as the prediction parameters, returns the indexes
		that are selected. Then you can use these indexes to filt the bounding boxes and the
		scores.
		---------------------
		Parameters
		----------
		predictions: numpy 2darray (N, 5), along the 1-th axies (x1, y1, x2, y2, score)
		---------------------
		Returns
		----------
		numpy 1darray served as indexes of proposals (along the 0-th axis)
	'''
	# Considering the input predictions are all sorted by score, the algorithm will be elminate
	# all the indexes that has the IoU greater than threshold (from start)
	
	left = np.arange(predictions.shape[0])
	keep = []
	while len(left) > 0:
		# Considering the predictions are sorted, we add the first element into the output
		add = left[0]
		keep.append(add)

		# remove all bounding boxes' index whose IoU with the added bounding box is greater
		# than the threshold.
		IoUs = box_IoU(predictions[left][:4], predictions[add][:4])
		to_remove = np.where(IoUs > thresh)

		# remove those bounding boxes by index in the 'left'
		np.delete(left, [0])
		np.delete(left, to_remove)
	
	return keep

def proposal_layer(rpn_score, rpn_bbox, configs):
	'''	Extract useful proposals that are both high in score and big in area. (processed in terms of
	feature map size.)
		---------------------
		Parameters
		----------
		rpn_score: is np array (1, H, W, A*2) 4-dimensions
		rpn_bbox: the network predicted slight shift for the "ground true" bounding box relative to
			the anchor at the specific point. (1, H, W, A*4) 4-dimensions
			where A is the number of anchors for each point (in the feature map)
		anchor_scales: the anchor size if put onto the input images
		anchor_ratios: the ratio between the height and the width of the anchor
		configs: requires (rpn_min_size, rpn_max_ratio, nms_thresh, pre_nms_topN, post_nms_topN)
			keys as configuration
		----------------------
		Returns
		----------
		rpn_rois: a batch of bunch of rois that are elected from the network proposals.
			(1, H * W * A, 4) where H and W are the size of input feature map (rpn_bbox, rpn_score)
			and A is the number of anchors for each point in the feature map
			And the parameters along the 1-th axis is (x1, y1, x2, y2)
	'''
	#	Considering the output proposal for each image is different, you should provide batch with
	# only 1 item, or it is difficult for this method to return the same amount of proposals for
	# the whole batch.
	assert rpn_score.shape[0] == 1 & rpn_bbox.shape[0] == 1, \
		"Only single item batches are supported"

	#	Considering the output from feature map (The first step in RPN network) has the same size
	# as the feature map, we regard the rpn_score/rpn_bbox just has the same size as bounding box
	# size.
	#	Be sure to check the output of self.score_conv and self.bbox_conv for the output size!!!!
	feature_shape = rpn_score.shape[1:3]
	
	# 0. get anchors in (H * W * _num_anchors, 4) where only the last dimension is the anchor parameter
	anchors = generate_anchor(feature_shape, \
		scales= configs["anchor_scales"], ratios= configs["anchor_ratios"])

	# 1. reshape the predicted bboxes and scores to (H * W * _num_anchors, 4)
	# and apply delta (from bbox prediction) to each of the anchor
	pd_bbox = rpn_bbox.reshape((-1, 4))
	    # the first set of channels are back_ground probs
		# the second set are the fore_ground probs, which we want
	pd_scores = rpn_score.reshape((-1, 2))[:, 1]
	pd_bbox = apply_delta(anchors, pd_bbox)

	# 2. clip prediction bounding boxes to image size (using the size of the feature map here)
	pd_bbox = clip_boxes(pd_bbox, feature_shape)

	# 3. get rid of anchors with too small size or too strange height/width ratio
	keep = _filter_boxes(pd_bbox, configs["rpn_min_size"], configs["rpn_max_ratio"])
	pd_bbox = pd_bbox[keep]
	pd_scores = pd_scores[keep]

	# 4. sort all (proposal, score) pair by score from HIGHest to LOWest
	# 5. take the pre_nms_topN number of proposals
	order = pd_scores.ravel().argsort()[::-1]
	if (configs["pre_nms_topN"] > 0):
		order = order[:configs["pre_nms_topN"]]
	pd_bbox = pd_bbox[order]
	pd_scores = pd_scores[order]

	# 6. perform NMS
	#rois = np.hstack((pd_bbox, np.expand_dims(pd_scores, 1)))
	rois = pd_bbox # there should be no need for scores
	keep = nms(rois, configs["nms_thresh"])
	rois = rois[keep]

	return np.expand_dims(rois, 0)

def anchor_targets_layer(rpn_cls_score, gt_bbox, configs):
	'''	This method generates targets for the entire region proposal network,
	which is not a differentiable neural network layer, just a part of the network.
		From the annotated labels (ground truth bounding boxes), assign anchors
	ground-targets, and produce anchor classifications (foreground/background)
	as well as bounding boxes regression targets.
		---------------------
		Inputs:
		--------
		rpn_cls_score: the output from rpn_score layer, which contains the feature
			map size. It has to be transposed to (1, H, W, A*2) 4-dimension numpy array.
		gt_bbox: (N, 4) 2-dimension numpy array. The annotated ground-truth bounding
			boxes. It only needs to be in image scale, this function will take care of
			that.
		configs: configurations from the RPN network module, which must contains
			the following fields: "anchor_scales", "anchor_ratios"
		--------------------
		Returns:
		--------
		rpn_labels: (H*W*A,) 1-dimension numpy array, where the order cooresponding
			to the anchors are based on the 'generate_anchor' function, which doesn't
			requires further concern. Along the 1-th axis, 1 denotes foreground, 0 denotes
			background, -1 denotes dont care.
		rpn_bbox_targets: (H*W*A, 4) 2-dimension numpy array, The target predictions for
			each bounding box, which is the result from x to t transform (according to
			the paper)
	'''
	assert rpn_cls_score.shape[0] == 1, \
		"Only support batch with size 1"

	feature_shape = rpn_cls_score.shape[1:3]
	anchors = generate_anchor(feature_shape, scales= configs["anchor_scales"], ratios= configs["anchor_ratios"])

	# calculate overlaps between each anchor and each gt_bbox
	# BTW: change the gt_bbox into feature map scale to make the comparison
	gt_bbox = gt_bbox / configs["vgg_rate"]
	overlaps = cal_overlaps(anchors, gt_bbox)

	# find out the greatest IoU between each anchor and the gt_bbox
	max_IoU_ind = overlaps.argmax(axis = 1)
	max_IoU = overlaps.max(axis = 1)

	# one of the output (rpn_labels)
	rpn_labels = np.ones((anchors.shape[0],)) * -1
	rpn_labels[max_IoU > configs["IoU_high_thresh"]] = 1
	rpn_labels[max_IoU < configs["IoU_low_thresh"]] = 0

	# another output (rpn_bbox_targets)
	# (1) assign the gt_bbox with the greatest IoU for each anchors
	target_bboxes = [np.expand_dims(gt_bbox[i], 0) for i in max_IoU_ind]
	target_bboxes = np.concatenate(target_bboxes, axis= 0)
	# (2) get the target predictions
	rpn_bbox_targets = bbox_transform(anchors, target_bboxes)

	return rpn_labels, rpn_bbox_targets

def jitter_box(boxes):
	'''	Parameters:
		-----------
		boxes: (N, 4) [x1, y1, x2, y2] numpy 2d array, which denotes the corner coordinates in the 
			readme description.
		-----------
		Output:
		-------
		jittered: (N, 4) [x1, y1, x2, y2] numpy 2d array.
	'''
	# The target offsets has to be calculated additionally. You can figure out why it is not deltas
	# The initial random value is in [0, 1), which need to be transformed to [-0.5, 0.5]
	deltas = np.random.rand(*boxes.shape) - (np.ones([*boxes.shape]) * 0.5)

	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	H = x2 - x1
	W = y2 - y1

	jittered = np.concatenate([
			np.expand_dims((x1 - np.multiply(deltas[:, 0], H)), axis = 1), \
			np.expand_dims((y1 - np.multiply(deltas[:, 1], W)), axis = 1), \
			np.expand_dims((x2 - np.multiply(deltas[:, 2], H)), axis = 1), \
			np.expand_dims((y2 - np.multiply(deltas[:, 3], W)), axis = 1), \
		], axis = 1)

	return jittered

def proposal_targets(pd_rois, gt_bbox, img2feat_ratio, features_shape):
	'''	Assign ground truth bounding boxes to the roi proposals in terms of the feature map.
	And this is a simple version, do not kick me.... please.
		----------------
		Inputs:
		-------
		pd_rois: (1, G, 4) numpy 3d array, produced directly from the RPN network. It is not needed
			for simplicity.
		gt_bbox: (N, 4) numpy 2d array, along whose 1-th axis is [x1, y1, x2, y2] and be sure to
			check the coordinate system in terms of the image.
		img2feat_ratio: the rate between the input image to the entire network and the feature
			map output from the RPN network.
		features_shape: (N, H, W) index-able value, which helps to clip the generated boxes.
		----------------
		Outputs:
		-------
		rois: (1, N, 4) numpy 3d array the target roi in the scale of feature map.
		offsets_targets: (1, N, 4) [dx1, dy1, dx2, dy2] Considering the rest of faeter-rcnn net are
			predicting the offset based on the given pooled region.
			All value is the ratio of delta in terms of the height and width (H, W)
			Assuming the proposal for the image is [x1, y1, x2, y2], the dx1 in the offset target
			means that the predicted bounding boxes coordinates should be like: x1 + dx1 * (x2 - x1)
	'''
	# This part of the process is implemented on my own, which does not based on any exiting code.
	# May be referring a little to the longcw repo, but it did not light me up. I still implemented
	# these through my own understanding.

	jittered = (jitter_box(gt_bbox) / img2feat_ratio).astype(int)

	# clip the boxes
	feat_H = features_shape[2]
	feat_W = features_shape[3]
	jittered = clip_boxes(jittered, (feat_H, feat_W))

	jit_x1 = jittered[:, 0]
	jit_y1 = jittered[:, 1]
	jit_x2 = jittered[:, 2]
	jit_y2 = jittered[:, 3]
	# check to prevent roi of size 0
	jit_x2[jit_x2 == jit_x1] += 1
	jit_y2[jit_y2 == jit_y1] += 1
	# calculate the target offsets
	H = jit_x2 - jit_x1
	W = jit_y2 - jit_y1

	offsets_targets = np.concatenate([
			np.expand_dims(np.divide((gt_bbox[:, 0] - jit_x1), H) , axis = 1), \
			np.expand_dims(np.divide((gt_bbox[:, 1] - jit_y1), W) , axis = 1), \
			np.expand_dims(np.divide((gt_bbox[:, 2] - jit_x2), H) , axis = 1), \
			np.expand_dims(np.divide((gt_bbox[:, 3] - jit_y2), W) , axis = 1), \
		], axis = 1)

	feature_rois = np.expand_dims(( \
			np.concatenate([
				np.expand_dims(jit_x1, axis = 1), \
				np.expand_dims(jit_y1, axis = 1), \
				np.expand_dims(jit_x2, axis = 1), \
				np.expand_dims(jit_y2, axis = 1), \
				], axis = 1)\
			), axis = 0)
	return feature_rois, offsets_targets
	