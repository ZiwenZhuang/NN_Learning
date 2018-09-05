# The file is used as part of the region proposal network, which makes the whole network two
# parts :p
# And the code is based on https://github.com/longcw/faster_rcnn_pytorch/blob/master/faster_rcnn/rpn_msr/proposal_layer.py
# If you want to see the original code, please clone the whole repo for a closer view.

import numpy as np

def generate_anchor(img_info, stride = 1, scales = [8, 16, 32], ratios = [0.5, 1, 2]):
	'''	Generating the 4 parameters (x1, y1, x2, y2) of each anchor.
		And concatenate them along the 0-th dimension.
		---------------------------------
		img_info: the [height, width] of the image                         x1, y1--------+
		stride: the stride that the sliding window would move                 |          |
		scales: the width of anchor (when it is a square)                     |          |
		ratios: under certain scale, the ratio between width and height       +-------x2, y2
		---------------------------------
		output: a numpy matrix (1, H, W, _num_anchors*4) a series of anchor parameters.
			along the last axis is like (x1, y1, x2, y2, x1, y1, x2, y2, ...)
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

	# (1, _num_anchors*4, H, W)
	anchors = anchors.reshape((1, -1, img_info[0] // stride, img_info[1] // stride))

	# (1, H, W, _num_anchors*4)
	anchors = np.transpose(anchors, axis=(0, 2, 3, 1))
	return anchors

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
	boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[0] - 1), 0)
	# series of y2
	boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[1] - 1), 0)

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
		predictions: numpy 2darray (N, 5), along the second axies (x1, y1, x2, y2, score)
		---------------------
		Returns
		----------
		numpy 1darray served as indexes of proposals (along the 0-th axis)
	'''


def proposal_layer(rpn_score, rpn_bbox, img_info, anchor_scales, anchor_ratios, configs):
	'''	Extract useful proposals that are both high in score and big in area.
		---------------------
		Parameters
		----------
		rpn_score: is np array (1, H, W, A*2) 4-dimensions
		rpn_bbox: the network predicted slight shift for the "ground true" bounding box relative to
			the anchor at the specific point. (1, H, W, A*4) 4-dimensions
			where A is the number of anchors for each point (in the feature map)
		img_info: the [height, width] of the input image (I means the input image of the whole network)
		anchor_scales: the anchor size if put onto the input images
		anchor_ratios: the ratio between the height and the width of the anchor
		configs: requires (nms_thresh, pre_nms_topN, post_nms_topN) keys as configuration
		----------------------
		Returns
		----------
		rpn_rois: a batch of bunch of rois that are elected from the network proposals.
			(1, H * W * A, 5) where H and W are the size of input feature map (rpn_score, rpn_bbox)
			and A is the number of anchors for each point in the feature map
	'''
	#	Considering the output proposal for each image is different, you should provide batch with
	# only 1 item, or it is difficult for this method to return the same amount of proposals for
	# the whole batch.
	assert rpn_score.shape[0] == 1 & rpn_rpn_bbox.shape[0] == 1, \
		"Only single item batches are supported"

	#	Considering the output from feature map (The first step in RPN network) has the same size
	# as the feature map, we regard the rpn_score/rpn_bbox just has the same size as bounding box
	# size.
	#	Be sure to check the output of self.score_conv and self.bbox_conv for the output size!!!!
	feature_shape = rpn_score.shape[1:3]
	
	# 0. get anchors in (1, H, W, _num_anchors*4) where only the last dimension is the anchor parameter
	anchors = generate_anchor(rpn_bbox.shape[1:3], scales= anchor_scales, ratios= anchor_ratios)

	# 1. apply delta (from bbox prediction) to each of the anchor
	#	and reshape to (H * W * _num_anchors, 4)
	pd_bbox = anchors + rpn_bbox
	pd_bbox = pd_bbox.reshape((-1, 4))
	    # the first set of channels are back_ground probs
		# the second set are the fore_ground probs, which we want
	pd_scores = rpn_score.reshape((-1, 2))[:, 2]
	pd_scores = pd_scores.reshape(-1) # remove the 1-th dimension

	# 2. clip prediction bounding boxes to image size (using the size of the feature map here)
	pd_bbox = clip_boxes(pd_bbox, feature_shape)

	# 3. get rid of anchors with too small size or too strange height/width ratio
	keep = _filter_boxes(pd_bbox, img_info, feature_shape)
	pd_bbox = pd_bbox[keep]
	pd_scores = pd_scores[keep]

	# 4. sort all (proposal, score) pair by score from HIGHest to LOWest
	# 5. take the pre_nms_topN number of proposals
	order = pd_scores.ravel().argsort()[::-1]
	if (configs["pre_nms_topN"] > 0):
		order = order[configs["pre_nms_topN"], :]
	pd_bbox = pd_bbox[order]
	pd_scores = pd_scores[order]

	# 6. perform NMS
	keep = nms(np.hstack(pd_bbox, pd_scores), configs["nms_thresh"])