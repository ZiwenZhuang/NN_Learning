# This module performs 2 operations mentioned in the paper.
# from x, h, y, w to t_x, t_y, t_h, t_w (transform)
# from t_x, t_y, t_h, t_w to x, h, y, w (inverse)
# providing the anchors information (x1, y1, x2, y2) in a (N, 4) matrix

# To be more specific, I changes the coordinate definitions a little bit.
# Here in a matrix, I defines the row index as x                 +----------------->      
# and column index as y, which makes them looks                  |        w        y      
# just like the normal coordinate system.                       h|                        
# But there are still some drawbacks, where height               |                        
# and width are hard to define. As a compromise,                 |                        
# I have done something on the right.                           xV                        

import numpy as np

def corner2centered(box):
	''' A sub-tool that change the parameter form (x1, y1, x2, y2) into (x_c, y_c, h, w)
		The input has to be a 2-dimensional numpy array where the size should be (N, 4).
	'''
	x = (box[:, 2] + box[:, 0]) / 2
	y = (box[:, 3] + box[:, 1]) / 2
	h = (box[:, 2] - box[:, 0])
	w = (box[:, 3] - box[:, 1])
	
	return np.concatenate((x, y, h, w), axis= 1)

def centered2corner(box):
	'''	A tool that transform the parameter from (x_c, y_c, h, w) into (x1, y1, x2, y2)
		The input has to be a 2-dimensional numpy array where the size should be (N, 4)
	'''

	half_h = box[:, 2] / 2
	half_w = box[:, 3] / 2

	x1 = box[:, 0] - half_h
	y1 = box[:, 1] - half_w
	x2 = box[:, 0] + half_h
	y2 = box[:, 1] + half_w

	return np.concatenate((x1, y1, x2, y2), axis=1)

def bbox_transform(anchors, bbox):
	# x to x_t
	assert anchors.shape[0] == bbox.shape[0], \
		"Different number of anchors and the number of bounding boxes"

	anchors = corner2centered(anchors)
	bbox = corner2centered(bbox)
	# Now both anchors and bounding boxes are in a (x, y, h, w)-form
	
	tx = (bbox[:, 0] - anchors[:, 0]) / anchors[:, 2]
	ty = (bbox[:, 1] - anchors[:, 1]) / anchors[:, 3]
	th = np.log(bbox[:, 2] / anchors[:, 2])
	tw = np.log(bbox[:, 3] / anchors[:, 3])

	return np.concatenate((tx, ty, th, tw), axis= 1)


def bbox_transform_inv(anchors, gt_pred):
	# x_t to x (inverse)
	assert anchors.shape[0] == gt_pred.shape[0], \
		"Different number of anchors and the number of predictions"

	anchors = corner2centered(anchors)
	# Now both anchors and prediction are in (x, y, h, w)-form

	x = anchors[:, 0] + np.multiply(anchors[:, 2], gt_pred[:, 0])
	y = anchors[:, 1] + np.multiply(anchors[:, 3], gt_pred[:, 1])
	h = np.multiply(anchors[:, 2], np.exp(gt_pred[:, 2]))
	w = np.multiply(anchors[:, 3], np.exp(gt_pred[:, 3]))

	centered_form = np.concatenate((x, y, h, w), axis= 1)
	return centered2corner(centered_form)