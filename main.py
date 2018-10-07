# This is only a puppet main file that makes it easy to simply press F5 to run
# the program
import config
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import LeNetRecog.binHandler as binHandler
import LeNetRecog.LeNet as LeNet

import FasterRCNNRecog.FasterRCNN as FasterRCNN

import torchvision.datasets as dset
import torchvision.transforms as transforms

import cv2

######################### Working on the LeNet #####################
def test_LeNet_utils():
    imgs = torch.from_numpy(binHandler.all_img(config.MnistData["train_img"])).float()
    labels = binHandler.all_label(config.MnistData["train_label"])
    for i in range(60000):
        im = plt.imshow(imgs[i][0], cmap = plt.cm.Greys)
        print(labels[i])
        plt.pause(0.3)
        plt.draw()

def LeNetDemo():
	#test_LeNet_utils()
	trained_net = LeNet.train(config.MnistData)
	LeNet.test(config.MnistData, filepath = "./LeNetRecog/LeNet_learnt.pth")

######################## Working on the Faster-RCNN ########################
def test_torch_COCOapi():
	from pycocotools.coco import COCO
	# The code is based on https://pytorch.org/docs/stable/torchvision/datasets.html#coco
	detections = dset.CocoDetection(root = config.COCOData["train_img"],
									annFile = config.COCOData["train_instances"],
									transform = transforms.ToTensor())

	print('Number of samples: ', len(detections))
	img, targets = detections[3] # load 4th sample

	print("Image Size: ", img.size())
	# dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
	print("Annotation keys: ", targets[2].keys())
	#plt.imshow(img.numpy().transpose((1, 2, 0)))
	#plt.show()
	print("Bounding boxes coordinates: ", end = "")
	print(targets[0]["bbox"])
	print("type of Bounding boxes coordinates: ", end = "")
	print(type(targets[0]["bbox"][1]))

	print("type of category_id:", end = "")
	print(type(targets[0]["category_id"]))

def coor_transform(shape, coordinates):
	'''	shape has to have 2 elements indicating image height and width.
		coordinates has to contain 4 elements.
	'''
	x1 = int(coordinates[0])
	y1 = int(coordinates[1])
	x2 = int(coordinates[0] + coordinates[2])
	y2 = int(coordinates[1] + coordinates[3])
	return x1, y1, x2, y2

def test_add_bbox():
	from pycocotools.coco import COCO
	detections = dset.CocoDetection(root = config.COCOData["train_img"],
									annFile = config.COCOData["train_instances"],
									transform = transforms.ToTensor())
	
	print('Number of samples: ', len(detections))
	print("Ploting image and bounding boxes...")
	plt.show()
	for img, targets in detections:
		img = img.numpy().transpose((1,2,0))
		img = np.ascontiguousarray(img)
		print("showing image(" + str(img.shape) + ") with " + str(len(targets)) + " bounding boxes")
		gt_labels = []
		for target in targets:
			coordinates = target["bbox"]
			gt_labels.append(target["category_id"])
			print(coordinates)
			x1, y1, x2, y2 = coor_transform(img.shape[0:2], coordinates)
			img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
			#cv2.rectangle(img, (0, 0), (20, 20), color=(0, 255, 0), thickness=3)
		plt.imshow(img)
		plt.pause(2)
		
def FasterRCNNDemo():
	net = FasterRCNN.train(config.COCOData)

if __name__ == "__main__":
	FasterRCNNDemo()
	pass