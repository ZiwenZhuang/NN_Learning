# This is only a puppet main file that makes it easy to simply press F5 to run
# the program
import config
import torch
import matplotlib.pyplot as plt
import numpy as np

import LeNetRecog.binHandler as binHandler
import LeNetRecog.LeNet as LeNet

import torchvision.datasets as dset
import torchvision.transforms as transforms

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
	#trained_net = LeNet.train(config.MnistData)
	LeNet.test(config.MnistData, filepath = "./LeNetRecog/LeNet_learnt.pth")

def test_torch_COCOapi():
	from pycocotools.coco import COCO
	# The code is based on https://pytorch.org/docs/stable/torchvision/datasets.html#coco
	detections = dset.CocoDetection(root = config.COCOData["train_img"],
									annFile = config.COCOData["train_instances"],
									transform = transforms.ToTensor())

	print('Number of samples: ', len(detections))
	img, target = detections[3] # load 4th sample

	print("Image Size: ", img.size())
	print("Annotation keys: ", target[2].keys())
	print()


if __name__ == "__main__":
	test_torch_COCOapi()
	pass