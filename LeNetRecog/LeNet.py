﻿import torch
import torch.nn as nn
import numpy as np

from . import binHandler as bH

class LeNet(nn.Module):

	def __init__(self):
		super(LeNet, self).__init__()

		# start setting up the network
		self.conv = nn.Sequential(
			nn.Conv2d(1, 6, kernel_size = 5, stride = 1, padding = 2),
			nn.Sigmoid(),
			nn.AvgPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(6, 16, kernel_size = 5, stride = 1, padding = 0),
			nn.Sigmoid(),
			nn.AvgPool2d(kernel_size = 2, stride = 2),
			)
		self.fc = nn.Sequential(
			nn.Linear(400, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, 10),
			nn.Softmax(1)
			)

	def forward(self, x):
		''' Input x is the inital input for the net.
			There are 4 dimension for the input x (N, C, H, W), which means x contains the entire
		dataset when learning.
		'''
		# Check x dimensions
		assert list(x.shape)[1:] == [1, 28, 28]
		
		# feed the tensor forwards
		x = self.conv(x)
		# change the shape to feed to fully connected layers
		x = x.reshape((x.shape[0], -1))
		x = self.fc(x)

		return x

	@staticmethod
	def loss(out, target):
		''' Here I use cross-entropy loss function, since I didn't quite understand what the paper
		said the Maximum-likelihood Estimation.
		'''
		assert out.shape[0] == target.shape[0]

		criterion = nn.CrossEntropyLoss()
		return criterion(out, target)


def train(data_path):
	# read all the pictures 
	#and put them together in a single 4D matrix
	input = torch.from_numpy(bH.all_img(data_path["train_img"])).float()

	# read all labels 
	#and put them in the same shape as network output
	# Considering pytorch CrossEntropyLoss accept targets as only label numbers, I don't have to
	#change the shape.
	labels = torch.from_numpy(bH.all_label(data_path["train_label"])).long()

	# setup the network
	le_net = LeNet()
	# setup the optimizer and all those hyper-parameters
	lr = 2e-2	# learning rate
	optimizer = torch.optim.Adam(le_net.parameters(), lr = lr)

	print("\nStart training the network...")

	last_loss = 0
	for epo in range(999):
		# feed forward the tensors
		out = le_net(input)

		# get loss and calculate gradients
		optimizer.zero_grad()
		loss = LeNet.loss(out, labels)
		loss.backward()
	
		# perform optimization
		optimizer.step()

		# report porformance
		print("In epoch {:>3}, the loss is {:<15}".format(epo, loss.item()))
		if abs(loss.item() - last_loss) < 0.0001:
			break
		else:
			last_loss = loss.item()

	print("End training the network!")
	return le_net