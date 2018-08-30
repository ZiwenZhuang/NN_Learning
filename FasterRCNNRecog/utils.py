# This module contains several tools that helps chnging the metworks
import numpy as mp
import torch
import torch.nn as nn

def set_trainable(model, require_grad = True):
	for param in model.parameters():
		param.requires_grad = require_grad