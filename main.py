# This is only a puppet main file that makes it easy to simply press F5 to run
# the program
import config
import torch
import matplotlib.pyplot as plt
import LeNetRecog.binHandler as binHandler
import LeNetRecog.LeNet as LeNet
import numpy as np

def test_LeNet_utils():    
    imgs = torch.from_numpy(binHandler.all_img(config.MnistData["train_img"])).float()
    labels = binHandler.all_label(config.MnistData["train_label"])
    for i in range(60000):
        im = plt.imshow(imgs[i][0], cmap = plt.cm.Greys)
        print(labels[i])
        plt.pause(0.3)
        plt.draw()

if __name__ == "__main__":
    #test_LeNet_utils()
    #trained_net = LeNet.train(config.MnistData)
    LeNet.test(config.MnistData, filepath = "./LeNetRecog/LeNet_learnt.pth")