# This is only a puppet main file that makes it easy to simply press F5 to run
# the program
import config
import matplotlib.pyplot as plt
import LeNetRecog.binHandler as binHandler

if __name__ == "__main__":
    
    mat = binHandler.bin2mat(config.MnistData["train_img"])
    label = binHandler.bin2num(config.MnistData["train_label"])
    while True:
        im = plt.imshow(mat.__next__().astype(float), cmap = plt.cm.Greys)
        print(label.__next__())
        plt.pause(1)
        plt.draw()

    