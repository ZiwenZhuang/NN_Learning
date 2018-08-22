# This script helps transforming all FILE FOR THE MNIST DATABASE
# into png file into a given folder.
import numpy as np
import struct
import cv2
import png

# the protocol is from http://yann.lecun.com/exdb/mnist/
def bin2mat(filename):
    ''' This enumerator yields numpy matrices from the given filename.
        It yields one 2D matrix each time
    '''
    with open(filename, "rb") as file:
        # check magic number
        magic_number = struct.unpack(">i", file.read(4))[0]
        print("Reading file: {0}\nAnd the magic number is {1}".format(filename, magic_number))

        num_img = struct.unpack(">i", file.read(4))[0]
        (row, col) = struct.unpack(">ii", file.read(8))
        print("Total number of images is: {}".format(num_img))
        # yield matrix one by one.
        for i in range(num_img):
            mat = np.empty((row,col))
            for r in range(row):
                for c in range(col):
                    mat[r][c] = struct.unpack(">B", file.read(1))[0]
            yield mat

def bin2num(filename):
    ''' This enumerator yields a number (from 0 to 9) each time as the corresponding label when
    you use it along with bin2mat() together.
    '''
    with open(filename, "rb") as file:
        # check magic number
        magic_number = struct.unpack(">i", file.read(4))[0]
        print("Reading file: {0}\nAnd the magic number is {1}".format(filename, magic_number))

        num_label = struct.unpack(">i", file.read(4))[0]
        # yield number one by one
        for i in range(num_label):
            yield struct.unpack(">B", file.read(1))[0]