# This script helps transforming all FILE FOR THE MNIST DATABASE
# into png file into a given folder.
import numpy as np
import struct
import cv2
import png

# the protocol is from http://yann.lecun.com/exdb/mnist/
def bin2mat(filename, if_show_progress= False):
    ''' This enumerator yields numpy matrices from the given filename.
        It yields one 2D matrix each time
    '''
    with open(filename, "rb") as file:
        # check magic number
        magic_number = struct.unpack(">i", file.read(4))[0]
        print("Reading file: {0}\nAnd the magic number is {1}".format(filename, magic_number), end = "\t")

        num_img = struct.unpack(">i", file.read(4))[0]
        (row, col) = struct.unpack(">ii", file.read(8))
        print("Total number of images is: {}".format(num_img))
        # yield matrix one by one.
        for i in range(num_img):
            mat = np.empty((row,col))
            for r in range(row):
                for c in range(col):
                    mat[r][c] = struct.unpack(">B", file.read(1))[0]
            if if_show_progress:
                print("Data extracted: {:06.2f}%".format(100 * i / num_img), end="\r")
            yield mat

    print("Finished reading all images")

def all_img(filename):
    ''' This method follows the protocol and read all the number at once.
        And it returns a numpy 4D-array of the data (60000, 1, 28, 28)
    '''
    with open(filename, "rb") as file:
        # check magic number
        magic_number = struct.unpack(">i", file.read(4))[0]
        print("Reading file: {0}\nAnd the magic number is {1}".format(filename, magic_number), end = "\t")

        num_img = struct.unpack(">i", file.read(4))[0]
        (row, col) = struct.unpack(">ii", file.read(8))
        print("Total number of images is: {}".format(num_img))

        # extact all numbers at once into a multi dimensional matrix
        # (big-endian unsigned byte) https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.dtypes.html
        all_data = np.fromfile(file, dtype = ">B")

        # reshape the data then output
        # (all images have only one channel)
        all_data = all_data.reshape((num_img, 1, row, col))
    return all_data

def bin2label(filename, if_show_progress= False):
    ''' This enumerator yields a number (from 0 to 9) each time as the corresponding label when
    you use it along with bin2mat() together.
    '''
    with open(filename, "rb") as file:
        # check magic number
        magic_number = struct.unpack(">i", file.read(4))[0]
        print("Reading file: {0}\nAnd the magic number is {1}".format(filename, magic_number), end = "\t")

        num_label = struct.unpack(">i", file.read(4))[0]
        # yield number one by one
        for i in range(num_label):
            if if_show_progress:
                print("Data extracted: {:06.2f}%".format(100 * i / num_label), end="\r")
            yield struct.unpack(">B", file.read(1))[0]

    print("Finished reading all labels")

def all_label(filename):
    ''' This method read labels all at once.
    '''
    with open(filename, "rb") as file:
        # check magic number
        magic_number = struct.unpack(">i", file.read(4))[0]
        print("Reading file: {0}\nAnd the magic number is {1}".format(filename, magic_number), end = "\t")

        num_label = struct.unpack(">i", file.read(4))[0]
        print("Total number of labes is: {}".format(num_label))

        # get the data and reshape
        # (big-endian unsigned byte) https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.dtypes.html
        labels = np.fromfile(file, dtype = ">B")

    return labels