# LeNet-5

## Device
I implemented it only on CPU, since I haven't tried to figure out how to use cuda. Also, I don't hav ea GPU for now...

## Implementation Reference
1. I implemented the network architecture based on the image at https://medium.com/@shahariarrabby/lenet-5-alexnet-vgg-16-from-deeplearning-ai-2a4fa5f26344

2. Then I found that the activation function makes the learning speed a bit slow, and I modified it according to the implementation at https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py 

    This shows that LogSoftMax at the last layer really makes the learning faster.

## Dataset
* Using MNIST dataset at http://yann.lecun.com/exdb/mnist/
* You can extract all the files in the same folder, and the binHandler can read the binary images and labels using the functions in it.

## Usage
* Please invoke the training method at the main.py in the root folder. And don't forget to provide the dataset path at config.py
* You can also specify where you want to store the learnt model, which has been implemented in the LeNet class.
* When doing the test, you can provide either the learnt file or the object to the function.