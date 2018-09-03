# Faster RCNN
## Device
It seems that running on CPU is ok for all the faster-rcnn process...

## Notes (What I have known about the faster RCNN)
* It is all computed using network, but not a continuous one. You have to train the RPN layer and the clasifier layer seperately.

## Reference...
1. The illustration from [this](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8) seems clear enough for me to start setting up the framework.  
	![illustration](https://cdn-images-1.medium.com/max/1000/1*wwKCoG-VtBycFeACBES4nA.jpeg)
2. The code is based on [this](https://github.com/longcw/faster_rcnn_pytorch)
3. The vgg16 architecture is as follows  
	![vgg_configuration](https://www.pyimagesearch.com/wp-content/uploads/2017/03/imagenet_vggnet_table1.png)
