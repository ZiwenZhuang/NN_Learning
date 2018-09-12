# normally I don't think there is a GPU avaliable on your device
CUDA = False

Data_Path = "../VisionData/"

MnistData = {
    "train_img": Data_Path + "MNIST-dataset/train-images.idx3-ubyte",
    "train_label": Data_Path + "MNIST-dataset/train-labels.idx1-ubyte",
    "test_img": Data_Path + "MNIST-dataset/t10k-images.idx3-ubyte",
    "test_label": Data_Path + "MNIST-dataset/t10k-labels.idx1-ubyte"
    }

COCOData = {
	"train_img": Data_Path + "COCO2014/train2014/",
	"test_img": Data_Path + "COCO2014/val2014/",
	"annotations": Data_Path + "COCO2014/annotations/",
	}
COCOData.update({
	"train_captions": COCOData["annotations"] + "captions_train2014.json",
	"test_captions": COCOData["annotations"] + "captions_val2014.json",
	"train_instances": COCOData["annotations"] + "instances_train2014.json",
	"test_instances": COCOData["annotations"] + "instances_val2014.json",
	})