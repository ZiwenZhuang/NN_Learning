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
	"train_img": Data_Path,
	"train_target": Data_Path,
	"test_img": Data_Path,
	"test_target": Data_Path
	}