# training parameters
EPOCHS = 1
BATCH_SIZE = 4
NUM_CLASSES = 20
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
CHANNELS = 3

load_weights_before_training = False
load_weights_from_epoch = 0
save_frequency = 5

test_picture_dir = "/home/zhaoliu/data/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg"

test_images_during_training = False
training_results_save_dir = "./test_pictures/"
test_images_dir_list = ["", ""]

# When the iou value of the anchor and the real box is less than the IoU_threshold,
# the anchor is divided into negative classes, otherwise positive.
IOU_THRESHOLD = 0.6

# generate anchor
ASPECT_RATIOS = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
ANCHOR_STEPS = [8, 16, 32, 64, 100, 300]
MAP_SIZE = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
ANCHOR_SIZES = [[21., 45.], [45., 99.], [99., 153.], [153., 207.], [207., 261.], [261., 315.]]
# ANCHOR_SIZES = [(30., 60.), (60., 111.), (111., 162.), (162., 213.), (213., 264.), (264., 315.)]
# focal loss
alpha = 0.25
gamma = 2.0

reg_loss_weight = 0.5

# dataset
PASCAL_VOC_DIR = "//Users/james/Downloads/VOCdevkit/VOC2012/"
# The 20 object classes of PASCAL VOC
OBJECT_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

TXT_DIR = "/Users/james/IdeaProjects/TensorFlow2.0_SSD/voc_1.txt"

MAX_BOXES_PER_IMAGE = 20

# nms
NMS_IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
MAX_BOX_NUM = 50

# directory of saving model
save_model_dir = "/home/zhaoliu/workspace/TensorFlow2.0_SSD/saved_model/"
