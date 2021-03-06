from yacs.config import CfgNode as CN

_C = CN()

# Data
_C.DATA = CN()
_C.DATA.KITS19_DIR = '/data/kits19/data'
_C.DATA.TRAIN_CASES = list(range(0, 210))
_C.DATA.TEST_CASES = list(range(210, 300))
_C.DATA.CASES_TO_EXCLUDE = [15, 23, 37, 68, 125, 133]  # see 'An attempt at beating the 3D U-Net' paper
_C.DATA.KITS19_RESAMPLED_DIR = '/data/kits19_resampled'
_C.DATA.SPACING = [3.22, 1.62, 1.62]  # XXX: axis along the vertical slice comes the first
_C.DATA.FOLD_NUM = 5
_C.DATA.FOLD_SEED = 777
_C.DATA.FOLD_ID = 0

# DataLoader
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH_SIZE = 1  # 2 in 'An attempt at beating the 3D U-Net' paper
_C.DATALOADER.TRAIN_NUM_WORKERS = 1
_C.DATALOADER.VAL_BATCH_SIZE = 1  # 2 in 'An attempt at beating the 3D U-Net' paper
_C.DATALOADER.VAL_NUM_WORKERS = 1
_C.DATALOADER.SHUFFLE_SEED = 777

# Transform
_C.TRANSFORM = CN()
# intensity normalization
intensity_min = -79.0
intensity_max = 304.0
intensity_mean = 101.0
intensity_std = 76.9
_C.TRANSFORM.INTENSITY_MIN = intensity_min
_C.TRANSFORM.INTENSITY_MAX = intensity_max
_C.TRANSFORM.INTENSITY_MEAN = intensity_mean
_C.TRANSFORM.INTENSITY_STD = intensity_std
# padding and (random) cropping
_C.TRANSFORM.TRAIN_RANDOM_CROP_SIZE = [
    160, 160, 80
]  # RANDOM crop size BEFORE spatial augmentations for train set. axis along the vertical slice comes the last
_C.TRANSFORM.TRAIN_CROP_SIZE = [
    160, 160, 80
]  # CENTER crop size AFTER spatial augmentations for train set. axis along the vertical slice comes the last
_C.TRANSFORM.VAL_CROP_SIZE = [
    256, 256, 128
]  # center crop size for val set (augmentations are not applied). axis along the vertical slice comes the last
_C.TRANSFORM.IMAGE_PAD_MODE = 'constant'  # ['constant', 'edge']
_C.TRANSFORM.IMAGE_PAD_VALUE = (intensity_min - intensity_mean) / intensity_std  # only valid when mode='constant'
_C.TRANSFORM.LABEL_PAD_VALUE = 0  # only used for train label (val label is padded with IGNORE_LABEL value)
# random elastic deformation
_C.TRANSFORM.ENABLE_ELASTIC = False
_C.TRANSFORM.ELASTIC_SCALE = (0, 0.25)
_C.TRANSFORM.ELASTIC_PROB = 0.2
# random rotation
_C.TRANSFORM.ENABLE_ROTATION = True
_C.TRANSFORM.ROTATION_X = 15.0  # in deg
_C.TRANSFORM.ROTATION_Y = 15.0  # in deg
_C.TRANSFORM.ROTATION_Z = 15.0  # in deg
_C.TRANSFORM.ROTATION_PROB = 0.2
# random scale
_C.TRANSFORM.ENABLE_SCALE = True
_C.TRANSFORM.SCALE_RANGE = (0.85, 1.25)
_C.TRANSFORM.SCALE_PROB = 0.2
# random noises
_C.TRANSFORM.ENABLE_GAUSSIAN = True
_C.TRANSFORM.GAUSSIAN_VARIANCE = (0, 0.1)
_C.TRANSFORM.GAUSSIAN_PROB = 0.1
# random brightness
_C.TRANSFORM.ENABLE_BRIGHTNESS = True
_C.TRANSFORM.BRIGHTNESS_RANGE = (0.75, 1.25)
_C.TRANSFORM.BRIGHTNESS_PROB = 0.15
# random contrast
_C.TRANSFORM.ENABLE_CONTRAST = True
_C.TRANSFORM.CONTRAST_RANGE = (0.75, 1.25)
_C.TRANSFORM.CONTRAST_PROB = 0.15
# random gamma
_C.TRANSFORM.ENABLE_GAMMA = True
_C.TRANSFORM.GAMMA_RANGE = (0.7, 1.5)
_C.TRANSFORM.GAMMA_RETAIN_STATS = True
_C.TRANSFORM.GAMMA_INVERT_IMAGE = True
_C.TRANSFORM.GAMMA_PROB = 0.1
# random seed
_C.TRANSFORM.AUGMENTATION_SEED = 777

# Model
_C.MODEL = CN()
_C.MODEL.NAME = 'plane_unet_3d'  # ['plane_unet_3d']
_C.MODEL.INPUT_CHANNELS = 1
_C.MODEL.OUTPUT_CHANNELS = 3
_C.MODEL.BASE_FEATURE_CHANNELS = 30
_C.MODEL.MAX_FEATURE_CHANNELS = 320
_C.MODEL.BASE_MODULE = 'double_conv'  # ['double_conv']
_C.MODEL.NUM_LEVELS = 6
_C.MODEL.NORMALIZATION = 'instance_norm'  # ['instance_norm', 'batch_norm']
_C.MODEL.NON_LINEARITY = 'leaky_relu'  # ['leaky_relu', 'relu']
_C.MODEL.CONV_KERNEL_SIZE = (3, 3, 3)
_C.MODEL.PADDING_WIDTH = (1, 1, 1)
_C.MODEL.FIRST_DOWNSAMPLE_STRIDE = (2, 2, 1)
_C.MODEL.ACTIVATION = 'softmax'  # ['softmax', 'sigmoid']
_C.MODEL.INITIALIZER = 'kaiming_normal'
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.WEIGHT = 'none'

# Training
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 1000
_C.TRAIN.LR = 1e-2
_C.TRAIN.LR_SCHEDULER = 'poly'  # ['poly']
_C.TRAIN.LR_POLY_EXPONENT = 0.9
_C.TRAIN.OPTIMIZER = 'sgd'  # ['sgd', 'adam']
_C.TRAIN.OPTIMIZER_SGD_MOMENTUM = 0.99
_C.TRAIN.OPTIMIZER_SGD_NESTEROV = True
_C.TRAIN.LOSSES = ['ce', 'dice']  # ['ce', 'dice']
_C.TRAIN.LOSS_WEIGHTS = [1.0, 1.0]
_C.TRAIN.WEIGHT_DECAY = 3e-5
_C.TRAIN.MAIN_VAL_METRIC = 'val/kits19/dice'
_C.TRAIN.VAL_INTERVAL = 2
_C.TRAIN.IGNORE_LABEL = -1
_C.TRAIN.CHECKPOINT_PATH = 'none'
_C.TRAIN.SEED = 777

# Inference
_C.TEST = CN()
_C.TEST.THRESHOLD_KIDNEY = 0.5
_C.TEST.THRESHOLD_TUMOR = 0.5

# Misc
_C.OUTPUT_DIR = './outputs'


def get_default_config():
    """Get default config.

    Returns:
        YACS CfgNode: default config.
    """
    return _C.clone()
