import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor, RenameTransform


from batchgenerators.transforms.color_transforms import (BrightnessMultiplicativeTransform,
                                                         ContrastAugmentationTransform, GammaTransform)
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2

from .transforms import RemoveChannelAxisFromSeg


def get_transforms(config, is_train):
    """Get transforms.

    Args:
        config (YACS CfgNode): config.
        is_train (bool): True if the transforms are for train set.

    Returns:
        batchgenerators.transforms.Compose: transforms.
    """
    if is_train:
        # augmentations applied to train set
        transforms = [
            SpatialTransform_2(
                config.TRANSFORM.TRAIN_CROP_SIZE,
                do_elastic_deform=config.TRANSFORM.ENABLE_ELASTIC,
                deformation_scale=config.TRANSFORM.ELASTIC_SCALE,
                p_el_per_sample=config.TRANSFORM.ELASTIC_PROB,
                do_rotation=config.TRANSFORM.ENABLE_ROTATION,
                angle_x=(-config.TRANSFORM.ROTATION_X / 180. * np.pi, config.TRANSFORM.ROTATION_X / 180. * np.pi),
                angle_y=(-config.TRANSFORM.ROTATION_Y / 180. * np.pi, config.TRANSFORM.ROTATION_Y / 180. * np.pi),
                angle_z=(-config.TRANSFORM.ROTATION_Z / 180. * np.pi, config.TRANSFORM.ROTATION_Z / 180. * np.pi),
                p_rot_per_sample=config.TRANSFORM.ROTATION_PROB,
                do_scale=config.TRANSFORM.ENABLE_SCALE,
                scale=config.TRANSFORM.SCALE_RANGE,
                independent_scale_for_each_axis=False,
                p_scale_per_sample=config.TRANSFORM.SCALE_PROB,
                border_mode_data='constant',
                border_cval_data=config.TRANSFORM.IMAGE_PAD_VALUE,
                border_mode_seg='constant',
                border_cval_seg=config.TRANSFORM.LABEL_PAD_VALUE,
                order_seg=0,  # nearest interpolation for seg label
                order_data=3,
                random_crop=False)
        ]
        # random gaussign noise
        if config.TRANSFORM.ENABLE_GAUSSIAN:
            transforms.append(
                GaussianNoiseTransform(noise_variance=config.TRANSFORM.GAUSSIAN_VARIANCE,
                                       p_per_sample=config.TRANSFORM.GAUSSIAN_PROB))
        # random brightness
        if config.TRANSFORM.ENABLE_BRIGHTNESS:
            transforms.append(
                BrightnessMultiplicativeTransform(multiplier_range=config.TRANSFORM.BRIGHTNESS_RANGE,
                                                  p_per_sample=config.TRANSFORM.BRIGHTNESS_PROB))
        # random contrast
        if config.TRANSFORM.ENABLE_CONTRAST:
            transforms.append(
                ContrastAugmentationTransform(contrast_range=config.TRANSFORM.CONTRAST_RANGE,
                                              p_per_sample=config.TRANSFORM.CONTRAST_PROB))

        # random gamma
        if config.TRANSFORM.ENABLE_GAMMA:
            transforms.append(
                GammaTransform(gamma_range=config.TRANSFORM.GAMMA_RANGE,
                               invert_image=config.TRANSFORM.GAMMA_INVERT_IMAGE,
                               per_channel=False,
                               retain_stats=config.TRANSFORM.GAMMA_RETAIN_STATS,
                               p_per_sample=config.TRANSFORM.GAMMA_PROB))
    else:
        # augmentations applied to val set
        transforms = []

    # transforms mutual for train and val
    transforms.extend([
        NumpyToTensor(['data'], cast_to='float'),
        NumpyToTensor(['seg'], cast_to='long'),
        RemoveChannelAxisFromSeg(),
        RenameTransform(in_key='data', out_key='image', delete_old=True),
        RenameTransform(in_key='seg', out_key='target', delete_old=True)
    ])

    return Compose(transforms)
