"""Resample KiTS19 images and labels with a fixed spacing and save them as .npy files.
"""

# TODO: use skimage for image/label resampling like nnUNet
# https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py#L109

import timeit
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from kits19_3d_segmentation.configs import load_config


def resample_spacing(config, itk_image, resample_method):
    """Apply resampling to an image with a fixed spacing.

    Args:
        config (YACS CfgNode): config.
        itk_image (SimpleITK Image): SimpleITK Image object to resample.
        resample_method (int): SimpleITK resampler flag.

    Returns:
        SimpleITK Image: SimpleITK Image object after resampling.
    """
    size = np.array(itk_image.GetSize())
    spacing = np.array(itk_image.GetSpacing())
    new_spacing = np.array(config.DATA.SPACING)
    new_size = (size * (spacing / new_spacing)).astype(int)

    return sitk.Resample(itk_image, new_size.tolist(), sitk.Transform(), resample_method, itk_image.GetOrigin(),
                         new_spacing, itk_image.GetDirection(), 0.0, itk_image.GetPixelID())


def resample(config, case_id, is_label):
    """Load an image, apply resampling with a given spacing,
    and save it as a numpy file.

    Args:
        config (YACS CfgNode): config.
        case_id (int): case id of the target image.
        is_label (bool): True if target is labels.
    """
    if is_label:
        filename = 'segmentation'
        dtype = int
        resample_method = sitk.sitkNearestNeighbor
    else:
        filename = 'imaging'
        dtype = float
        resample_method = sitk.sitkBSplineResamplerOrder3

    kits19_dir = Path(config.DATA.KITS19_DIR)
    path = kits19_dir / f'case_{case_id:05}/{filename}.nii.gz'
    itk_image = sitk.ReadImage(str(path))
    itk_image = resample_spacing(config, itk_image, resample_method)
    array = sitk.GetArrayFromImage(itk_image).astype(dtype)
    # axis along the vertical slice comes the last in the array
    # e.g., array.shape = (248, 248, 128)

    # save
    out_root = Path(config.DATA.KITS19_RESAMPLED_DIR)
    out_dir = out_root / f'case_{case_id:05}'
    out_dir.mkdir(exist_ok=True, parents=True)
    np.save(out_dir / f'{filename}.npy', array)


def main():
    t0 = timeit.default_timer()

    config = load_config()
    print('successfully loaded config:')
    print(config)

    print("processing labels...")
    for case_id in tqdm(config.DATA.TRAIN_CASES):
        resample(config, case_id, is_label=True)

    print('processing images...')
    for case_id in tqdm(config.DATA.TRAIN_CASES):
        resample(config, case_id, is_label=False)

    elapsed = timeit.default_timer() - t0
    print('time: {:.3f} min'.format(elapsed / 60.0))


if __name__ == '__main__':
    main()
