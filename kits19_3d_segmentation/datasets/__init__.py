import json
from pathlib import Path

from batchgenerators.dataloading import MultiThreadedAugmenter

from ..transforms import get_transforms
from .kits19 import KiTS19DataLoader


def get_dataloader(config, is_train):
    """Get batchgenerators MultiThreadedAugmenter for KiTS19.

    Args:
        config (YACS CfgNode): config.
        is_train (bool): True if the dataloader is for train set.

    Returns:
        batchgenerators.dataloading.MultiThreadedAugmenter: KiTS19 MultiThreadedAugmenter.
    """
    # configure KiTS19DataLoader
    data_root = Path(config.DATA.KITS19_RESAMPLED_DIR)
    fold_id = config.DATA.FOLD_ID
    case_path = data_root / f'train_{fold_id}.json' if is_train else data_root / f'val_{fold_id}.json'
    with open(str(case_path)) as f:
        case_ids = json.load(f)
    dataloader = KiTS19DataLoader(config, case_ids, is_train)

    # configure Transform
    transforms = get_transforms(config, is_train)

    # configure MultiThreadedAugmenter
    num_workers = config.DATALOADER.TRAIN_NUM_WORKERS if is_train else config.DATALOADER.VAL_NUM_WORKERS
    augmenter = MultiThreadedAugmenter(dataloader,
                                       transforms,
                                       num_processes=num_workers,
                                       seeds=[config.TRANSFORM.AUGMENTATION_SEED] * num_workers)
    augmenter.restart()
    return augmenter
