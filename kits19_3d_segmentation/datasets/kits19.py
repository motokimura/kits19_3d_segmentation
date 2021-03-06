from pathlib import Path

import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading.data_loader import DataLoader


class KiTS19DataLoader(DataLoader):
    """KiTS19 DataLoader.
    """

    CLASSES = [
        'background',  # label=0
        'kidney',  # label=1
        'tumor'  # label=2
    ]

    def __init__(self, config, case_ids, is_train):
        """

        Args:
            config (YACS CfgNode): config.
            case_ids (list[int]): target case ids.
            is_train (bool): True if the dataloader is for train set.
        """
        # exclude some cases which may have faulty segmentation label
        case_ids = [i for i in case_ids if i not in config.DATA.CASES_TO_EXCLUDE]

        # init batchgenerators DataLoader
        if is_train:
            batch_size = config.DATALOADER.TRAIN_BATCH_SIZE  # N
            num_workers = config.DATALOADER.TRAIN_NUM_WORKERS
            shuffle = True
        else:
            batch_size = config.DATALOADER.VAL_BATCH_SIZE  # N
            num_workers = config.DATALOADER.VAL_NUM_WORKERS
            shuffle = False

        super(KiTS19DataLoader, self).__init__(data=case_ids,
                                               batch_size=batch_size,
                                               num_threads_in_multithreaded=num_workers,
                                               seed_for_shuffle=config.DATALOADER.SHUFFLE_SEED,
                                               return_incomplete=True,
                                               shuffle=shuffle,
                                               infinite=False)

        self.indices = list(range(len(case_ids)))

        # load parameters from config
        self.data_root = Path(config.DATA.KITS19_RESAMPLED_DIR)

        self.intensity_min = config.TRANSFORM.INTENSITY_MIN
        self.intensity_max = config.TRANSFORM.INTENSITY_MAX
        self.intensity_mean = config.TRANSFORM.INTENSITY_MEAN
        self.intensity_std = config.TRANSFORM.INTENSITY_STD

        self.image_pad_mode = config.TRANSFORM.IMAGE_PAD_MODE
        if self.image_pad_mode == 'constant':
            self.image_pad_kwargs = {'constant_values': config.TRANSFORM.IMAGE_PAD_VALUE}
        else:
            self.image_pad_kwargs = None

        if is_train:
            self.crop_type = 'random'
            self.crop_size = config.TRANSFORM.TRAIN_RANDOM_CROP_SIZE  # H, W, D
            self.target_pad_value = config.TRANSFORM.LABEL_PAD_VALUE
        else:
            self.crop_type = 'center'
            self.crop_size = config.TRANSFORM.VAL_CROP_SIZE  # H, W, D
            self.target_pad_value = config.TRAIN.IGNORE_LABEL

    def generate_train_batch(self):
        """Get a batch.

        Returns:
            dict: dict containing 'data', 'seg', and 'case_id'.
        """
        indices = self.get_indices()
        case_ids = [self._data[i] for i in indices]

        image = np.zeros([self.batch_size, 1, *self.crop_size], dtype=np.float32)  # (N, 1, H, W, D)
        target = np.zeros([self.batch_size, 1, *self.crop_size], dtype=np.float32)  # (N, 1, H, W, D)

        for i, case_id in enumerate(case_ids):
            # load image and labe as ndarray
            image_path = self.data_root / f'case_{case_id:05}/imaging.npy'
            a_image = np.load(image_path).astype(np.float32)
            a_image = a_image[np.newaxis, np.newaxis, :, :, :]  # (1, 1, h, w, d)

            target_path = self.data_root / f'case_{case_id:05}/segmentation.npy'
            a_target = np.load(target_path).astype(np.float32)
            a_target = a_target[np.newaxis, np.newaxis, :, :, :]  # (1, 1, h, w, d)

            assert a_image.shape == a_target.shape

            # normalize image intensity
            a_image = np.clip(a_image, self.intensity_min, self.intensity_max)
            a_image = (a_image - self.intensity_mean) / self.intensity_std

            # pad image and label if they are smaller than crops_size
            # TODO: check if pad_nd_image is really necessary (padding may be included in crop function)
            a_image = pad_nd_image(a_image,
                                   new_shape=self.crop_size,
                                   mode=self.image_pad_mode,
                                   kwargs=self.image_pad_kwargs)  # (1, 1, H', W', D')
            a_target = pad_nd_image(a_target,
                                    new_shape=self.crop_size,
                                    mode='constant',
                                    kwargs={'constant_values': self.target_pad_value})  # (1, 1, H', W', D')

            assert a_image.shape == a_target.shape

            # crop image and label
            a_image, a_target = crop(a_image, a_target, self.crop_size, crop_type=self.crop_type)  # (1, 1, H, W, D)

            image[0] = a_image[0]
            target[0] = a_target[0]

        return {
            'data': image,  # numpy.ndarray with shape (N, 1, H, W, D)
            'seg': target,  # numpy.ndarray with shape (N, 1, H, W, D)
            'case_id': case_ids  # list[int]
        }
