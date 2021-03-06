from batchgenerators.transforms.abstract_transforms import AbstractTransform


class RemoveChannelAxisFromSeg(AbstractTransform):
    """Remove channel axis (the 2nd axis) from seg array with shape: (N, C=1, H, W, D) -> (N, H, W, D).
    """
    def __init__(self, seg_key='seg'):
        """

        Args:
            seg_key (str, optional): seg key in data_dict. Defaults to 'seg'.
        """
        self.seg_key = seg_key

    def __call__(self, **data_dict):
        """

        Returns:
            dict: data_dict.
        """
        assert data_dict[self.seg_key].shape[1] == 1
        data_dict[self.seg_key] = data_dict[self.seg_key][:, 0, :, :, :]
        return data_dict
