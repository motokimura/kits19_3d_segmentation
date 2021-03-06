import torch
import torch.nn.functional as F


def compute_dice(pred, target, p=1.0, epsilon=1e-6, compute_per_sample=False, ignore_label=-1):
    """Compute dice score for each class.

    Args:
        pred (Tensor): tensor with shape (N, C, H, W, D). Each voxel represents class scores.
        target (Tensor): tensor with shape (N, H, W, D). Each voxel represents target label.
        p (float): exponential to compute union term. Defaults to 1.0.
        epsilon (float): parameter to avoid zero division. Defaults to 1e-6.
        compute_per_sample (bool): compute dice score per sample in the batch if True.
        ignore_label (int): label assigned to the voxels which will be excluded from the dice computation. Defaults to -1.

    Returns:
        Tensor: computed dice with shape (C, ).
    """
    # only the voxels where mask=True contribute dice computation
    mask = (target != ignore_label)  # (N, H, W, D)
    target_ = target.clone()  # necessary to avoid in-place operation
    target_[torch.logical_not(mask)] = 0  # set label=0 anyway (these voxels will be excluded from dice computation)

    # convert target to one-hot representation
    target_ = F.one_hot(target_, num_classes=pred.size(1))  # (N, H, W, D, C)
    target_ = target_.permute(4, 0, 1, 2, 3)  # (C, N, H, W, D)
    target_ = target_.float()

    pred = pred.permute(1, 0, 2, 3, 4)  # (C, N, H, W, D)

    assert pred.shape == target_.shape

    # flatten spatial dims
    pred = pred.flatten(start_dim=2)  # (C, N, S=H*W*D)
    target_ = target_.flatten(start_dim=2)  # (C, N, S)
    mask = mask.flatten(start_dim=1)  # (N, S)

    def _compute_dice(pred, target, spatial_dim):
        """Compute dice score for each class.

        Args:
            pred (Tensor): tensor with shape (C, S). Each voxel represents class scores.
            target (Tensor): tensor with shape (C, S). Each voxel represents class scores.
            spatial_dim (int): index of spatial dim of pred & target tensors.

        Returns:
            Tensor: computed dice with shape (C, ).
        """
        intersect = (pred * target).sum(spatial_dim)  # (C, )
        union = (pred**p).sum(spatial_dim) + (target**p).sum(spatial_dim)  # (C, )
        return 2.0 * (intersect / union.clamp(min=epsilon))  # (C, )

    if compute_per_sample:
        # compute dice score for each sample of the batch, and compute mean dice in the batch
        C = pred.size(0)
        N = pred.size(1)
        dice = torch.zeros(N, C).to(pred.device)  # (N, C)
        for i in range(N):
            pred_i = pred[:, i]  # (C, S)
            target_i = target_[:, i]  # (C, S)
            mask_i = mask[i]  # (S, )

            pred_i = pred_i[:, mask_i]  # (C, S')
            target_i = target_i[:, mask_i]  # (C, S')
            dice[i] = _compute_dice(pred_i, target_i, spatial_dim=1)
        return dice.mean(0)  # (N, C) -> (C, )
    else:
        # consider batch dimension as a pseudo spatial dimension
        pred = pred[:, mask]  # (C, S')
        target_ = target_[:, mask]  # (C, S')
        return _compute_dice(pred, target_, spatial_dim=1)  # (C, )


def compute_kits19_metrics(pred, target, tumor_label=2, kidney_label=1, ignore_label=-1):
    """Compute KiTS19 official evaluation metrics.

    Args:
        pred (Tensor): tensor with shape (N, H, W, D). Each voxel represents pred label.
        target (Tensor): tensor with shape (N, H, W, D). Each voxel represents target label.
        tumor_channel (int, optional): label of tumor class. Defaults to 2.
        kidney_channel (int, optional): label of kidney class. Defaults to 1.
        ignore_label (int): label assigned to the voxels which will be excluded from the dice computation. Defaults to -1.

    Returns:
        dict: dice scores used for KiTS19 evaluation.
    """
    assert pred.shape == target.shape

    # compute dice each for tumor class and kidney class
    num_classes = 3  # ['background', 'kidney', 'tumor']
    pred = F.one_hot(pred, num_classes)  # (N, H, W, D, C=3)
    pred = pred.permute(0, 4, 1, 2, 3)  # (N, C, H, W, D)
    pred = pred.float()  # (N, C, H, W, D)

    dice_per_class = compute_dice(pred, target, p=1.0, compute_per_sample=True, ignore_label=ignore_label)
    tumor_dice = dice_per_class[tumor_label].item()
    kidney_dice = dice_per_class[kidney_label].item()

    # compute dice for (tumor|kidney) class
    # notice pred tensor is already converted into one-hot
    tk_pred = (pred[:, tumor_label] + pred[:, kidney_label]) > 0  # (N, H, W, D)
    tk_pred = tk_pred.unsqueeze(1)  # (N, 1, H, W, D)
    bg_pred = torch.logical_not(tk_pred).to(tk_pred.device)  # (N, 1, H, W, D)
    tk_pred = torch.cat([bg_pred, tk_pred], dim=1)  # (N, C=2, H, W, D), c=0:background, c=1:(tumor|kidney)
    tk_pred = tk_pred.float()

    tk_target = (target == tumor_label) | (target == kidney_label)  # (N, H, W, D)
    tk_target = tk_target.long()  # 0:background, 1:(tumor|dice)
    tk_target[target == ignore_label] = ignore_label  # leave ignore_label as it is

    tk_dice = compute_dice(tk_pred, tk_target, p=1.0, compute_per_sample=True, ignore_label=ignore_label)  # (C=2,)
    tk_dice = tk_dice[1].item()  # get dice for c=1:(tumor|kidney)

    # compute KiTS19 main evaluation metric
    dice = (tk_dice + tumor_dice) / 2

    return {
        'kits19/dice': dice,
        'kits19/dice_tumor+kidney': tk_dice,
        'kits19/dice_tumor': tumor_dice,
        'kits19/dice_kidney': kidney_dice
    }
