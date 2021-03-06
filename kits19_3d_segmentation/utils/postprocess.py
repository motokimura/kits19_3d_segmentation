import torch


def postprocess(tensor, tumor_label=2, kidney_label=1, tumor_thresh=0.5, kidney_thresh=0.5):
    """Apply post-processing to convert the predicted tensor into KiTS19 class labels.

    Args:
        tensor (Tensor): predicted tensor with shape (N, C, H, W, D).
        tumor_label (int, optional): tumor class label (=channel index) in pred tensor. Defaults to 2.
        kidney_label (int, optional): kidney class label (=channel index) in pred tensor. Defaults to 1.
        tumor_thresh (float, optional): threshold to detect tumor. Defaults to 0.5.
        kidney_thresh (float, optional): threshold to detect kidney. Defaults to 0.5.

    Returns:
        Tensor: tensor with shape (N, H, W, D). Each voxel represents label: {0:'background', 1:'kidney', 2:'tumor'}.
    """

    n, c, h, w, d = tensor.size()

    tensor = tensor.transpose(1, 0)  # (C, N, H, W, D)

    # 1. init all voxels with background label
    processed = torch.zeros([n, h, w, d], dtype=int).to(tensor.device)

    # 2. assign kidney label
    kidney_mask = tensor[kidney_label] >= kidney_thresh  # (N, H, W, D)
    processed[kidney_mask] = kidney_label

    # 3. assign tumor label
    tumor_mask = tensor[tumor_label] >= tumor_thresh  # (N, H, W, D)
    processed[tumor_mask] = tumor_label

    # 4. handle voxels where scores both for kidney and tumor are higher than the thresholds:
    # assing the class with higher score

    tk_mask = tumor_mask & kidney_mask  # (N, H, W, D)
    if tk_mask.sum() == 0:
        return processed  # if there is no such voxel, postprocess is already done

    # suppressss scores for other classes than tumor or kidney
    # so that they do not appear after torch.argmax() below
    other_classes = [i for i in range(c) if i not in [kidney_label, tumor_label]]
    tensor[other_classes] = -1

    # find the class (kidney or tumor) with higher score
    labels = torch.argmax(tensor[:, tk_mask].view(c, -1), dim=0)  # (S=tk_mask.sum(),)

    # all labels must be either `kidney_channel` or `tumor_channel`
    assert ((labels == kidney_label) | (labels == tumor_label)).sum() == labels.numel()

    processed[tk_mask] = labels
    return processed
