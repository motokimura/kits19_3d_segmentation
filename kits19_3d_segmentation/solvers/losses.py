# Borowed a large part of the code from:
# https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.activations import get_activation
from ..utils.metrics import compute_dice


class CombinedLoss(nn.Module):
    """Loss module to compute weighted sum of the multiple losses.
    """
    def __init__(self, loss_modules, loss_weights):
        """

        Args:
            loss_modules (list[Module]): list of loss modules.
            loss_weights (list[float]): weight per each loss.
        """
        super(CombinedLoss, self).__init__()

        assert len(loss_modules) == len(loss_weights)

        self.loss_modules = loss_modules
        self.loss_weights = loss_weights

    def forward(self, pred, target):
        """

        Args:
            pred (Tensor): tensor with shape (N, C, H, W, D).
            target (Tensor): tensor with shape (N, C, H, W, D).

        Returns:
            Tensor: computed loss with shape (1,).
        """
        losses = self.loss_modules
        weights = self.loss_weights

        loss = losses[0](pred, target) * weights[0]
        for i in range(1, len(losses)):
            loss = loss + losses[i](pred, target) * weights[i]

        return loss


class DiceLoss(nn.Module):
    """Dice loss module.
    """
    def __init__(self, weight=None, p=1.0, epsilon=1e-6, activation=None, ignore_label=-1):
        """

        Args:
            weight (Tensor, optional): tensor of the weight per class with shape (C,). Defaults to None.
            p (float): exponential to compute union term of dice coeff. Defaults to 1.0.
            epsilon (float, optional): parameter to avoid zero division. Defaults to 1e-6.
            ignore_label (int): label assigned to the voxels which will be excluded from loss computation. Defaults to -1.
        """
        super(DiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.weight = weight
        self.p = p
        self.epsilon = epsilon
        if activation is None:
            self.activation = None
        else:
            self.activation = get_activation(activation)
        self.ignore_label = ignore_label

    def forward(self, pred, target):
        """

        Args:
            pred (Tensor): tensor with shape (N, C, H, W, D).
            target (Tensor): tensor with shape (N, H, W, D).

        Returns:
            Tensor: computed loss with shape (1,).
        """
        if self.activation is not None:
            pred = self.activation(pred)

        dice = compute_dice(pred,
                            target,
                            p=self.p,
                            epsilon=self.epsilon,
                            compute_per_sample=False,
                            ignore_label=self.ignore_label)  # (C, )

        # weight per class
        if self.weight is not None:
            dice = self.weight * dice

        return 1.0 - torch.mean(dice)


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss module whose input target can be one-hot.
    """
    def __init__(self, weight=None, ignore_label=-1):
        """

        Args:
            weight (Tensor, optional): tensor of the weight per class with shape (C,). Defaults to None.
            ignore_label (int): label assigned to the voxels which will be excluded from loss computation. Defaults to -1.
        """
        super(CrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.weight = weight
        self.ignore_label = ignore_label

    def forward(self, pred, target):
        """

        Args:
            pred (Tensor): tensor with shape (N, C, H, W, D).
            target (Tensor): tensor with shape (N, H, W, D).

        Returns:
            Tensor: computed loss with shape (1,).
        """
        return F.cross_entropy(pred, target, weight=self.weight, ignore_index=self.ignore_label)
