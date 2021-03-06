"""Utilities for model training and validation.
"""
import math

import numpy as np
import torch
from tqdm import tqdm

from kits19_3d_segmentation.utils.activations import get_activation
from kits19_3d_segmentation.utils.logging import DictAverageMeter
from kits19_3d_segmentation.utils.metrics import compute_dice, compute_kits19_metrics
from kits19_3d_segmentation.utils.postprocess import postprocess


def train(train_dataloader, model, criterion, optimizer, config):
    """Run training for an epoch.

    Args:
        train_dataloader (batchgenerators.dataloading.MultiThreadedAugmenter): train data iterator.
        model (torch.nn.Module): model to train.
        criterion (torch.nn.Module): loss module.
        optimizer (torch.optim.Optimizer): optimizer.
        config (YACS CfgNode): config.

    Returns:
        dict: a dict which contains average value of train loss.
    """
    avg_meter = DictAverageMeter()
    num_iters = math.ceil(len(train_dataloader.generator.indices) / train_dataloader.generator.batch_size)

    model.train()
    for sample in tqdm(train_dataloader, total=num_iters):
        images = sample['image'].to(config.MODEL.DEVICE)
        targets = sample['target'].to(config.MODEL.DEVICE)

        # compute train loss
        preds = model(images)
        loss = criterion(preds, targets)

        # update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: check if the gradient accumulation improve the accuracy.

        # update average meter
        log = {'loss': loss.item()}
        avg_meter.update(log, n=1)

    # add prefix 'train/' to all keys
    logs = {}
    for k, v in avg_meter.avg.items():
        logs[f'train/{k}'] = v

    return logs


def validate(val_dataloader, model, criterion, config, save_dir=None):
    """Run validation.

    Args:
        val_dataloader (batchgenerators.dataloading.MultiThreadedAugmenter): val data iterator.
        model (torch.nn.Module): model to train.
        criterion (torch.nn.Module): loss module.
        config (YACS CfgNode): config.
        save_dir (pathlib.PosixPath, optional): path to save validation results. Defaults to None.

    Returns:
        dict: a dict which contains average values of validation loss and validation metrics.
    """
    avg_meter = DictAverageMeter()
    num_iters = math.ceil(len(val_dataloader.generator.indices) / val_dataloader.generator.batch_size)

    classes = val_dataloader.generator.CLASSES
    tumor_label = classes.index('tumor')
    kidney_label = classes.index('kidney')
    tumor_thresh = config.TEST.THRESHOLD_TUMOR
    kidney_thresh = config.TEST.THRESHOLD_KIDNEY

    activation = get_activation(config.MODEL.ACTIVATION)

    model.eval()
    with torch.no_grad():
        for sample in tqdm(val_dataloader, total=num_iters):
            images = sample['image'].to(config.MODEL.DEVICE)
            targets = sample['target'].to(config.MODEL.DEVICE)

            # compute val loss
            preds = model(images)
            loss = criterion(preds, targets)

            # apply activation (softmax, sigmoid, etc.)
            preds = activation(preds)

            # compute dice scores
            dices = {}
            dice = compute_dice(preds, targets, compute_per_sample=True, ignore_label=config.TRAIN.IGNORE_LABEL)
            dices['dice'] = ((dice[tumor_label] + dice[kidney_label]) / 2).item()
            for c in classes:
                dices[f'dice_{c}'] = dice[classes.index(c)].item()

            # compute KiTS19 dice scores
            preds_label = postprocess(preds,
                                      tumor_label=tumor_label,
                                      kidney_label=kidney_label,
                                      tumor_thresh=tumor_thresh,
                                      kidney_thresh=kidney_thresh)

            kits19_dices = compute_kits19_metrics(preds_label,
                                                  targets,
                                                  tumor_label=tumor_label,
                                                  kidney_label=kidney_label,
                                                  ignore_label=config.TRAIN.IGNORE_LABEL)

            # update average meter
            log = {'loss': loss.item()}
            log.update(dices)
            log.update(kits19_dices)
            avg_meter.update(log, n=1)

            # save image, pred, target arrays as .npy files
            if save_dir is not None:
                images = images.to('cpu').detach().numpy()
                targets = targets.to('cpu').detach().numpy()
                preds = preds.to('cpu').detach().numpy()
                preds_label = preds_label.to('cpu').detach().numpy()
                case_ids = sample['case_id']
                for i in range(len(images)):  # iterate over mini-batch
                    out_dir = save_dir / f'{case_ids[i]:05d}'
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(str(out_dir / 'image.npy'), images[i])
                    np.save(str(out_dir / 'target.npy'), targets[i])
                    np.save(str(out_dir / 'pred.npy'), preds[i])
                    np.save(str(out_dir / 'pred_label.npy'), preds_label[i])

    # add prefix 'val/' to all keys
    logs = {}
    for k, v in avg_meter.avg.items():
        logs[f'val/{k}'] = v

    return logs


def print_logs(logs, n_digits=4, prefix=''):
    """Print train/val logs nicely.

    Args:
        logs (dict): log of training or validation.
        n_digits (int, optional): number of the digits after the decimal point to show. Defaults to 4.
        prefix (str, optional): string to show before the log. Defaults to ''.
    """
    print(prefix, {k: round(v, n_digits) if isinstance(v, float) else v for k, v in logs.items()})
