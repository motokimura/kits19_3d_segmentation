from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from .losses import CombinedLoss, CrossEntropyLoss, DiceLoss


def get_loss(config):
    """Get combined loss module.

    Args:
        config (YACS CfgNode): config.

    Raises:
        ValueError:

    Returns:
        CombinedLoss: combined loss module.
    """
    def __get_loss(config, loss_name):
        if loss_name == 'ce':
            return CrossEntropyLoss(ignore_label=config.TRAIN.IGNORE_LABEL)
        elif loss_name == 'dice':
            return DiceLoss(activation=config.MODEL.ACTIVATION, ignore_label=config.TRAIN.IGNORE_LABEL)
        else:
            raise ValueError()

    losses = [__get_loss(config, loss_name) for loss_name in config.TRAIN.LOSSES]
    return CombinedLoss(losses, config.TRAIN.LOSS_WEIGHTS)


def get_lr_scheduler(config, optimizer):
    """Get learning rate scheduler.

    Args:
        config (YACS CfgNode): config.
        optimizer (torch.optim.Optimizer): optimizer.

    Raises:
        ValueError:

    Returns:
        torch.optim.lr_scheduler._LRScheduler: learning rate scheduler.
    """
    scheduler_name = config.TRAIN.LR_SCHEDULER
    if scheduler_name == 'poly':
        p = config.TRAIN.LR_POLY_EXPONENT
        max_epochs = config.TRAIN.EPOCHS
        return LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs)**p)
    else:
        raise ValueError()


def get_optimizer(config, model):
    """Get optimizer.

    Args:
        config (YACS CfgNode): config.
        model (torch.nn.Module): model to optimize.

    Raises:
        ValueError:

    Returns:
        torch.optim.Optimizer: optimizer.
    """
    optimizer_name = config.TRAIN.OPTIMIZER
    if optimizer_name == 'adam':
        return Adam(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif optimizer_name == 'sgd':
        return SGD(model.parameters(),
                   lr=config.TRAIN.LR,
                   weight_decay=config.TRAIN.WEIGHT_DECAY,
                   momentum=config.TRAIN.OPTIMIZER_SGD_MOMENTUM,
                   nesterov=config.TRAIN.OPTIMIZER_SGD_NESTEROV)
    else:
        raise ValueError()
