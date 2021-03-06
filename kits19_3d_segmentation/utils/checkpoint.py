import torch


def save_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, epoch, best_score, best_val_logs):
    """Save training checkpoint as a file.

    Args:
        checkpoint_path (str): path to save the checkpoint.
        model (torch.nn.Module): model.
        optimizer (torch.optim.Optimizer): optimizer.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler.
        epoch (int): current epoch.
        best_score (float): best val score.
        best_val_logs ([type]): val metrics at the epoch where the best val score achieved.
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'best_score': best_score,
        'best_val_logs': best_val_logs
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, epoch, best_score, best_val_logs):
    """Load training states from a checkpoint file.

    Args:
        checkpoint_path (srt): path to the checkpoint file to load.
        model (torch.nn.Module): model.
        optimizer (torch.optim.Optimizer): optimizer.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler.
        epoch (int): current epoch (dummy).
        best_score (float): best val score (dummy).
        best_val_logs ([type]): val metrics at the epoch where the best val score achieved (dummy).

    Returns:
        tuple: training states loaded from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch']
    best_score = checkpoint['best_score']
    best_val_logs = checkpoint['best_val_logs']

    return model, optimizer, lr_scheduler, epoch, best_score, best_val_logs
