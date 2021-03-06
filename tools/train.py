"""Train model on KiTS19 dataset with a given configuration.
"""

import timeit
from pathlib import Path

import torch
import wandb

from kits19_3d_segmentation.configs import load_config
from kits19_3d_segmentation.datasets import get_dataloader
from kits19_3d_segmentation.models import get_model
from kits19_3d_segmentation.solvers import get_loss, get_lr_scheduler, get_optimizer
from kits19_3d_segmentation.utils.checkpoint import load_checkpoint, save_checkpoint
from kits19_3d_segmentation.utils.misc import configure_cudnn, set_seed
from kits19_3d_segmentation.utils.training import print_logs, train, validate
from kits19_3d_segmentation.utils.wandb import configure_wandb


def main():
    t0 = timeit.default_timer()

    # load config
    config = load_config()
    print('config:')
    print(config)
    print('')

    # fix random seed
    set_seed(config.TRAIN.SEED)
    configure_cudnn(deterministic=True, benchmark=False)

    # prepare directory to save results
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # init w&b
    # name of the output directory is used as w&b group name
    # TODO: make this better
    configure_wandb(project='kits19_3d_segmentation', group=out_dir.name, config=config)

    # dump config to a file
    with open(str(out_dir / 'config.yaml'), 'w') as f:
        f.write(str(config))

    # prepare data loadears
    train_dataloader = get_dataloader(config, is_train=True)
    val_dataloader = get_dataloader(config, is_train=False)

    # prepare model to train
    model = get_model(config)

    # prepare optimizer and lr-scheduler
    optimizer = get_optimizer(config, model)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    # prepare loss
    criterion = get_loss(config)

    start_epoch = 0
    best_score = -1.0
    best_val_logs = {}

    # resume from checkpoint if provided
    checkpoint_path = config.TRAIN.CHECKPOINT_PATH
    if checkpoint_path and checkpoint_path != 'none':
        model, optimizer, lr_scheduler, start_epoch, best_score, best_val_logs = load_checkpoint(
            checkpoint_path, model, optimizer, lr_scheduler, start_epoch, best_score, best_val_logs)
        print(f'resume training from {checkpoint_path}')

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        logs = {'epoch': epoch}

        lr = lr_scheduler.get_last_lr()[0]
        print(f"\nEpoch: {epoch} / {config.TRAIN.EPOCHS}, lr: {lr:.9f}")
        logs['lr'] = lr

        # train
        print('training...')
        train_logs = train(train_dataloader, model, criterion, optimizer, config)
        print_logs(train_logs)
        logs.update(train_logs)

        # val
        if (epoch + 1) % config.TRAIN.VAL_INTERVAL == 0:
            print('validating...')
            val_logs = validate(val_dataloader, model, criterion, config)
            print_logs(val_logs)
            logs.update(val_logs)

            score = val_logs[config.TRAIN.MAIN_VAL_METRIC]
            if score > best_score:
                # update best score and save model weight
                best_score = score
                best_val_logs = val_logs
                torch.save(model.state_dict(), str(out_dir / 'model_best.pth'))
            print_logs(best_val_logs, prefix='best val scores:')
        else:
            print(f"skip val since val interval is set to {config.TRAIN.VAL_INTERVAL}")

        # update lr
        lr_scheduler.step()

        # save checkpoint
        save_checkpoint(str(out_dir / 'checkpoint_latest.pth'), model, optimizer, lr_scheduler, epoch + 1, best_score,
                        best_val_logs)

        # log
        wandb.log(logs)

    wandb.finish()

    elapsed = timeit.default_timer() - t0
    print('time: {:.3f} min'.format(elapsed / 60.0))


if __name__ == '__main__':
    main()
