# kits19_3d_segmentation

## Requirements

- Ubuntu OS (tested with Ubuntu 20.04 LTS)
- NVIDIA Driver (tested with 455.32.00)
- Docker (tested with 19.03.12)
- Docker Compose (tested with 1.27.4)
- NVIDIA Container Tookit (tested with 2.4.0)

May work with other versions.

## Setup

Prepare the container:

```
docker-compose run --rm --service-ports kits19 bash
```

All the commands below have to be executed in this container.

## Prepare KiTS19 Dataset

### Download

Download KiTS19 dataset:

```
./scripts/download_kits19.sh
```

This script downloads the KiTS19 dataset under `$HOME/data/kits19/data` on your host machine
(host `$HOME/data` is mounted to container `/data`).

### Resample

Apply re-sampling with a fixed spacing to the images:

```
python tools/resample_kits19.py
```

This script saves the re-sampled images and labels under `$HOME/data/kits19_preprocessed` on your host machine.

### Make train/val folds

Split case ids into N folds (N=5 at default) randomly:

```
python tools/make_folds.py
```

This script saves `train_*.json` and `val_*.json` under `$HOME/data/kits19_preprocessed` on your host machine.

## Training

Get your API key from [W&B](https://wandb.ai) and then:

```
# replace xxxx with your own W&B API key
echo 'WANDB_API_KEY = "xxxx"' > .env

# train.py loads the API key from `.env` to send training logs to W&B
python tools/train.py OUTPUT_DIR ./outputs/training
```

This script saves training results (trained weight files, model configurations, etc.) under `./outputs/training` directory.

## Evaluation

Run validation:

```
python tools/val.py --config ./outputs/training/config.yaml MODEL.WEIGHT ./outputs/training/model_best.pth OUTPUT_DIR  ./outputs/validation
```

This script saves validation results (validation score, prediction results, etc.) under `./outputs/validation` directory.


## Inference & Visualization

To be updated.
