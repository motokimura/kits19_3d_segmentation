"""Do model validation on KiTS19 dataset with a given configuration.
"""

import json
import timeit
from pathlib import Path

from kits19_3d_segmentation.configs import load_config
from kits19_3d_segmentation.datasets import get_dataloader
from kits19_3d_segmentation.models import get_model
from kits19_3d_segmentation.solvers import get_loss
from kits19_3d_segmentation.utils.misc import configure_cudnn
from kits19_3d_segmentation.utils.training import print_logs, validate


def main():
    t0 = timeit.default_timer()

    configure_cudnn(deterministic=True, benchmark=False)

    # load config
    config = load_config()
    print('config:')
    print(config)
    print('')

    # prepare directory to save results
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dump config to a file
    with open(str(out_dir / 'config.yaml'), 'w') as f:
        f.write(str(config))

    # prepare data loadears
    val_dataloader = get_dataloader(config, is_train=False)

    # prepare model to validate
    model = get_model(config)

    # get loss
    criterion = get_loss(config)

    # val
    val_logs = validate(val_dataloader, model, criterion, config, save_dir=(out_dir / 'data'))
    print_logs(val_logs)

    # dump val results
    with open(str(out_dir / 'metrics.json'), 'w') as f:
        json.dump(val_logs, f)

    elapsed = timeit.default_timer() - t0
    print('time: {:.3f} min'.format(elapsed / 60.0))


if __name__ == '__main__':
    main()
