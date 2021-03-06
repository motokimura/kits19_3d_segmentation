"""Randomly split case ids into N folds and save them as train_{fold_id}.json and val_{fold_id}.json.
"""

import json
import random
import timeit
from pathlib import Path

import numpy as np

from kits19_3d_segmentation.configs import load_config


def dump_case_ids(cases, out_path):
    """Dump case ids into a json file.

    Args:
        cases (list[int]): case ids.
        out_path (str): path to save json file.
    """
    with open(out_path, 'w') as f:
        json.dump(cases, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))


def main():
    t0 = timeit.default_timer()

    config = load_config()
    print('successfully loaded config:')
    print(config)

    out_dir = Path(config.DATA.KITS19_RESAMPLED_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)

    random.seed(config.DATA.FOLD_SEED)
    cases = config.DATA.TRAIN_CASES
    random.shuffle(cases)

    N = config.DATA.FOLD_NUM
    cases_divided = np.array([cases[i::N] for i in range(N)])

    for val_idx in range(N):
        # dump val case ids into a json file
        val_cases = cases_divided[val_idx].tolist()
        dump_case_ids(val_cases, str(out_dir / f'val_{val_idx}.json'))

        # dump train case ids into a json file
        train_mask = np.ones(N, dtype=bool)
        train_mask[val_idx] = False
        train_cases = cases_divided[train_mask]
        train_cases = np.concatenate(train_cases, axis=0).tolist()
        dump_case_ids(train_cases, str(out_dir / f'train_{val_idx}.json'))

    elapsed = timeit.default_timer() - t0
    print('time: {:.3f} min'.format(elapsed / 60.0))


if __name__ == '__main__':
    main()
