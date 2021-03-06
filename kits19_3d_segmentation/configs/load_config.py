import argparse

from kits19_3d_segmentation.configs.defaults import get_default_config


def load_config():
    """Load config by merging default config with
    parameters given by YAML file and by command line args.

    Returns:
        YACS CfgNode: loaded config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to YAML config file', type=str)
    parser.add_argument('opts', default=None, help='parameter name and value pairs', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config:
        # overwrite hyper parameters with the ones given by YAML file.
        config.merge_from_file(args.config)
    if args.opts:
        # overwrite hyper parameters with the ones given by command line args.
        config.merge_from_list(args.opts)
    config.freeze()

    return config
