from os import environ

import wandb
from dotenv import load_dotenv


def configure_wandb(project=None, group=None, config=None):
    """Configure W&B.

    Args:
        project (str, optional): W&B project name. Defaults to None.
        group (str, optional): W&B group name. Defaults to None.
        config (YACS CfgNode, optional): config. Defaults to None.
    """
    load_dotenv()  # load WANDB_API_KEY from .env file
    assert "WANDB_API_KEY" in environ, '"WANDB_API_KEY" is empty. Create ".env" file with your W&B API key. See ".env.sample" for the file format'

    wandb.init(project=project, group=group, config=config)
