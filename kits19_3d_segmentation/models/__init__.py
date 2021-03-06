import torch

from .unet3d import PlainUNet3D


def get_model(config):
    """Get model loaded on the device ('cpu' or 'cuda') and with pre-trained weights if specified.

    Args:
        config (YACS CfgNode): config.

    Raises:
        ValueError:

    Returns:
        torch.nn.Module: model loaded on the device ('cpu' or 'cuda') and with pre-trained weights if specified.
    """
    model_name = config.MODEL.NAME

    if model_name == 'plane_unet_3d':
        model = PlainUNet3D(in_channels=config.MODEL.INPUT_CHANNELS,
                            out_channels=config.MODEL.OUTPUT_CHANNELS,
                            base_feature_channels=config.MODEL.BASE_FEATURE_CHANNELS,
                            max_feature_channels=config.MODEL.MAX_FEATURE_CHANNELS,
                            base_module=config.MODEL.BASE_MODULE,
                            num_levels=config.MODEL.NUM_LEVELS,
                            normalization=config.MODEL.NORMALIZATION,
                            nonlinearity=config.MODEL.NON_LINEARITY,
                            conv_kernel_size=config.MODEL.CONV_KERNEL_SIZE,
                            padding_width=config.MODEL.PADDING_WIDTH,
                            first_downsample_stride=config.MODEL.FIRST_DOWNSAMPLE_STRIDE,
                            initializer=config.MODEL.INITIALIZER)
    else:
        raise ValueError()

    if config.MODEL.WEIGHT and config.MODEL.WEIGHT != 'none':
        # load weight from file
        model.load_state_dict(torch.load(config.MODEL.WEIGHT, map_location=torch.device('cpu')))

    return model.to(config.MODEL.DEVICE)
