# Borrowed a large part of the code from:
# https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/initialization.py

import torch.nn as nn


def get_initializer(initializer):
    """Get weight initializer.

    Args:
        initializer (str): name of the initializer to get. Currently only 'kaiming_normal' is supported.

    Raises:
        ValueError:

    Returns:
        Callable or None: weight initializer.
    """
    if initializer:
        if initializer == 'kaiming_normal':
            return KaimingNormal()
        else:
            raise ValueError()
    else:
        return None


class KaimingNormal:
    """Callable to init weights of `nn.Conv3d` and `nn.ConvTranspose3d` with Kaiming normal method.
    """
    def __init__(self, negative_slope=1e-2):
        """

        Args:
            negative_slope (float, optional): negative slope for 'leaky_relu' mode. Defaults to 1e-2.
        """
        # negative_slope of LeakyReLU is 1e-2 at default
        self.negative_slope = negative_slope

    def __call__(self, module):
        """

        Args:
            module (torch.nn.Module): network module to init weights.
        """
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(module.weight, a=self.negative_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
