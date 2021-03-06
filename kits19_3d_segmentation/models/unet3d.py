# Borrowed a large part of the code from:
# https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/model.py

import torch.nn as nn

from .initializations import get_initializer
from .unet3d_blocks import create_decoder_blocks, create_encoder_blocks, get_feature_channels_per_level


class UNet3D(nn.Module):
    """3D U-Net base class.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_feature_channels,
                 max_feature_channels,
                 base_module,
                 num_levels,
                 normalization,
                 nonlinearity,
                 conv_kernel_size=(3, 3, 3),
                 padding_width=(1, 1, 1),
                 first_downsample_stride=(2, 2, 2),
                 decoder_min_channels=None,
                 initializer='kaiming_normal'):
        """

        Args:
            in_channels (int): number of the input channels of U-Net.
            out_channels (int): number of the output channels of U-Net.
            base_feature_channels (int): number of the output channels of the first block in the encoder part.
            max_feature_channels (int): max output channels of the encoder/decoder blocks.
            base_module (str): name of the base module making up the block. Currently only 'double_conv' is supported.
            num_levels (int): number of the levels (= blocks) in the encoder/decoder part.
            normalization (str): name of the normalization layer to use. 'batch_norm' or 'instance_norm'.
            nonlinearity (str): name of the non-linearity layer to use. 'relu' or 'leaky_relu'.
            conv_kernel_size (int or tuple[int]): kernel size of the 3D convolution layers. Defaults to (3, 3, 3).
            padding_width (int or tuple[int]): padding width of the 3D convolution layers. Defaults to (1, 1, 1).
            first_downsample_stride (int or tuple[int]): stride of the first strided 3D convolution layer (= the first 3D convolution layer of the second encoder block). Defaults to (2, 2, 2).
            decoder_min_channels (int, optional): min output channels of the decoder blocks. Defaults to None.
            initializer (str, optional): name of the weight initializer. Defaults to 'kaiming_normal'.

        Raises:
            ValueError:
        """
        super(UNet3D, self).__init__()

        assert num_levels > 1, "U-Net requires at least 2 levels."

        encoder_feature_channels, decoder_feature_channels = get_feature_channels_per_level(
            base_feature_channels, max_feature_channels, num_levels, decoder_min_channels)

        self.encoder_blocks = create_encoder_blocks(in_channels, encoder_feature_channels, base_module,
                                                    conv_kernel_size, padding_width, first_downsample_stride,
                                                    normalization, nonlinearity)
        self.decoder_blocks = create_decoder_blocks(encoder_feature_channels, decoder_feature_channels, base_module,
                                                    conv_kernel_size, padding_width, first_downsample_stride,
                                                    normalization, nonlinearity)

        self.final_conv = nn.Conv3d(decoder_feature_channels[-1], out_channels, 1)

        # init weights
        initializer = get_initializer(initializer)
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        # forward encoder
        encoder_outputs = []
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            if i != len(self.encoder_blocks) - 1:
                encoder_outputs.append(x)

        # forward decoder
        for decoder_block in self.decoder_blocks:
            x = decoder_block(encoder_outputs[-1], x)
            del encoder_outputs[-1]

        # head
        x = self.final_conv(x)
        # XXX: final activation (e.g., sigmoid, softmax) has to be applied outside.
        # This is because some PyTorch loss modules are integrated with activation operations to stabilize backprop computation.

        return x


class PlainUNet3D(UNet3D):
    """Plain 3D U-Net proposed in the paper “An Attempt at Beating the 3D U-Net” (arXiv: http://arxiv.org/abs/1908.02182).
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=3,
                 base_feature_channels=30,
                 max_feature_channels=320,
                 base_module='double_conv',
                 num_levels=6,
                 normalization='instance_norm',
                 nonlinearity='leaky_relu',
                 conv_kernel_size=(3, 3, 3),
                 padding_width=(1, 1, 1),
                 first_downsample_stride=(2, 2, 1),
                 decoder_min_channels=None,
                 initializer='kaiming_normal'):
        super(PlainUNet3D, self).__init__(in_channels, out_channels, base_feature_channels, max_feature_channels,
                                          base_module, num_levels, normalization, nonlinearity, conv_kernel_size,
                                          padding_width, first_downsample_stride, decoder_min_channels, initializer)
