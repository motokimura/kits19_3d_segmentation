# Borrowed a large part of the code from:
# https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/buildingblocks.py

import torch
import torch.nn as nn


def get_feature_channels_per_level(base_feature_channels, max_feature_channels, num_levels, decoder_min_channels=None):
    """Get the number of output channels for each of the encoder and decoder blocks.

    Args:
        base_feature_channels (int): number of the output channels of the first block in the encoder part.
            Number of the output channels of the encoder blocks is given by the geometric progression: `base_feature_channels` ** `k` (`k`: level of the encoder block).
        max_feature_channels (int): max output channels of the encoder/decoder blocks.
        num_levels (int): number of the levels (= blocks) in the encoder/decoder part.
        decoder_min_channels (int, optional): min output channels of the decoder blocks. Defaults to None.

    Returns:
        tuple[list[int]]: encoder output channels and decoder output channels.
    """
    # output channels of the encoder blocks
    encoder_feature_channels = [min(max_feature_channels, base_feature_channels * (2**i)) for i in range(num_levels)]

    # output channels of the decoder blocks
    decoder_feature_channels = list(reversed(encoder_feature_channels))[1:]
    if decoder_min_channels is not None:
        for i in range(len(decoder_feature_channels)):
            decoder_feature_channels[i] = max(decoder_min_channels, decoder_feature_channels[i])

    return encoder_feature_channels, decoder_feature_channels


def create_encoder_blocks(in_channels, encoder_feature_channels, base_module, conv_kernel_size, padding_width,
                          first_downsample_stride, normalization, nonlinearity):
    """Get U-Net encoder blocks as torch.nn.ModuleList.

    Args:
        in_channels (int): number of the input channels of U-Net.
        encoder_feature_channels (list[int]): number of the output channels of each encoder block.
        base_module (str): name of the base module making up the block. Currently only 'double_conv' is supported.
        conv_kernel_size (int or tuple[int]): kernel size of the 3D convolution layers.
        padding_width (int or tuple[int]): padding width of the 3D convolution layers.
        first_downsample_stride (int or tuple[int]): stride of the first strided 3D convolution layer (= the first 3D convolution layer of the second encoder block).
        normalization (str): name of the normalization layer to use. 'batch_norm' or 'instance_norm'.
        nonlinearity (str): name of the non-linearity layer to use. 'relu' or 'leaky_relu'.

    Returns:
        torch.nn.ModuleList: list of the block for each level of the U-Net encoder part.
    """
    encoder_blocks = []
    for i, out_ch in enumerate(encoder_feature_channels):
        # encoder input channels
        in_ch = in_channels if i == 0 else encoder_feature_channels[i - 1]

        # stride of the first conv layer of the encoder block
        if i == 0:
            first_conv_stride = 1  # do not apply down sampling in the first encoder block
        elif i == 1:
            first_conv_stride = first_downsample_stride
        else:
            first_conv_stride = 2

        encoder_blocks.append(
            EncoderBlock(in_ch, out_ch, base_module, conv_kernel_size, padding_width, first_conv_stride, normalization,
                         nonlinearity))

    return nn.ModuleList(encoder_blocks)


def create_decoder_blocks(encoder_feature_channels,
                          decoder_feature_channels,
                          base_module,
                          conv_kernel_size,
                          padding_width,
                          last_upsample_stride,
                          normalization,
                          nonlinearity,
                          decoder_min_channels=None):
    """Get U-Net decoder blocks as torch.nn.ModuleList.

    Args:
        encoder_feature_channels (list[int]): number of the output channels of each encoder block.
        decoder_feature_channels (list[int]): number of the output channels of each decoder block.
        base_module (str): name of the base module making up the block. Currently only 'double_conv' is supported.
        conv_kernel_size (int or tuple[int]): kernel size of the 3D convolution layers.
        padding_width (int or tuple[int]): padding width of the 3D convolution layers.
        last_upsample_stride (int or tuple[int]): scale factor for upsampling in the last decoder block.
        normalization (str): name of the normalization layer to use. 'batch_norm' or 'instance_norm'.
        nonlinearity (str): name of the non-linearity layer to use. 'relu' or 'leaky_relu'.
        decoder_min_channels (int, optional): min output channels of the decoder blocks. Defaults to None.

    Returns:
        torch.nn.ModuleList: list of the block for each level of the U-Net encoder part.
    """

    decoder_blocks = []
    skip_feature_channels = list(reversed(encoder_feature_channels))[1:]
    for i, (out_ch, skip_ch) in enumerate(zip(decoder_feature_channels, skip_feature_channels)):
        # channel of the feature from bottom decoder block
        in_ch = encoder_feature_channels[-1] if i == 0 else decoder_feature_channels[i - 1]

        if i == len(decoder_feature_channels) - 1:
            # for the last decoder block
            upsample_stride = last_upsample_stride
        else:
            upsample_stride = 2

        decoder_blocks.append(
            DecoderBlock(in_ch, out_ch, skip_ch, base_module, conv_kernel_size, padding_width, upsample_stride,
                         normalization, nonlinearity))

    return nn.ModuleList(decoder_blocks)


class EncoderBlock(nn.Module):
    """Layers that make up one level of the U-Net encoder part.
    """
    def __init__(self, in_channels, out_channels, base_module, conv_kernel_size, padding_width, first_conv_stride,
                 normalization, nonlinearity):
        """

        Args:
            in_channels (int): number of the input channels.
            out_channels (int): number of the output channels.
            base_module (str): name of the base module making up the block. Currently only 'double_conv' is supported.
            conv_kernel_size (int or tuple[int]): kernel size of the 3D convolution layers.
            padding_width (int or tuple[int]): padding width of the 3D convolution layers.
            first_conv_stride (int or tuple[int]): stride of the first 3D convolution layer.
            normalization (str): name of the normalization layer to use. 'batch_norm' or 'instance_norm'.
            nonlinearity (str): name of the non-linearity layer to use. 'relu' or 'leaky_relu'.

        Raises:
            ValueError:
        """
        super(EncoderBlock, self).__init__()

        # prepare basic module
        conv1_in_channels = in_channels
        conv1_out_channels = out_channels
        conv2_in_channels = out_channels
        conv2_out_channels = out_channels

        if base_module == 'double_conv':
            self.base_module = DoubleConv(conv1_in_channels, conv1_out_channels, conv2_in_channels, conv2_out_channels,
                                          conv_kernel_size, padding_width, first_conv_stride, normalization,
                                          nonlinearity)
        else:
            raise ValueError()

    def forward(self, x):
        return self.base_module(x)


class DecoderBlock(nn.Module):
    """Layers that make up one level of the U-Net decoder part.
    """
    def __init__(self, in_channels, out_channels, skip_channels, base_module, conv_kernel_size, padding_width,
                 upsample_stride, normalization, nonlinearity):
        """

        Args:
            in_channels (int): number of the input channels.
            out_channels (int): number of the output channels.
            skip_channels (int): number of the feature channels skipped from the encoder block.
            base_module (str): name of the base module making up the block. Currently only 'double_conv' is supported.
            conv_kernel_size (int or tuple[int]): kernel size of the 3D convolution layers.
            padding_width (int or tuple[int]): padding width of the 3D convolution layers.
            upsample_stride (int or tuple[int]): scale factor for upsampling.
            normalization (str): name of the normalization layer to use. 'batch_norm' or 'instance_norm'.
            nonlinearity (str): name of the non-linearity layer to use. 'relu' or 'leaky_relu'.

        Raises:
            ValueError:
        """
        super(DecoderBlock, self).__init__()

        upsample_out_channels = skip_channels  # channels of the upsampled feature

        # prepare upsample module
        # TODO: need to investivate if `bias=False` affects the accuracy
        self.upsample = nn.ConvTranspose3d(in_channels,
                                           upsample_out_channels,
                                           kernel_size=upsample_stride,
                                           stride=upsample_stride,
                                           bias=False)

        # prepare basic module
        conv1_in_channels = skip_channels + upsample_out_channels  # = 2 * skip_channels
        conv1_out_channels = out_channels
        conv2_in_channels = out_channels
        conv2_out_channels = out_channels
        first_conv_stride = 1

        if base_module == 'double_conv':
            self.base_module = DoubleConv(conv1_in_channels, conv1_out_channels, conv2_in_channels, conv2_out_channels,
                                          conv_kernel_size, padding_width, first_conv_stride, normalization,
                                          nonlinearity)
        else:
            raise ValueError()

    def forward(self, encoder_features, x):
        x = torch.cat([encoder_features, self.upsample(x)], dim=1)
        return self.base_module(x)


class DoubleConv(nn.Sequential):
    """Chain of 2 `ConvBlock` modules.
    """
    def __init__(self, conv1_in_channels, conv1_out_channels, conv2_in_channels, conv2_out_channels, conv_kernel_size,
                 padding_width, first_conv_stride, normalization, nonlinearity):
        """

        Args:
            conv1_in_channels (int): number of the input channels of the first 3D convolution layer.
            conv1_out_channels (int): number of the output channels of the first 3D convolution layer.
            conv2_in_channels (int): number of the input channels of the second 3D convolution layer.
            conv2_out_channels (int): number of the output channels of the second 3D convolution layer.
            conv_kernel_size (int or tuple[int]): kernel size of the 3D convolution layers.
            padding_width (int or tuple[int]): padding width of the 3D convolution layers.
            first_conv_stride (int or tuple[int]): stride of the first 3D convolution layer.
            normalization (str): name of the normalization layer to use. 'batch_norm' or 'instance_norm'.
            nonlinearity (str): name of the non-linearity layer to use. 'relu' or 'leaky_relu'.
        """
        super(DoubleConv, self).__init__()

        # conv1 layer
        self.add_module(
            'conv1',
            ConvBlock(conv1_in_channels,
                      conv1_out_channels,
                      conv_kernel_size,
                      padding_width,
                      conv_stride=first_conv_stride,
                      normalization=normalization,
                      nonlinearity=nonlinearity))

        # conv2 layer
        self.add_module(
            'conv2',
            ConvBlock(conv2_in_channels,
                      conv2_out_channels,
                      conv_kernel_size,
                      padding_width,
                      conv_stride=1,
                      normalization=normalization,
                      nonlinearity=nonlinearity))


class ConvBlock(nn.Module):
    """Chain of 3D convolution + normalization + non-linearity layers.
    """
    def __init__(self, in_channels, out_channels, conv_kernel_size, padding_width, conv_stride, normalization,
                 nonlinearity):
        """

        Args:
            in_channels (int): number of the input channels.
            out_channels (int): number of the output channels.
            conv_kernel_size (int or tuple[int]): kernel size of the 3D convolution layer.
            padding_width (int or tuple[int]): padding width of the 3D convolution layer.
            conv_stride (int or tuple[int]): stride of the 3D convolution layer.
            normalization (str): name of the normalization layer to use. 'batch_norm' or 'instance_norm'.
            nonlinearity (str): name of the non-linearity layer to use. 'relu' or 'leaky_relu'.

        Raises:
            ValueError:
            ValueError:
        """
        super(ConvBlock, self).__init__()

        # non-linearity
        if nonlinearity:
            if nonlinearity == 'relu':
                self.nonlinearity = nn.ReLU(inplace=True)
            elif nonlinearity == 'leaky_relu':
                self.nonlinearity = nn.LeakyReLU(inplace=True)
            else:
                raise ValueError()
        else:
            self.nonlinearity = None

        # normalization
        if normalization:
            if normalization == 'batch_norm':
                self.normalization = nn.BatchNorm3d(out_channels)
            elif normalization == 'instance_norm':
                self.normalization = nn.InstanceNorm3d(out_channels, affine=True)
            else:
                raise ValueError()
        else:
            self.normalization = None

        # convolution
        bias = True if self.normalization is None else False  # add learnable bias when normalization layer is absent
        self.conv = nn.Conv3d(in_channels, out_channels, conv_kernel_size, conv_stride, padding_width, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        return self.nonlinearity(x)
