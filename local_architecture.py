from typing import Sequence

import torch
from torch import nn


def _as_tuple(value):
    if isinstance(value, int):
        return (value, value, value)
    return tuple(value)


class ConvDropoutNormReLU(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        conv_bias=True,
        norm_eps=1e-5,
        norm_affine=True,
        with_nonlin=True,
    ):
        super().__init__()
        kernel_size = _as_tuple(kernel_size)
        stride = _as_tuple(stride)
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding, bias=conv_bias)
        self.norm = nn.InstanceNorm3d(output_channels, eps=norm_eps, affine=norm_affine)
        modules = [self.conv, self.norm]
        if with_nonlin:
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
            modules.append(self.nonlin)
        self.all_modules = nn.Sequential(*modules)

    def forward(self, x):
        return self.all_modules(x)


class StackedConvBlocks(nn.Module):
    def __init__(
        self,
        num_convs,
        input_channels,
        output_channels,
        kernel_size,
        initial_stride,
        conv_bias=True,
        norm_eps=1e-5,
        norm_affine=True,
    ):
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs
        blocks = [
            ConvDropoutNormReLU(
                input_channels,
                output_channels[0],
                kernel_size,
                initial_stride,
                conv_bias=conv_bias,
                norm_eps=norm_eps,
                norm_affine=norm_affine,
                with_nonlin=True,
            )
        ]
        for idx in range(1, num_convs):
            blocks.append(
                ConvDropoutNormReLU(
                    output_channels[idx - 1],
                    output_channels[idx],
                    kernel_size,
                    1,
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                    norm_affine=norm_affine,
                    with_nonlin=True,
                )
            )
        self.convs = nn.Sequential(*blocks)

    def forward(self, x):
        return self.convs(x)


class BasicBlockD(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        conv_bias=True,
        norm_eps=1e-5,
        norm_affine=True,
    ):
        super().__init__()
        stride = _as_tuple(stride)
        self.conv1 = ConvDropoutNormReLU(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            norm_affine=norm_affine,
            with_nonlin=True,
        )
        self.conv2 = ConvDropoutNormReLU(
            output_channels,
            output_channels,
            kernel_size,
            1,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            norm_affine=norm_affine,
            with_nonlin=False,
        )
        self.nonlin2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        has_stride = any(s != 1 for s in stride)
        requires_projection = input_channels != output_channels
        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(nn.AvgPool3d(kernel_size=stride, stride=stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(
                        input_channels,
                        output_channels,
                        1,
                        1,
                        conv_bias=False,
                        norm_eps=norm_eps,
                        norm_affine=norm_affine,
                        with_nonlin=False,
                    )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.nonlin2(self.conv2(self.conv1(x)) + self.skip(x))


class StackedResidualBlocks(nn.Module):
    def __init__(
        self,
        n_blocks,
        input_channels,
        output_channels,
        kernel_size,
        initial_stride,
        conv_bias=True,
        norm_eps=1e-5,
        norm_affine=True,
    ):
        super().__init__()
        blocks = [
            BasicBlockD(
                input_channels,
                output_channels,
                kernel_size,
                initial_stride,
                conv_bias=conv_bias,
                norm_eps=norm_eps,
                norm_affine=norm_affine,
            )
        ]
        for _ in range(1, n_blocks):
            blocks.append(
                BasicBlockD(
                    output_channels,
                    output_channels,
                    kernel_size,
                    1,
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                    norm_affine=norm_affine,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResidualEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        features_per_stage,
        kernel_sizes,
        strides,
        n_blocks_per_stage,
        conv_bias=True,
        norm_eps=1e-5,
        norm_affine=True,
    ):
        super().__init__()
        self.stem = StackedConvBlocks(
            1,
            input_channels,
            features_per_stage[0],
            kernel_sizes[0],
            1,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            norm_affine=norm_affine,
        )
        input_channels = features_per_stage[0]
        stages = []
        for idx, output_channels in enumerate(features_per_stage):
            stages.append(
                StackedResidualBlocks(
                    n_blocks_per_stage[idx],
                    input_channels,
                    output_channels,
                    kernel_sizes[idx],
                    strides[idx],
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                    norm_affine=norm_affine,
                )
            )
            input_channels = output_channels
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class UNetDecoder(nn.Module):
    def __init__(
        self,
        features_per_stage: Sequence[int],
        strides: Sequence[Sequence[int]],
        num_classes: int,
        n_conv_per_stage_decoder: Sequence[int],
        conv_bias=True,
        norm_eps=1e-5,
        norm_affine=True,
        deep_supervision=False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = nn.Identity()
        encoder_channels = list(features_per_stage)
        decoder_channels = list(features_per_stage[:-1][::-1])
        bottleneck_channels = features_per_stage[-1]
        transpose_strides = list(strides[1:][::-1])

        self.stages = nn.ModuleList()
        self.transpconvs = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        input_channels = bottleneck_channels
        for idx, output_channels in enumerate(decoder_channels):
            self.transpconvs.append(
                nn.ConvTranspose3d(
                    input_channels,
                    output_channels,
                    kernel_size=_as_tuple(transpose_strides[idx]),
                    stride=_as_tuple(transpose_strides[idx]),
                )
            )
            self.stages.append(
                StackedConvBlocks(
                    n_conv_per_stage_decoder[idx],
                    output_channels + encoder_channels[-(idx + 2)],
                    output_channels,
                    3,
                    1,
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                    norm_affine=norm_affine,
                )
            )
            self.seg_layers.append(nn.Conv3d(output_channels, num_classes, 1, 1, 0))
            input_channels = output_channels

    def forward(self, skips):
        x = skips[-1]
        seg_outputs = []
        for idx, stage in enumerate(self.stages):
            x = self.transpconvs[idx](x)
            x = torch.cat((x, skips[-(idx + 2)]), dim=1)
            x = stage(x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[idx](x))
            elif idx == len(self.stages) - 1:
                seg_outputs.append(self.seg_layers[-1](x))
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


class ResidualEncoderUNet(nn.Module):
    def __init__(
        self,
        input_channels,
        features_per_stage,
        kernel_sizes,
        strides,
        n_blocks_per_stage,
        num_classes,
        n_conv_per_stage_decoder,
        conv_bias=True,
        norm_eps=1e-5,
        norm_affine=True,
        deep_supervision=False,
    ):
        super().__init__()
        self.encoder = ResidualEncoder(
            input_channels,
            features_per_stage,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            norm_affine=norm_affine,
        )
        self.decoder = UNetDecoder(
            features_per_stage,
            strides,
            num_classes,
            n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            norm_affine=norm_affine,
            deep_supervision=deep_supervision,
        )
        self.decoder.encoder = self.encoder

    def forward(self, x):
        return self.decoder(self.encoder(x))
