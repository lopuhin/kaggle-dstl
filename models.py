import json
from functools import partial
from pathlib import Path

import attr
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F


@attr.s(slots=True)
class HyperParams:
    classes = attr.ib(default=list(range(10)))
    net = attr.ib(default='UNet')
    n_channels = attr.ib(default=12)  # max 20
    total_classes = 10
    thresholds = attr.ib(default=[0.5])

    patch_inner = attr.ib(default=64)
    patch_border = attr.ib(default=16)

    augment_rotations = attr.ib(default=10.0)  # degrees
    augment_flips = attr.ib(default=0)
    augment_channels = attr.ib(default=0.0)

    validation_square = attr.ib(default=400)

    dropout = attr.ib(default=0.0)
    bn = attr.ib(default=1)
    activation = attr.ib(default='relu')
    top_scale = attr.ib(default=2)
    log_loss = attr.ib(default=1.0)
    dice_loss = attr.ib(default=0.0)
    jaccard_loss = attr.ib(default=0.0)
    dist_loss = attr.ib(default=0.0)
    dist_dice_loss = attr.ib(default=0.0)
    dist_jaccard_loss = attr.ib(default=0.0)

    filters_base = attr.ib(default=32)

    n_epochs = attr.ib(default=100)
    oversample = attr.ib(default=0.0)
    lr = attr.ib(default=0.0001)
    lr_decay = attr.ib(default=0.0)
    weight_decay = attr.ib(default=0.0)
    batch_size = attr.ib(default=128)

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def has_all_classes(self):
        return self.n_classes == self.total_classes

    @property
    def needs_dist(self):
        return (self.dist_loss != 0 or self.dist_dice_loss != 0 or
                self.dist_jaccard_loss != 0)

    @classmethod
    def from_dir(cls, root: Path):
        params = json.loads(root.joinpath('hps.json').read_text())
        fields = {field.name for field in attr.fields(HyperParams)}
        return cls(**{k: v for k, v in params.items() if k in fields})

    def update(self, hps_string: str):
        if hps_string:
            values = dict(pair.split('=') for pair in hps_string.split(','))
            for field in attr.fields(HyperParams):
                v = values.pop(field.name, None)
                if v is not None:
                    default = field.default
                    assert not isinstance(default, bool)
                    if isinstance(default, (int, float, str)):
                        v = type(default)(v)
                    elif isinstance(default, list):
                        v = [type(default[0])(x) for x in v.split('-')]
                    setattr(self, field.name, v)
            if values:
                raise ValueError('Unknown hyperparams: {}'.format(values))


class BaseNet(nn.Module):
    def __init__(self, hps: HyperParams):
        super().__init__()
        self.hps = hps
        if hps.dropout:
            self.dropout2d = nn.Dropout2d(p=hps.dropout)
        else:
            self.dropout2d = lambda x: x
        self.register_buffer('global_step', torch.IntTensor(1).zero_())


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


def concat(xs):
    return torch.cat(xs, 1)


class MiniNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 4, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, hps.n_classes, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, :, b:-b, b:-b])


class OldNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, hps.n_classes, 7, padding=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, :, b:-b, b:-b])


class SmallNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, hps.n_classes, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, :, b:-b, b:-b])


# UNet:
# http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


class SmallUNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, hps.n_classes, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = self.pool(x)
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = F.relu(self.conv5(x1))
        x1 = self.upsample(x1)
        x = torch.cat([x, x1], 1)
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, :, b:-b, b:-b])


class UNetModule(nn.Module):
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        self.conv1 = conv3x3(in_, out)
        self.conv2 = conv3x3(out, out)
        self.bn = hps.bn
        self.activation = getattr(F, hps.activation)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(out)
            self.bn2 = nn.BatchNorm2d(out)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        x = self.activation(x)
        return x


class UNet(BaseNet):
    module = UNetModule
    filter_factors = [1, 2, 4, 8, 16]

    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_top = nn.MaxPool2d(hps.top_scale, hps.top_scale)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample_top = nn.UpsamplingNearest2d(scale_factor=hps.top_scale)
        filter_sizes = [hps.filters_base * s for s in self.filter_factors]
        self.down, self.up = [], []
        for i, nf in enumerate(filter_sizes):
            low_nf = hps.n_channels if i == 0 else filter_sizes[i - 1]
            self.down.append(self.module(hps, low_nf, nf))
            setattr(self, 'down_{}'.format(i), self.down[-1])
            if i != 0:
                self.up.append(self.module(hps, low_nf + nf, low_nf))
                setattr(self, 'conv_up_{}'.format(i), self.up[-1])
        self.conv_final = nn.Conv2d(filter_sizes[0], hps.n_classes, 1)

    def forward(self, x):
        xs = []
        for i, down in enumerate(self.down):
            if i == 0:
                x_in = x
            elif i == 1:
                x_in = self.pool_top(xs[-1])
            else:
                x_in = self.pool(xs[-1])
            x_out = down(x_in)
            x_out = self.dropout2d(x_out)
            xs.append(x_out)

        x_out = xs[-1]
        for i, (x_skip, up) in reversed(list(enumerate(zip(xs[:-1], self.up)))):
            upsample = self.upsample_top if i == 0 else self.upsample
            x_out = up(torch.cat([upsample(x_out), x_skip], 1))
            x_out = self.dropout2d(x_out)

        x_out = self.conv_final(x_out)
        b = self.hps.patch_border
        return F.sigmoid(x_out[:, :, b:-b, b:-b])


class Conv3BN(nn.Module):
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if hps.bn else None
        self.activation = getattr(F, hps.activation)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x, inplace=True)
        return x


class UNet3lModule(nn.Module):
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(hps, in_, out)
        self.l2 = Conv3BN(hps, out, out)
        self.l3 = Conv3BN(hps, out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


class UNet3l(UNet):
    module = UNet3lModule


class UNet2Module(nn.Module):
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(hps, in_, out)
        self.l2 = Conv3BN(hps, out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class UNet2(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        b = hps.filters_base
        self.filters = [b * 2, b * 2, b * 4, b * 8, b * 16]
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.down, self.down_pool, self.mid, self.up = [[] for _ in range(4)]
        for i, nf in enumerate(self.filters):
            low_nf = hps.n_channels if i == 0 else self.filters[i - 1]
            self.down_pool.append(
                nn.Conv2d(low_nf, low_nf, 3, padding=1, stride=2))
            setattr(self, 'down_pool_{}'.format(i), self.down_pool[-1])
            self.down.append(UNet2Module(hps, low_nf, nf))
            setattr(self, 'down_{}'.format(i), self.down[-1])
            if i != 0:
                self.mid.append(Conv3BN(hps, low_nf, low_nf))
                setattr(self, 'mid_{}'.format(i), self.mid[-1])
                self.up.append(UNet2Module(hps, low_nf + nf, low_nf))
                setattr(self, 'up_{}'.format(i), self.up[-1])
        self.conv_final = nn.Conv2d(self.filters[0], hps.n_classes, 1)

    def forward(self, x):
        xs = []
        for i, (down, down_pool) in enumerate(zip(self.down, self.down_pool)):
            x_out = down(down_pool(xs[-1]) if xs else x)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, up, mid in reversed(list(zip(xs[:-1], self.up, self.mid))):
            x_out = up(torch.cat([self.upsample(x_out), mid(x_skip)], 1))

        x_out = self.conv_final(x_out)
        b = self.hps.patch_border
        return F.sigmoid(x_out[:, :, b:-b, b:-b])


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        out_1 = out * 3 // 8
        out_2 = out * 2 // 8
        self.conv1x1 = BasicConv2d(in_, out_1, kernel_size=1)
        self.conv3x3_pre = BasicConv2d(in_, in_ // 2, kernel_size=1)
        self.conv3x3 = BasicConv2d(in_ // 2, out_1, kernel_size=3, padding=1)
        self.conv5x5_pre = BasicConv2d(in_, in_ // 4, kernel_size=1)
        self.conv5x5 = BasicConv2d(in_ // 4, out_2, kernel_size=5, padding=2)
        assert hps.bn
        assert hps.activation == 'relu'

    def forward(self, x):
        return torch.cat([
            self.conv1x1(x),
            self.conv3x3(self.conv3x3_pre(x)),
            self.conv5x5(self.conv5x5_pre(x)),
        ], 1)


class Inception2Module(nn.Module):
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        self.l1 = InceptionModule(hps, in_, out)
        self.l2 = InceptionModule(hps, out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class InceptionUNet(UNet):
    module = InceptionModule


class Inception2UNet(UNet):
    module = Inception2Module


class SimpleSegNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        s = hps.filters_base
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.input_conv = BasicConv2d(hps.n_channels, s, 1)
        self.enc_1 = BasicConv2d(s * 1, s * 2, 3, padding=1)
        self.enc_2 = BasicConv2d(s * 2, s * 4, 3, padding=1)
        self.enc_3 = BasicConv2d(s * 4, s * 8, 3, padding=1)
        self.enc_4 = BasicConv2d(s * 8, s * 8, 3, padding=1)
        # https://github.com/pradyu1993/segnet - decoder lacks relu (???)
        self.dec_4 = BasicConv2d(s * 8, s * 8, 3, padding=1)
        self.dec_3 = BasicConv2d(s * 8, s * 4, 3, padding=1)
        self.dec_2 = BasicConv2d(s * 4, s * 2, 3, padding=1)
        self.dec_1 = BasicConv2d(s * 2, s * 1, 3, padding=1)
        self.conv_final = nn.Conv2d(s, hps.n_classes, 1)

    def forward(self, x):
        # Input
        x = self.input_conv(x)
        # Encoder
        x = self.enc_1(x)
        x = self.pool(x)
        x = self.enc_2(x)
        x = self.pool(x)
        x = self.enc_3(x)
        x = self.pool(x)
        x = self.enc_4(x)
        # Decoder
        x = self.dec_4(x)
        x = self.upsample(x)
        x = self.dec_3(x)
        x = self.upsample(x)
        x = self.dec_2(x)
        x = self.upsample(x)
        x = self.dec_1(x)
        # Output
        x = self.conv_final(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, :, b:-b, b:-b])


class DenseLayer(nn.Module):
    def __init__(self, in_, out, *, dropout, bn):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_) if bn else None
        self.activation = nn.ReLU(inplace=True)
        self.conv = conv3x3(in_, out)
        self.dropout = nn.Dropout2d(p=dropout) if dropout else None

    def forward(self, x):
        x = self.activation(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_, k, n_layers, dropout, bn):
        super().__init__()
        self.out = k * n_layers
        layer_in = in_
        self.layers = []
        for i in range(n_layers):
            layer = DenseLayer(layer_in, k, dropout=dropout, bn=bn)
            self.layers.append(layer)
            setattr(self, 'layer_{}'.format(i), layer)
            layer_in += k

    def forward(self, x):
        inputs = [x]
        outputs = []
        for i, layer in enumerate(self.layers[:-1]):
            outputs.append(layer(inputs[i]))
            inputs.append(concat([outputs[i], inputs[i]]))
        return torch.cat([self.layers[-1](inputs[-1])] + outputs, 1)


class DenseUNetModule(DenseBlock):
    def __init__(self, hps: HyperParams, in_: int, out: int):
        n_layers = 4
        super().__init__(in_, out // n_layers, n_layers,
                         dropout=hps.dropout, bn=hps.bn)


class DenseUNet(UNet):
    module = DenseUNetModule


class DownBlock(nn.Module):
    def __init__(self, in_, out, scale, *, dropout, bn):
        super().__init__()
        self.in_ = in_
        self.bn = nn.BatchNorm2d(in_) if bn else None
        self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_, out, 1)
        self.dropout = nn.Dropout2d(p=dropout) if dropout else None
        self.pool = nn.MaxPool2d(scale, scale)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_, out, scale):
        super().__init__()
        self.up_conv = nn.Conv2d(in_, out, 1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale)

    def forward(self, x):
        return self.upsample(self.up_conv(x))


class DenseNet(BaseNet):
    """ https://arxiv.org/pdf/1611.09326v2.pdf
    """
    def __init__(self, hps):
        super().__init__(hps)
        k = hps.filters_base
        block_layers = [3, 5, 7, 5, 3]
        block_in = [n * k for n in [3, 8, 16, 8, 4]]
        scale_factors = [4, 2]
        dense = partial(DenseBlock, dropout=hps.dropout, bn=hps.bn)
        self.input_conv = nn.Conv2d(hps.n_channels, block_in[0], 3, padding=1)
        self.blocks = []
        self.scales = []
        self.n_layers = len(block_layers) // 2
        for i, (in_, l) in enumerate(zip(block_in, block_layers)):
            if i < self.n_layers:
                block = dense(in_, k, l)
                scale = DownBlock(block.out + in_, block_in[i + 1],
                                  scale_factors[i],
                                  dropout=hps.dropout, bn=hps.bn)
            elif i == self.n_layers:
                block = dense(in_, k, l)
                scale = None
            else:
                block = dense(in_ + self.scales[2 * self.n_layers - i].in_,
                              k, l)
                scale = UpBlock(self.blocks[-1].out, in_,
                                scale_factors[2 * self.n_layers - i])
            setattr(self, 'block_{}'.format(i), block)
            setattr(self, 'scale_{}'.format(i), scale)
            self.blocks.append(block)
            self.scales.append(scale)
        self.output_conv = nn.Conv2d(self.blocks[-1].out, hps.n_classes, 1)

    def forward(self, x):
        # Input
        x = self.input_conv(x)
        # Network
        skips = []
        for i, (block, scale) in enumerate(zip(self.blocks, self.scales)):
            if i < self.n_layers:
                x = concat([block(x), x])
                skips.append(x)
                x = scale(x)
            elif i == self.n_layers:
                x = block(x)
            else:
                x = block(concat([scale(x), skips[2 * self.n_layers - i]]))
        # Output
        x = self.output_conv(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, :, b:-b, b:-b])
