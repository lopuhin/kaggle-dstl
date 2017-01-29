import attr
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F


@attr.s(slots=True)
class HyperParams:
    classes = attr.ib(default=list(range(10)))
    net = attr.ib(default='DefaultNet')
    n_channels = attr.ib(default=20)
    total_classes = 10
    thresholds = attr.ib(default=[0.2, 0.3, 0.4, 0.5, 0.6])

    patch_inner = attr.ib(default=64)
    patch_border = attr.ib(default=16)

    dropout_keep_prob = attr.ib(default=0.0)  # TODO
    jaccard_loss = attr.ib(default=0)

    n_epochs = attr.ib(default=30)
    oversample = attr.ib(default=0.0)
    lr = attr.ib(default=0.0001)
    batch_size = attr.ib(default=128)

    @property
    def n_classes(self):
        return len(self.classes)

    def update(self, hps_string: str):
        if hps_string:
            for pair in hps_string.split(','):
                k, v = pair.split('=')
                if k == 'classes':
                    v = [int(x) for x in v.split('-')]
                elif '.' in v:
                    v = float(v)
                elif k != 'net':
                    v = int(v)
                setattr(self, k, v)


class BaseNet(nn.Module):
    def __init__(self, hps: HyperParams):
        super().__init__()
        self.hps = hps
        self.register_buffer('global_step', torch.IntTensor(1).zero_())


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


def upsample2d(x):
    # repeat is missing: https://github.com/pytorch/pytorch/issues/440
    # return x.repeat(1, 1, 2, 2)
    x = torch.stack([x[:, :, i // 2, :] for i in range(x.size()[2] * 2)], 2)
    x = torch.stack([x[:, :, :, i // 2] for i in range(x.size()[3] * 2)], 3)
    return x


# UNet:
# http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


class SmallUNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
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
        x1 = upsample2d(x1)
        x = torch.cat([x, x1], 1)
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, :, b:-b, b:-b])


class UNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.pool = nn.MaxPool2d(2, 2)
        conv = lambda c_in, c_out: nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv1_1 = conv(hps.n_channels, 32)
        self.conv1_2 = conv(32, 32)
        self.conv2_1 = conv(32, 64)
        self.conv2_2 = conv(64, 64)
        self.conv3_1 = conv(64, 128)
        self.conv3_2 = conv(128, 128)
        self.conv4_1 = conv(128, 256)
        self.conv4_2 = conv(256, 256)
        self.conv5_1 = conv(256, 512)
        self.conv5_2 = conv(512, 512)
        self.conv6_1 = conv(512 + 256, 256)
        self.conv6_2 = conv(256, 256)
        self.conv7_1 = conv(256 + 128, 128)
        self.conv7_2 = conv(128, 128)
        self.conv8_1 = conv(128 + 64, 64)
        self.conv8_2 = conv(64, 64)
        self.conv9_1 = conv(64 + 32, 32)
        self.conv9_2 = conv(32, 32)
        self.conv10 = nn.Conv2d(32, hps.n_classes, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1_1(x))
        x1 = F.relu(self.conv1_2(x1))

        x2 = self.pool(x1)
        x2 = F.relu(self.conv2_1(x2))
        x2 = F.relu(self.conv2_2(x2))

        x3 = self.pool(x2)
        x3 = F.relu(self.conv3_1(x3))
        x3 = F.relu(self.conv3_2(x3))

        x4 = self.pool(x3)
        x4 = F.relu(self.conv4_1(x4))
        x4 = F.relu(self.conv4_2(x4))

        x5 = self.pool(x4)
        x5 = F.relu(self.conv5_1(x5))
        x5 = F.relu(self.conv5_2(x5))

        x6 = torch.cat([upsample2d(x5), x4], 1)
        x6 = F.relu(self.conv6_1(x6))
        x6 = F.relu(self.conv6_2(x6))

        x7 = torch.cat([upsample2d(x6), x3], 1)
        x7 = F.relu(self.conv7_1(x7))
        x7 = F.relu(self.conv7_2(x7))

        x8 = torch.cat([upsample2d(x7), x2], 1)
        x8 = F.relu(self.conv8_1(x8))
        x8 = F.relu(self.conv8_2(x8))

        x9 = torch.cat([upsample2d(x8), x1], 1)
        x9 = F.relu(self.conv9_1(x9))
        x9 = F.relu(self.conv9_2(x9))

        x10 = self.conv10(x9)
        b = self.hps.patch_border
        return F.sigmoid(x10[:, :, b:-b, b:-b])
