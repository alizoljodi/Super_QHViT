# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# modified from OFA: https://github.com/mit-han-lab/once-for-all
import torch.nn.functional as F

from collections import OrderedDict
import torch.nn as nn
from .nn_utils import get_same_padding, build_activation, make_divisible, drop_connect
from .nn_base import MyModule
from .activations import *
from copy import deepcopy
from misc.quantize_utils import QModule, QConv2d, QLinear


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvBnActLayer.__name__: ConvBnActLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
    }

    layer_name = layer_config.pop("name")
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class SELayer(nn.Module):
    REDUCTION = 4

    def __init__(self, channel):
        super(SELayer, self).__init__()

        self.channel = channel
        self.reduction = SELayer.REDUCTION

        num_mid = make_divisible(self.channel // self.reduction, divisor=8)

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("reduce", nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("expand", nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
                    ("h_sigmoid", Hsigmoid(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        # x: N, C, H, W
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)  # N, C, 1, 1
        y = self.fc(y)
        return x * y


class QSELayer(nn.Module):
    REDUCTION = 4

    def __init__(self, channel, w_bit=-1, a_bit=-1, half_wave=True):
        super(QSELayer, self).__init__()

        self.channel = channel
        self.reduction = QSELayer.REDUCTION

        num_mid = make_divisible(self.channel // self.reduction, divisor=8)

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "reduce",
                        QConv2d(
                            self.channel,
                            num_mid,
                            1,
                            1,
                            0,
                            bias=True,
                            w_bit=w_bit,
                            a_bit=a_bit,
                            half_wave=half_wave,
                        ),
                    ),
                    ("relu", nn.ReLU(inplace=True)),
                    (
                        "expand",
                        QConv2d(
                            num_mid,
                            self.channel,
                            1,
                            1,
                            0,
                            bias=True,
                            w_bit=w_bit,
                            a_bit=a_bit,
                            half_wave=half_wave,
                        ),
                    ),
                    ("h_sigmoid", Hsigmoid(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        # x: N, C, H, W
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)  # N, C, 1, 1
        y = self.fc(y)
        return x * y


class ConvBnActLayer(MyModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        use_bn=True,
        act_func="relu",
    ):
        super(ConvBnActLayer, self).__init__()
        # default normal 3x3_Conv with bn and relu
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func

        pad = get_same_padding(self.kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride,
            pad,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

        self.act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_Conv" % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedGroupConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_GroupConv" % (kernel_size[0], kernel_size[1])
        conv_str += "_O%d" % self.out_channels
        return conv_str

    @property
    def config(self):
        return {
            "name": ConvBnActLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return ConvBnActLayer(**config)


class QConvBnActLayer(QModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        use_bn=True,
        act_func="relu",
        w_bit=-1,
        a_bit=-1,
        half_wave=True,
    ):
        super(QConvBnActLayer, self).__init__(
            w_bit=w_bit, a_bit=a_bit, half_wave=half_wave
        )
        # default normal 3x3_Conv with bn and relu
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func

        self.w_bit = w_bit
        self.a_bit = a_bit
        # self.half_wave=half_wave

        pad = get_same_padding(self.kernel_size)
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )
        """QConv2d(in_channels, out_channels, self.kernel_size,
            stride, pad, dilation=dilation, groups=groups, bias=bias, w_bit=w_bit,a_bit=a_bit,half_wave=half_wave
        )"""
        """nn.Conv2d(in_channels, out_channels, self.kernel_size,
            stride, pad, dilation=dilation, groups=groups, bias=bias)"""
        """"""
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

        self.act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        in_channel = x.size(1)
        filters = self.conv.weight[: self.out_channels, :in_channel, :, :].contiguous()
        inputs, weight, bias = self._quantize(inputs=x, weight=filters, bias=None)

        padding = get_same_padding(self.kernel_size)
        x = F.conv2d(inputs, filters, None, self.stride, padding, self.dilation, 1)
        # self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_Conv" % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedGroupConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_GroupConv" % (kernel_size[0], kernel_size[1])
        conv_str += "_O%d" % self.out_channels
        return conv_str

    @property
    def config(self):
        return {
            "name": QConvBnActLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "w_bit": self.w_bit,
            "a_bit": self.a_bit,
            "half_wave": self.half_wave,
        }

    @staticmethod
    def build_from_config(config):
        return QConvBnActLayer(**config)


class IdentityLayer(MyModule):
    def __init__(
        self,
    ):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

    @property
    def module_str(self):
        return "Identity"

    @property
    def config(self):
        return {
            "name": IdentityLayer.__name__,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)


class LinearLayer(MyModule):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # self.dropout_rate = dropout_rate
        # if self.dropout_rate > 0:
        #    self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        # else:
        #    self.dropout = None

        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        # if dropout is not None:
        #    x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self):
        return "%dx%d_Linear" % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            "name": LinearLayer.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            #'dropout_rate': self.dropout_rate,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)


class QLinearLayer(MyModule):
    def __init__(
        self, in_features, out_features, bias=True, w_bit=-1, a_bit=-1, half_wave=True
    ):
        super(QLinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.w_bit = w_bit
        self.a_bit = a_bit
        self.half_wave = half_wave

        # self.dropout_rate = dropout_rate
        # if self.dropout_rate > 0:
        #    self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        # else:
        #    self.dropout = None

        self.linear = QLinear(
            in_features,
            out_features,
            bias,
            w_bit=w_bit,
            a_bit=a_bit,
            half_wave=half_wave,
        )

    def forward(self, x):
        # if dropout is not None:
        #    x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self):
        return "%dx%d_Linear" % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            "name": QLinearLayer.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "w_bit": self.w_bit,
            "a_bit": self.a_bit,
            "half_wave": self.half_wave,
            #'dropout_rate': self.dropout_rate,
        }

    @staticmethod
    def build_from_config(config):
        return QLinearLayer(**config)


class ShortcutLayer(MyModule):
    def __init__(self, in_channels, out_channels, reduction=1):
        super(ShortcutLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.reduction = reduction

        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        if self.reduction > 1:
            padding = 0 if x.size(-1) % 2 == 0 else 1
            x = F.avg_pool2d(x, self.reduction, padding=padding)
        if self.in_channels != self.out_channels:
            x = self.conv(x)
        return x

    @property
    def module_str(self):
        if self.in_channels == self.out_channels and self.reduction == 1:
            conv_str = "IdentityShortcut"
        else:
            if self.reduction == 1:
                conv_str = "%d-%d_Shortcut" % (self.in_channels, self.out_channels)
            else:
                conv_str = "%d-%d_R%d_Shortcut" % (
                    self.in_channels,
                    self.out_channels,
                    self.reduction,
                )
        return conv_str

    @property
    def config(self):
        return {
            "name": ShortcutLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "reduction": self.reduction,
        }

    @staticmethod
    def build_from_config(config):
        return ShortcutLayer(**config)


class QShortcutLayer(QModule):
    def __init__(
        self, in_channels, out_channels, reduction=1, w_bit=-1, a_bit=-1, half_wave=True
    ):
        super(QShortcutLayer, self).__init__(
            w_bit=w_bit, a_bit=w_bit, half_wave=half_wave
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.reduction = reduction

        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        if self.reduction > 1:
            padding = 0 if x.size(-1) % 2 == 0 else 1
            x = F.avg_pool2d(x, self.reduction, padding=padding)
        if self.in_channels != self.out_channels:
            filters = self.conv.weight[
                : self.out_channels, : self.in_channels, :, :
            ].contiguous()
            inputs, weight, bias = self._quantize(inputs=x, weight=filters, bias=None)
            x = F.conv2d(inputs, weight, None)
        inputs, _, _ = self._quantize(inputs=x, weight=None, bias=None)
        return inputs


class MBInvertedConvLayer(MyModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        mid_channels=None,
        act_func="relu6",
        use_se=False,
        channels_per_group=1,
    ):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se

        self.channels_per_group = channels_per_group

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                self.in_channels, feature_dim, 1, 1, 0, bias=False
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(feature_dim)),
                        ("act", build_activation(self.act_func, inplace=True)),
                    ]
                )
            )

        assert feature_dim % self.channels_per_group == 0
        active_groups = feature_dim // self.channels_per_group
        pad = get_same_padding(self.kernel_size)
        depth_conv_modules = [
            (
                "conv",
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size,
                    stride,
                    pad,
                    groups=active_groups,
                    bias=False,
                ),
            ),
            ("bn", nn.BatchNorm2d(feature_dim)),
            ("act", build_activation(self.act_func, inplace=True)),
        ]
        if self.use_se:
            depth_conv_modules.append(("se", SELayer(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                    ("bn", nn.BatchNorm2d(out_channels)),
                ]
            )
        )

        self.rescale = (
            1.0  # nn.Parameter(1. * torch.ones((out_channels)), requires_grad=False)
        )

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        # if x.shape[-1] % 2 != 0:
        #     x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x * self.rescale  # .reshape([1, -1, 1, 1])

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = "%dx%d_MBConv%d_%s" % (
            self.kernel_size,
            self.kernel_size,
            expand_ratio,
            self.act_func.upper(),
        )
        if self.use_se:
            layer_str = "SE_" + layer_str
        layer_str += "_O%d" % self.out_channels
        return layer_str

    @property
    def config(self):
        return {
            "name": MBInvertedConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "use_se": self.use_se,
            "channels_per_group": self.channels_per_group,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)


class MBInvertedQConvLayer(MyModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        mid_channels=None,
        act_func="relu6",
        use_se=False,
        pw_w_bit=-1,
        pw_a_bit=-1,
        dw_w_bit=-1,
        dw_a_bit=-1,
        **kwargs
    ):
        super(MBInvertedQConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se

        self.pw_w_bit = pw_w_bit
        self.pw_a_bit = pw_a_bit
        self.dw_w_bit = dw_w_bit
        self.dw_a_bit = dw_a_bit

        if self.mid_channels is None:
            # print(self.in_channels, self.expand_ratio)
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            QConv2d(
                                self.in_channels,
                                feature_dim,
                                1,
                                1,
                                0,
                                bias=False,
                                w_bit=8,
                                a_bit=8,
                                half_wave=False,
                            ),
                        ),  # , w_bit=pw_w_bit, a_bit=pw_a_bit, half_wave=False)),
                        ("bn", nn.BatchNorm2d(feature_dim)),
                        ("act", build_activation(self.act_func, inplace=True)),
                    ]
                )
            )

        pad = get_same_padding(self.kernel_size)
        depth_conv_modules = [
            (
                "conv",
                QConv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size,
                    stride,
                    pad,
                    groups=feature_dim,
                    bias=False,
                    w_bit=8,
                    a_bit=8,
                    half_wave=False,
                ),
            ),
            ("bn", nn.BatchNorm2d(feature_dim)),
            ("act", build_activation(self.act_func, inplace=True)),
        ]
        if self.use_se:
            depth_conv_modules.append(("se", SELayer(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        QConv2d(
                            feature_dim,
                            out_channels,
                            1,
                            1,
                            0,
                            bias=False,
                            w_bit=8,
                            a_bit=8,
                            half_wave=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_channels)),
                ]
            )
        )

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    def set_quantization_policy(
        self, pw_w_bit=None, pw_a_bit=None, dw_w_bit=None, dw_a_bit=None
    ):
        for name, module in self.named_modules():
            if isinstance(module, QConv2d):
                if module.in_channels == module.groups:
                    module.w_bit = dw_w_bit
                    module.a_bit = dw_a_bit
                else:
                    module.w_bit = pw_w_bit
                    module.a_bit = pw_a_bit

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = "%dx%d_MBConv%d_%s" % (
            self.kernel_size,
            self.kernel_size,
            expand_ratio,
            self.act_func.upper(),
        )
        return layer_str

    @property
    def config(self):
        return {
            "name": MBInvertedQConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "pw_w_bit": self.pw_w_bit,
            "pw_a_bit": self.pw_a_bit,
            "dw_w_bit": self.dw_w_bit,
            "dw_a_bit": self.dw_a_bit,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedQConvLayer(**config)


class MobileInvertedResidualBlock(MyModule):
    def __init__(
        self, mobile_inverted_conv, shortcut, drop_connect_rate=0, bit_width=[8]
    ):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        self.drop_connect_rate = drop_connect_rate
        self.active_bit_width = max(bit_width)

    def forward(self, x, writer=None, batch_ix=None, block_id=None):
        in_channel = x.size(1)
        input_ = deepcopy(x.clone().detach())

        if (
            self.mobile_inverted_conv is None
        ):  # or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None:  # or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
            # print('or this happen')
        else:
            # self.shortcut.w_bit=self.active_bit_width
            # self.shortcut.a_bit=self.active_bit_width
            self.shortcut.bit_width = self.active_bit_width

            im = self.shortcut(x)
            self.mobile_inverted_conv.active_bit_width = self.active_bit_width
            x = self.mobile_inverted_conv(x)
            if (
                self.drop_connect_rate > 0
                and in_channel == im.size(1)
                and self.shortcut.reduction == 1
            ):
                x = drop_connect(x, p=self.drop_connect_rate, training=self.training)
            res = x + im
        return res

    @property
    def module_str(self):
        return "(%s, %s)" % (
            (
                self.mobile_inverted_conv.module_str
                if self.mobile_inverted_conv is not None
                else None
            ),
            self.shortcut.module_str if self.shortcut is not None else None,
        )

    @property
    def config(self):
        return {
            "name": MobileInvertedResidualBlock.__name__,
            "mobile_inverted_conv": (
                self.mobile_inverted_conv.config
                if self.mobile_inverted_conv is not None
                else None
            ),
            "shortcut": self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config["mobile_inverted_conv"])
        shortcut = set_layer_from_config(config["shortcut"])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)
