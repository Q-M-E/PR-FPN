import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
from torch import nn
import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone

__all__ = ["build_resnet_prfpn_backbone", "build_retinanet_resnet_prfpn_backbone", "PRFPN_backbone"]

from dcn_v2 import DCN as dcn_v2
from DCNv4 import DCNv4 as dcn_v4
from detectron2.layers import (CNNBlockBase, Conv2d, DeformConv, ModulatedDeformConv, ShapeSpec, get_norm, )


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


'''                                                        SCR                                              '''


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRB(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True,
                 num_fusion=2,
                 compress_c=8):
        super().__init__()
        if num_fusion == 2:
            self.weight_level_1 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_2 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_levels = nn.Conv2d(compress_c * 2, num_fusion, kernel_size=1, stride=1, padding=0)
        elif num_fusion == 3:
            self.weight_level_1 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_2 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_3 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_levels = nn.Conv2d(compress_c * 3, num_fusion, kernel_size=1, stride=1, padding=0)
        else:
            self.weight_level_1 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_2 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_3 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_4 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)

        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):

        if len(x) == 2:
            input1, input2 = x
            level_1_weight_v = self.weight_level_1(input1)
            level_2_weight_v = self.weight_level_2(input2)

            levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
            levels_weight = self.weight_levels(levels_weight_v)
            levels_weight = F.sigmoid(levels_weight)

            fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + input2 * levels_weight[:, 1:2, :, :]
        elif len(x) == 3:
            input1, input2, input3 = x
            level_1_weight_v = self.weight_level_1(input1)
            level_2_weight_v = self.weight_level_2(input2)
            level_3_weight_v = self.weight_level_3(input3)

            levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
            levels_weight = self.weight_levels(levels_weight_v)
            levels_weight = F.sigmoid(levels_weight)

            fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + input2 * levels_weight[:, 1:2, :, :] + input3 * \
                                levels_weight[:, 2:, :, :]
        else:
            input1, input2, input3, input4 = x
            level_1_weight_v = self.weight_level_1(input1)
            level_2_weight_v = self.weight_level_2(input2)
            level_3_weight_v = self.weight_level_2(input3)
            level_4_weight_v = self.weight_level_4(input4)

            levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v, level_4_weight_v), 1)
            levels_weight = self.weight_levels(levels_weight_v)
            levels_weight = F.sigmoid(levels_weight)

            fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                                input2 * levels_weight[:, 1:2, :, :] + \
                                input3 * levels_weight[:, 2:3, :, :] + \
                                input4 * levels_weight[:, 3:, :, :]

        y = self.restructure(fused_out_reduced)
        return y

    def restructure(self, fused_out_reduced):
        gn_x = self.gn(fused_out_reduced)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * fused_out_reduced
        x_2 = w2 * fused_out_reduced
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)

        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class CRB(nn.Module):
    """
    gamma: 0<gamma<1
    """

    def __init__(self,
                 op_channel: int,
                 gamma: float = 1 / 2,
                 squeeze_ratio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.low_channel = low_channel = int(gamma * op_channel)
        self.up_channel = up_channel = op_channel - low_channel
        self.squeeze1 = nn.Conv2d(low_channel, low_channel // squeeze_ratio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(up_channel, up_channel // squeeze_ratio, kernel_size=1, bias=False)
        # lower
        self.GWC = nn.Conv2d(low_channel // squeeze_ratio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(low_channel // squeeze_ratio, op_channel, kernel_size=1, bias=False)

        # upper
        self.PWC2 = nn.Conv2d(up_channel // squeeze_ratio, op_channel - up_channel // squeeze_ratio, kernel_size=1,
                              bias=False)

        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # partition
        low, up = torch.split(x, [self.low_channel, self.up_channel], dim=1)
        low, up = self.squeeze1(low), self.squeeze2(up)
        # re-extract
        Z1 = self.GWC(low) + self.PWC1(low)
        Z2 = torch.cat([self.PWC2(up), up], dim=1)
        # re-fuse
        out = torch.cat([Z1, Z2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class SCR(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 12,
                 group_kernel_size: int = 3,
                 gate_treshold: float = 0.5,
                 gamma: float = 1 / 2,
                 squeeze_ratio: int = 2,
                 group_size: int = 2,
                 num_fusion=2,
                 compress_c=8
                 ):
        super().__init__()
        self.SRB = SRB(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold,
                       num_fusion=num_fusion,
                       compress_c=compress_c)
        self.CRB = CRB(op_channel,
                       gamma=gamma,
                       squeeze_ratio=squeeze_ratio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRB(x)
        x = self.CRB(x)
        return x


'''                                             PR-FPN                                            '''


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            # nn.Upsample(scale_factor=scale_factor, mode='bilinear')
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

        # carafe
        # from mmcv.ops import CARAFEPack
        # self.upsample = nn.Sequential(
        #     BasicConv(in_channels, out_channels, 1),
        #     CARAFEPack(out_channels, scale_factor=scale_factor)
        # )

    def forward(self, x):
        x = self.upsample(x)

        return x


class adjust(nn.Module):
    def __init__(self, out_nc=128, norm=None):
        super(adjust, self).__init__()
        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.dcnpack = dcn_v4(out_nc, 3, stride=1, padding=1, dilation=1, group=4,
                              extra_offset_mask=True)
        # self.dcnpack = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
        #                         extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_u, feat_s, main_path=None):
        # HW = feat_u.size()[2:]
        # if feat_l.size()[2:] != feat_s.size()[2:]:
        #     feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        # else:
        #     feat_up = feat_s
        offset = self.offset(torch.cat([feat_u, feat_s * 2], dim=1))  # concat for offset by computing the dif
        feat_adjust = self.relu(self.dcnpack([feat_s, offset]))  # [feat, offset]
        return feat_adjust


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x


class PRFPN_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=None, norm=''):
        super(PRFPN_2, self).__init__()

        if channel is None:
            channel = [64, 128]
        self.inter_dim = inter_dim
        compress_c = 8
        self.level = level
        if self.level == 0:
            self.upsample = Upsample(channel[1], channel[0])
            lateral_norm = get_norm(norm=norm, out_channels=channel[0])
            self.adjust_up = adjust(channel[0], norm=lateral_norm)  # proposed fapn
        else:
            self.downsample = Downsample(channel[0], channel[1])
            lateral_norm = get_norm(norm=norm, out_channels=channel[1])
            self.adjust_down = adjust(channel[1], norm=lateral_norm)

        self.refine = SCR(self.inter_dim, self.inter_dim, num_fusion=2, compress_c=compress_c)

    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
            input2 = self.adjust_up(input1, input2)
        elif self.level == 1:
            input1 = self.downsample(input1)
            input1 = self.adjust_down(input2, input1)

        out = self.refine((input1, input2))

        return out


class PRFPN_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=None, norm=''):
        super(PRFPN_3, self).__init__()

        if channel is None:
            channel = [64, 128, 256]
        self.inter_dim = inter_dim
        compress_c = 8

        self.refine = SCR(self.inter_dim, self.inter_dim, num_fusion=3, compress_c=compress_c)
        # else:
        #     self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        self.level = level
        if self.level == 0:
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
            lateral_norm = get_norm(norm=norm, out_channels=channel[0])
            self.adjust4x = adjust(channel[0], norm=lateral_norm)
            self.adjust2x = adjust(channel[0], norm=lateral_norm)

        elif self.level == 1:
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
            lateral_norm = get_norm(norm=norm, out_channels=channel[1])
            self.adjust_up2x1 = adjust(channel[1], norm=lateral_norm)
            self.adjust_down2x1 = adjust(channel[1], norm=lateral_norm)

        elif self.level == 2:
            self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)
            lateral_norm = get_norm(norm=norm, out_channels=channel[2])
            self.adjust_down2x = adjust(channel[2], norm=lateral_norm)
            self.adjust_down4x = adjust(channel[2], norm=lateral_norm)

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
            input2 = self.adjust2x(input1, input2)
            input3 = self.adjust4x(input1, input3)
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input3 = self.adjust_up2x1(input2, input3)
            input1 = self.downsample2x1(input1)
            input1 = self.adjust_down2x1(input2, input1)
        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input1 = self.adjust_down4x(input3, input1)
            input2 = self.downsample2x(input2)
            input2 = self.adjust_down2x(input3, input2)

        out = self.refine((input1, input2, input3))
        # self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        return out


class PRFPN_4(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=None, norm=''):
        super(PRFPN_4, self).__init__()
        if channel is None:
            channel = [256, 512, 1024, 2048]
        self.inter_dim = inter_dim
        compress_c = 8

        self.level = level
        if self.level == 0:
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
            self.upsample8x = Upsample(channel[3], channel[0], scale_factor=8)
            lateral_norm = get_norm(norm=norm, out_channels=channel[0])
            self.adjust4x = adjust(channel[0], norm=lateral_norm)
            self.adjust2x = adjust(channel[0], norm=lateral_norm)
            self.adjust8x = adjust(channel[0], norm=lateral_norm)

        elif self.level == 1:
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.upsample4x1 = Upsample(channel[3], channel[1], scale_factor=4)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
            lateral_norm = get_norm(norm=norm, out_channels=channel[1])
            self.adjust_up2x1 = adjust(channel[1], norm=lateral_norm)
            self.adjust_down2x1 = adjust(channel[1], norm=lateral_norm)
            self.adjust_up4x1 = adjust(channel[1], norm=lateral_norm)

        elif self.level == 2:
            self.upsample2x2 = Upsample(channel[3], channel[2], scale_factor=2)
            self.downsample2x2 = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x2 = Downsample(channel[0], channel[2], scale_factor=4)
            lateral_norm = get_norm(norm=norm, out_channels=channel[2])
            self.adjust_up2x2 = adjust(channel[2], norm=lateral_norm)
            self.adjust_down2x2 = adjust(channel[2], norm=lateral_norm)
            self.adjust_down4x2 = adjust(channel[2], norm=lateral_norm)

        elif self.level == 3:
            self.downsample2x3 = Downsample(channel[2], channel[3], scale_factor=2)
            self.downsample4x3 = Downsample(channel[1], channel[3], scale_factor=4)
            self.downsample8x3 = Downsample(channel[0], channel[3], scale_factor=8)
            lateral_norm = get_norm(norm=norm, out_channels=channel[3])
            self.adjust_down2x3 = adjust(channel[3], norm=lateral_norm)
            self.adjust_down8x3 = adjust(channel[3], norm=lateral_norm)
            self.adjust_down4x3 = adjust(channel[3], norm=lateral_norm)

        self.refine = SCR(self.inter_dim, self.inter_dim, num_fusion=4, compress_c=compress_c)

    def forward(self, x):
        input1, input2, input3, input4 = x

        if self.level == 0:
            input2 = self.upsample2x(input2)
            input2 = self.adjust2x(input1, input2)
            input3 = self.upsample4x(input3)
            input3 = self.adjust4x(input1, input3)
            input4 = self.upsample8x(input4)
            input4 = self.adjust8x(input1, input4)

        elif self.level == 1:
            input1 = self.downsample2x1(input1)
            input1 = self.adjust_down2x1(input2, input1)
            input3 = self.upsample2x1(input3)
            input3 = self.adjust_up2x1(input2, input3)
            input4 = self.upsample4x1(input4)
            input4 = self.adjust_up4x1(input2, input4)

        elif self.level == 2:
            input1 = self.downsample4x2(input1)
            input1 = self.adjust_down4x2(input3, input1)
            input2 = self.downsample2x2(input2)
            input2 = self.adjust_down2x2(input3, input2)
            input4 = self.upsample2x2(input4)
            input4 = self.adjust_up2x2(input3, input4)

        elif self.level == 3:

            input1 = self.downsample8x3(input1)
            input1 = self.adjust_down8x3(input4, input1)
            input2 = self.downsample4x3(input2)
            input2 = self.adjust_down4x3(input4, input2)
            input3 = self.downsample2x3(input3)
            input3 = self.adjust_down2x3(input4, input3)

        out = self.refine((input1, input2, input3, input4))

        return out


class Body(nn.Module):
    def __init__(self, channels=None, norm='', nums_fusion=4):
        super(Body, self).__init__()

        if channels is None:
            channels = [64, 128, 256, 512]
        self.nums_fusion = nums_fusion

        self.prfpn_2_level0 = PRFPN_2(inter_dim=channels[0], level=0, channel=channels, norm=norm)
        self.prfpn_2_level1 = PRFPN_2(inter_dim=channels[1], level=1, channel=channels, norm=norm)

        self.prfpn_3_level0 = PRFPN_3(inter_dim=channels[0], level=0, channel=channels, norm=norm)
        self.prfpn_3_level1 = PRFPN_3(inter_dim=channels[1], level=1, channel=channels, norm=norm)
        self.prfpn_3_level2 = PRFPN_3(inter_dim=channels[2], level=2, channel=channels, norm=norm)

        if self.nums_fusion == 4:
            self.prfpn_4_level0 = PRFPN_4(inter_dim=channels[0], level=0, channel=channels, norm=norm)
            self.prfpn_4_level1 = PRFPN_4(inter_dim=channels[1], level=1, channel=channels, norm=norm)
            self.prfpn_4_level2 = PRFPN_4(inter_dim=channels[2], level=2, channel=channels, norm=norm)
            self.prfpn_4_level3 = PRFPN_4(inter_dim=channels[3], level=3, channel=channels, norm=norm)

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.prfpn_2_level0((x0, x1))
        x1 = self.prfpn_2_level1((x0, x1))

        x0 = self.prfpn_3_level0((x0, x1, x2))
        x1 = self.prfpn_3_level1((x0, x1, x2))
        x2 = self.prfpn_3_level2((x0, x1, x2))
        if self.nums_fusion == 4:
            out0 = self.prfpn_4_level0((x0, x1, x2, x3))
            out1 = self.prfpn_4_level1((x0, x1, x2, x3))
            out2 = self.prfpn_4_level2((x0, x1, x2, x3))
            out3 = self.prfpn_4_level3((x0, x1, x2, x3))
            return out0, out1, out2, out3
        return x0, x1, x2


class PRFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 norm='',
                 nums_fusion=4):
        super(PRFPN, self).__init__()

        in_channels_reduced = [i // 2 for i in in_channels]
        self.fp16_enabled = False

        self.conv0 = Conv(in_channels[0], in_channels_reduced[0], 1)
        self.conv1 = Conv(in_channels[1], in_channels_reduced[1], 1)
        self.conv2 = Conv(in_channels[2], in_channels_reduced[2], 1)
        self.conv3 = Conv(in_channels[3], in_channels_reduced[3], 1)

        self.prfpn_body = Body(channels=in_channels_reduced, norm=norm, nums_fusion=nums_fusion)

        self.conv00 = Conv(in_channels_reduced[0], out_channels, 1)
        self.conv11 = Conv(in_channels_reduced[1], out_channels, 1)
        self.conv22 = Conv(in_channels_reduced[2], out_channels, 1)
        self.conv33 = Conv(in_channels_reduced[3], out_channels, 1)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0, out1, out2, out3 = self.prfpn_body((x0, x1, x2, x3))

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)

        return [out0, out1, out2, out3]


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(stride, strides[i - 1])


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class PRFPN_backbone(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, nums_fusion=4):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
        """
        super(PRFPN_backbone, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        _assert_strides_are_log2_contiguous(strides)

        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        self.prfpn = PRFPN(in_channels=in_channels_per_feature, out_channels=out_channels, norm=norm, nums_fusion=nums_fusion)

        stage = int(math.log2(strides[len(in_channels_per_feature) - 1]))
        # Place convs into top-down order (from low to high resolution) to make the top-down computation in forward clearer.

        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = self.prfpn((x[3], x[2], x[1], x[0]))
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for
                name in self._out_features}


@BACKBONE_REGISTRY.register()
def build_resnet_prfpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = PRFPN_backbone(bottom_up=bottom_up, in_features=in_features, out_channels=out_channels,
                              norm=cfg.MODEL.FPN.NORM, top_block=LastLevelMaxPool(), nums_fusion=4
                              )
    return backbone


@BACKBONE_REGISTRY.register()
def build_retinanet_resnet_prfpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = PRFPN_backbone(bottom_up=bottom_up, in_features=in_features, out_channels=out_channels,
                              norm=cfg.MODEL.FPN.NORM,
                              top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
                              nums_fusion=3
                              )
    return backbone
