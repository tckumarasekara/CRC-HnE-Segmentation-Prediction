import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer
import numpy as np


#### ===== basic Unet ===== ####
# These modules are the building blocks for the U-Net architecture derived from
# https://arxiv.org/pdf/1505.04597
# With slight alteration
class UnetConv(nn.Module):
    """
    UnetConv is a standard U-Net convolution block with 2 convolution layers, seperated by ReLu and Dropout Layers

    in_size: channel dimension of input
    out_size: channel dimension of output
    is_batchnorm: boolean option, whether a batchnorm layer is added or not
    ks: kernel size normally 3
    stride: stride of convolution, should stay 1
    padding: padding of convolution, needs to be 1 to keep dimensions
    gpus: whether gpus are used for implementation. Currently only on Linux!
    dropout_val: p of dropout layer. 0 will mean input=output
    """

    def __init__(self, in_size: int, out_size: int, is_batchnorm: bool, ks=3, stride=1, padding=1, gpus=False,
                 dropout_val=0.001):
        super(UnetConv, self).__init__()
        self.ks = ks
        self.stride = stride
        self.padding = padding

        if is_batchnorm:
            self.conv = nn.Sequential(nn.Sequential(
                nn.Dropout(dropout_val),
                nn.Conv2d(in_size, out_size, ks, padding=padding, stride=stride),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_val),
                nn.Conv2d(out_size, out_size, ks, padding=padding, stride=stride),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True)))
        else:
            self.conv = nn.Sequential(nn.Sequential(
                nn.Dropout(dropout_val),
                nn.Conv2d(in_size, out_size, ks, padding=padding, stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_val),
                nn.Conv2d(out_size, out_size, ks, padding=padding, stride=stride),
                nn.ReLU(inplace=True)))
        if gpus:
            self.conv.cuda()

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class UnetUp(nn.Module):
    """
    UnetUp is a upsampling layer with a 1x1 Convolution and a prepended dropout layer

    in_size: channel dimension of input
    out_size: channel dimension of output
    ks: kernel size normally 3
    stride: stride of convolution, should stay 1
    padding: padding of convolution, needs to be 1 to keep dimensions
    gpus: whether gpus are used for implementation. Currently only on Linux!
    dropout_val: p of dropout layer. 0 will mean input=output
    """

    def __init__(self, in_size: int, out_size: int, gpus: bool = False, dropout_val: float = 0.0):
        super(UnetUp, self).__init__()
        self.conv = UnetConv(in_size, out_size, False)
        self.up = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_size, out_size, 1)
        )
        if gpus:
            self.conv.cuda()
            self.up.cuda()

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], dim=1)
        return self.conv(outputs0)


#### ===== Context Unet ===== #####
# These modules are derived from
# https://arxiv.org/abs/1802.10508
class SimpleUnetConv(nn.Module):
    """SimpleUnetConv is a one layer Convolution layer

    in_size: channel dimension of input
    out_size: channel dimension of output
    is_batchnorm: boolean option, whether a batchnorm layer is added or not
    ks: kernel size normally 3
    stride: stride of convolution, should stay 1
    padding: padding of convolution, needs to be 1 to keep dimensions
    gpus: whether gpus are used for implementation. Currently only on Linux!
    dropout_val: p of dropout layer. 0 will mean input=output
    """

    def __init__(self, in_size: int, out_size: int, ks: int = 3, stride: int = 2, padding: int = 1, gpus: bool = False,
                 dropout_val: float = 0):
        super(SimpleUnetConv, self).__init__()
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.conv = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, out_size, ks, padding=padding, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True))
        if gpus:
            self.conv.cuda()

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class SimpleUnetUp(nn.Module):
    """SimpleUnetUp is a upsampling module without dropout layer

    in_size: channel dimension of input
    out_size: channel dimension of output
    gpus: whether gpus are used for implementation. Currently only on Linux!
    """

    def __init__(self, in_size, out_size, gpus=False):
        super(SimpleUnetUp, self).__init__()
        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_size, out_size, 1),
        )
        if gpus:
            self.up.cuda()

    def forward(self, feature):
        return self.up(feature)

class ContextModule(nn.Module):
    """ContextModule is a double convolution layer with a dropout layer of 0.3 inbetween

        in_size: channel dimension of input
        out_size: channel dimension of output
        gpus: whether gpus are used for implementation. Currently only on Linux!
        """

    def __init__(self, in_size, out_size, gpus=False):
        super(ContextModule, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(out_size),
            nn.Conv2d(out_size, out_size, 3, padding=1)
        )
        if gpus:
            self.conv.cuda()

    def forward(self, x):
        return self.conv(x)


class Localization(nn.Module):
    """Localization is a Module with a different order of layers to the standard convolution layer

    in_size: channel dimension of input
    out_size: channel dimension of output
    gpus: whether gpus are used for implementation. Currently only on Linux!
    dropout_val: p of dropout layer. 0 will mean input=output
    """

    def __init__(self, in_size, out_size, dropout_val=0, gpus=False):
        super(Localization, self).__init__()
        self.conv = nn.Sequential(nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_val),
            nn.Conv2d(out_size, out_size, kernel_size=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)))
        if gpus:
            self.conv.cuda()

    def forward(self, x):
        return self.conv(x)


class SegmentationLayer(nn.Module):
    """SegmentationLayer is a Module which takes different level of feature maps and sum-wise adds them together

    x_size: size of the channel dimension of the lowest level of feature map
    y_size: size of the channel dimension of the second lowest level of feature map
    z_size: size of the channel dimension of the last feature map
    """

    def __init__(self, x_size, y_size, z_zize, gpus=False):
        super(SegmentationLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=x_size, out_channels=x_size, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=y_size, out_channels=y_size, kernel_size=1)
        self.up1 = SimpleUnetUp(x_size, y_size)
        self.up2 = SimpleUnetUp(y_size, z_zize)
        if gpus:
            self.conv1.cuda()
            self.conv2.cuda()
            self.up1.cuda()
            self.up2.cuda()

    def forward(self, x, y, z):
        con = self.conv1(x)
        up1 = self.up1(con)
        comb = up1 + self.conv2(y)
        return self.up2(comb) + z


#### ===== UneXt ===== ####
# These modules are the building blocks for the U-Net architecture with alterations inspired from swin transformer derived from
# http://arxiv.org/abs/2201.03545 (describes a ResNet block inspired by the swin stransformer)
# We have adapted the idea to fit U-net architecture
class UneXtConv(nn.Module):
    """
    UneXtConv is a convolution block with 1 depthwise convolution layer and 2 1x1 convolutionchannel-mixing layers
    seperated by GELU. Additionally, redidual connections are added at the end of the block adopting ResNet style
    from the paper.

    in_size: channel dimension of input and output
    ks: kernel size normally 7
    stride: stride of convolution, should stay 1
    padding: padding of convolution, needs to be 3 to keep dimensions
    gpus: whether gpus are used for implementation. Currently only on Linux!
    """

    def __init__(self, in_size: int, ks=7, stride=1, padding=3, gpus=False, dropout_val=0.001):
        super(UneXtConv, self).__init__()
        self.ks = ks
        self.stride = stride
        self.padding = padding

        self.block = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_size, in_size, ks, padding=padding, stride=stride, groups=in_size), # depthwise convolution
            nn.GroupNorm(1, in_size), # layerNorm
            nn.Conv2d(in_size, 4*in_size, 1, padding=0, stride=1),
            nn.GELU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(4*in_size, in_size, 1, padding=0, stride=1),
        ))

        if gpus:
            self.block.cuda()

    def forward(self, inputs):
        x = self.block(inputs)
        y = x + inputs
        return y


class UneXtUp(nn.Module):
    """
    UneXtUp is a upsampling layer with a 1x1 Convolution and a prepended dropout layer

    in_size: channel dimension of input
    out_size: channel dimension of output
    gpus: whether gpus are used for implementation. Currently only on Linux!
    """

    def __init__(self, in_size: int, out_size: int, gpus: bool = False, is_third: bool =False, dropout_val=0.001):
        super(UneXtUp, self).__init__()

        if is_third:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, 1),
                UneXtConv(out_size, dropout_val=dropout_val),
                UneXtConv(out_size, dropout_val=dropout_val),
                UneXtConv(out_size, dropout_val=dropout_val)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, 1),
                UneXtConv(out_size, dropout_val=dropout_val)
            )

        self.up = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_size, out_size, 1)
        )

        if gpus:
            self.conv.cuda()
            self.up.cuda()

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], dim=1)
        return self.conv(outputs0)


class UneXtDown(nn.Module):
    """
    UneXtDown is a downsampling layer with a 2x2 Convolution with stride 2

    in_size: channel dimension of input
    out_size: channel dimension of output
    gpus: whether gpus are used for implementation. Currently only on Linux!
    """

    def __init__(self, in_size: int, out_size: int, gpus: bool = False, dropout_val=0.001):
        super(UneXtDown, self).__init__()

        self.down = nn.Sequential(
            nn.GroupNorm(1, in_size),
            nn.Conv2d(in_size, out_size, kernel_size=2, stride=2)
        )

        if gpus:
            self.down.cuda()

    def forward(self, inputs):
        return self.down(inputs)


class SwinUnetrEnc(nn.Module):

    def __init__(self, in_size, embed_dim):
        super(SwinUnetrEnc, self).__init__()

        self.encoder = SwinTransformer(
            in_chans=in_size,
            embed_dim=embed_dim,
            window_size=(7, 7),
            patch_size=(4, 4),
            depths=[2, 2, 2, 2],
            num_heads=[2, 4, 8, 16],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=2,
        )

    def forward(self, x):
        features = self.encoder(x)
        return features[0], features[1], features[2], features[3]



#### ==== Spatial Transformer U-Net ==== ####
# This module does not work as intended
class SPTnet(nn.Module):
    """SPTnet is a module which is supposed to allow a spatial invariant convolution"""

    def __init__(self, in_size, out_size, val, stride=1, ks=3, dropout_val=0, gpus=False):
        super(SPTnet, self).__init__()
        self.gpus = gpus

        self.conv = nn.Sequential(nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, out_size, ks, padding=1, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_val),
            nn.Conv2d(out_size, out_size, ks, padding=1, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)))

        # spatial transformer localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_size, 32, kernel_size=7),  # 256*256*3
            nn.MaxPool2d(2, stride=2),  # 250*250*8
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),  # 125*125*8
            nn.MaxPool2d(2, stride=2),  # 121*121*16
            nn.ReLU(True)  # 60*60*16
        )
        # tranformation regressor for theta
        self.fc_loc = nn.Sequential(
            nn.Linear(val ** 2, 16),
            nn.Linear(16, 3 * 2)
        )
        self.fc_loc[1].weight.data.zero_()
        self.fc_loc[1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size(1) * xs.size(2) * xs.size(3))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, theta

    def rev_stn(self, x, theta: torch.tensor):
        if self.gpus:
            empty_tensor = torch.empty(theta.size()).cuda()
        else:
            empty_tensor = torch.empty(theta.size())
        for val, batch in enumerate(theta):
            new_theta = torch.cat((batch, torch.tensor([[0, 0, 1]]).cuda()), 0)
            theta = new_theta.inverse().cuda()
            theta = theta[:2, :]
            empty_tensor[val, :, :] = theta
        grid = F.affine_grid(empty_tensor, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, ori_img):
        x, theta = self.stn(ori_img)
        x = self.conv(x)
        x = self.rev_stn(x, theta)
        return x


class SAHead(nn.Module):
    def __init__(self, in_size, out_size, level, gpus, dropout_val, ks=3, padding=1, stride=1):
        super(SAHead, self).__init__()
        self.gpus = gpus
        self.conv1 = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, out_size, ks, padding=1, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(out_size, out_size, ks, padding=1, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_val),
            nn.Conv2d(out_size, out_size, ks, padding=1, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.localization = nn.Sequential(
            nn.Conv2d(out_size, 8, kernel_size=7, padding=3),  # 64**2/level**2 *8
            nn.MaxPool2d(2),  # 32*32*8
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),  # 32*32*16
            nn.MaxPool2d(2),  # 16*16*16
            nn.ReLU(True)
        )
        # W/4 H/4 16
        # W/8 H/8 16
        # tranformation regressor for theta
        self.fc_loc = nn.Sequential(
            nn.Linear(level * 16, 16),
            nn.Linear(16, 3 * 2)
        )
        self.fc_loc[1].weight.data.zero_()
        self.fc_loc[1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        if self.gpus:
            self.conv1.cuda()
            self.fc_loc.cuda()
            self.conv2.cuda()
            self.localization.cuda()
            self.sigmoid.cuda()

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size(1) * xs.size(2) * xs.size(3))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, theta

    def rev_stn(self, x, theta: torch.tensor):
        if self.gpus:
            empty_tensor = torch.empty(theta.size()).cuda()
        else:
            empty_tensor = torch.empty(theta.size())
        for val, batch in enumerate(theta):
            if self.gpus:
                new_theta = torch.cat((batch, torch.tensor([[0, 0, 1]]).cuda()), 0)
                theta = new_theta.inverse().cuda()

            else:
                new_theta = torch.cat((batch, torch.tensor([[0, 0, 1]])), 0)
                theta = new_theta.inverse()

            theta = theta[:2, :]
            empty_tensor[val, :, :] = theta
        grid = F.affine_grid(empty_tensor, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        upper = self.conv1(x)
        up, theta = self.stn(upper)
        middle = self.conv2(up)
        middle = self.sigmoid(middle)
        lower = self.conv1(x)
        t = self.rev_stn(middle, theta)
        vals = nn.functional.softmax(t, dim=1)
        return lower * vals


class multiHeadBlock(nn.Module):
    def __init__(self, num_heads, in_size, level, gpus, dropout_val, ks=3, padding=1, stride=1):
        super().__init__()
        self.gpus = gpus
        self.in_size = in_size
        self.mlist = nn.ModuleList()
        self.level = int(8/level)**2
        for i in range(0, num_heads):
            self.mlist.append(SAHead(in_size, in_size, self.level, gpus, dropout_val, ks, padding, stride))
        self.conv = SimpleUnetConv(in_size * num_heads, out_size=in_size, stride = 1, gpus=gpus,
                                   dropout_val=dropout_val)
        self.norm = nn.BatchNorm2d(in_size)
        if self.gpus:
            self.mlist.cuda()
            self.conv.cuda()
            self.norm.cuda()

    def forward(self, x):
        e_torch = self.mlist[0](x)
        if self.gpus:
            e_torch.cuda()
        for i in self.mlist[1:]:
            e_torch = torch.cat([e_torch, i(x)], dim=1)
        res = self.conv(e_torch)
        final = res + x
        final = self.norm(final)
        return final


class forwardProcessingBlock(nn.Module):
    def __init__(self, in_size, gpus, dropout_val, ks=3, padding=1, stride=1):
        super(forwardProcessingBlock, self).__init__()
        self.gpus = gpus
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, in_size, ks, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size, in_size, ks, padding=padding, stride=stride),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.BatchNorm2d(in_size)
        if self.gpus:
            self.conv.cuda()
            self.norm.cuda()

    def forward(self, x):
        conved = self.conv(x)
        res = conved + x
        return self.norm(res)

class SAHead2(nn.Module):
    """SAHead2
    Slight deviation in the architecture
    """
    def __init__(self, in_size, level, gpus, dropout_val, ks=3, stride=1):
        super(SAHead2, self).__init__()
        self.gpus = gpus
        self.conv1 = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, in_size, ks, padding=1, stride=stride),
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
        )
        self.convK = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, in_size, ks, padding=1, stride=stride),
            nn.ReLU(inplace=True)
        )
        self.convQ = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, in_size, ks, padding=1, stride=stride),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()
        self.localization = nn.Sequential(
            nn.Conv2d(in_size, 8, kernel_size=7, padding=3),  # 64**2/level**2 *8
            nn.MaxPool2d(2),  # 32*32*8
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),  # 32*32*16
            nn.MaxPool2d(2),  # 16*16*16
            nn.ReLU(True)
        )
        # W/4 H/4 16
        # W/8 H/8 16
        # tranformation regressor for theta
        self.fc_loc = nn.Sequential(
            nn.Linear(level * 16, 16),
            nn.Linear(16, 3 * 2)
        )
        self.fc_loc[1].weight.data.zero_()
        self.fc_loc[1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        if self.gpus:
            self.conv1.cuda()
            self.fc_loc.cuda()
            self.conv2.cuda()
            self.localization.cuda()
            self.sigmoid.cuda()
            self.convQ.cuda()
            self.convK.cuda()

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size(1) * xs.size(2) * xs.size(3))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, theta

    def rev_stn(self, x, theta: torch.tensor):
        if self.gpus:
            empty_tensor = torch.empty(theta.size()).cuda()
        else:
            empty_tensor = torch.empty(theta.size())
        for val, batch in enumerate(theta):
            if self.gpus:
                new_theta = torch.cat((batch, torch.tensor([[0, 0, 1]]).cuda()), 0)
                theta = new_theta.inverse().cuda()

            else:
                new_theta = torch.cat((batch, torch.tensor([[0, 0, 1]])), 0)
                theta = new_theta.inverse()

            theta = theta[:2, :]
            empty_tensor[val, :, :] = theta
        grid = F.affine_grid(empty_tensor, x.size())
        x = F.grid_sample(x, grid)
        return x
    def forward(self, x):
        upper = self.conv1(x)
        up, theta = self.stn(upper)
        middle = self.conv2(up)
        q = self.convQ(middle)
        k = self.convK(middle)
        mult = q*k
        sum = mult.sum(dim=1, keepdim=True)
        sum = sum/np.sqrt(sum.size(0))
        middle = self.sigmoid(sum.size(0))
        lower = self.conv1(x)
        t = self.rev_stn(middle, theta)
        vals = nn.functional.softmax(t, dim=1)
        return lower * vals

class multiHeadBlock2(nn.Module):
    def __init__(self, num_heads, in_size, level, gpus, dropout_val, ks=3, padding=1, stride=1):
        super().__init__()
        self.gpus = gpus
        self.in_size = in_size
        self.mlist = nn.ModuleList()
        self.level = int(8/level)**2
        for i in range(0, num_heads):
            self.mlist.append(SAHead2(in_size, in_size, self.level, gpus, dropout_val, ks, padding, stride))
        self.conv = SimpleUnetConv(in_size * num_heads, out_size=in_size, stride = 1, gpus=gpus,
                                   dropout_val=dropout_val)
        self.norm = nn.BatchNorm2d(in_size)
        if self.gpus:
            self.mlist.cuda()
            self.conv.cuda()
            self.norm.cuda()

    def forward(self, x):
        e_torch = self.mlist[0](x)
        if self.gpus:
            e_torch.cuda()
        for i in self.mlist[1:]:
            e_torch = torch.cat([e_torch, i(x)], dim=1)
        res = self.conv(e_torch)
        final = res + x
        final = self.norm(final)
        return final

