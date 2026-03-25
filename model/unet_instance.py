import sys
#sys.modules.pop("model.model_components")
from model.model_components import *
from model.unet_super import UnetSuper
from utils import weights_init
import sys


class Unet(UnetSuper):
    """Unet

    Basic Unet which is used for medical image segmentation and classification
    original paper: https://arxiv.org/pdf/1505.04597
    """
    def __init__(self, hparams, input_channels, min_filter=16, is_deconv=False, is_batchnorm=True, on_gpu=False, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [32, 64, 128, 256]
        self.conv1 = UnetConv(self.in_channels, filters[0], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv2 = UnetConv(filters[0], filters[1], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv3 = UnetConv(filters[1], filters[2], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.center = UnetConv(filters[2], filters[3], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        # upsampling
        self.up_concat3 = UnetUp(filters[3], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat2 = UnetUp(filters[2], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat1 = UnetUp(filters[1], filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], kwargs["num_classes"], 1)
        if on_gpu:
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.center.cuda()
            self.up_concat3.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()
        self.apply(weights_init)

    def forward(self, inputs):
        maxpool = nn.MaxPool2d(kernel_size=2)
        conv1 = self.conv1(inputs)  # 16*256*256
        maxpool1 = maxpool(conv1)  # 16*128*128

        conv2 = self.conv2(maxpool1)  # 32*128*128
        maxpool2 = maxpool(conv2)  # 32*64*64

        conv3 = self.conv3(maxpool2)  # 64*64*64
        maxpool3 = maxpool(conv3)  # 64*32*32

        center = self.center(maxpool3)

        up3 = self.up_concat3(center, conv3)  # 64*64*64
        up2 = self.up_concat2(up3, conv2)  # 32*128*128
        up1 = self.up_concat1(up2, conv1)  # 16*256*256

        final = self.final(up1)
        finalize = nn.functional.softmax(final, dim=1)
        return finalize


    def print(self, args: torch.Tensor) -> None:
        print(args)


class UneXt(UnetSuper):
    """UneXt

    U-Net architecture with alterations inspired from swin transformer derived from
    http://arxiv.org/abs/2201.03545 (describes a ResNet block inspired by the swin stransformer)
    We have adapted the idea to fit U-net architecture
    """
    def __init__(self, hparams, input_channels, min_filter=16, on_gpu=False, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.in_channels = input_channels
        self.input = input_channels
        filters = [32, 64, 128, 256]

        # encoder
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, filters[0], kernel_size=3, stride=1, padding=1))
        self.conv1 = UneXtConv(filters[0],gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv2 = UneXtConv(filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv3 = nn.Sequential(
            UneXtConv(filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"]),
            UneXtConv(filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"]),
            UneXtConv(filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"]))

        # downsampling
        self.down1 = UneXtDown(filters[0], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.down2 = UneXtDown(filters[1], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.down3 = UneXtDown(filters[2], filters[3], gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        # inverted bottleneck
        self.center = UneXtConv(filters[3], gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        # upsampling
        self.up_concat3 = UneXtUp(filters[3], filters[2], gpus=on_gpu, is_third=True, dropout_val=kwargs["dropout_val"])
        self.up_concat2 = UneXtUp(filters[2], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat1 = UneXtUp(filters[1], filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        # final conv (without any concat)
        self.final = nn.Sequential(
            nn.GroupNorm(1, filters[0]),
            nn.Conv2d(filters[0], kwargs["num_classes"], 1))

        if on_gpu:
            self.stem.cuda()
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.down1.cuda()
            self.down2.cuda()
            self.down3.cuda()
            self.center.cuda()
            self.up_concat3.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()

        self.apply(weights_init)

    def forward(self, inputs):

        stem = self.stem(inputs) # 16*256*256
        conv1 = self.conv1(stem)  # 16*256*256
        down1 = self.down1(conv1)  # 32*128*128

        conv2 = self.conv2(down1)  # 32*128*128
        down2 = self.down2(conv2)  # 64*64*64

        conv3 = self.conv3(down2)  # 64*64*64
        down3 = self.down3(conv3)  # 128*32*32

        center = self.center(down3) # 128*32*32

        up3 = self.up_concat3(center, conv3)  # 64*64*64
        up2 = self.up_concat2(up3, conv2)  # 32*128*128
        up1 = self.up_concat1(up2, conv1)  # 16*256*256

        final = self.final(up1)
        finalize = nn.functional.softmax(final, dim=1)

        return finalize


    def print(self, args: torch.Tensor) -> None:
        print(args)


class swinUNETR(UnetSuper):

    def __init__(self, hparams, input_channels, min_filter=16, on_gpu=False, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.in_channels = input_channels
        filters = [32, 64, 128, 256]

        self.encoder = SwinUnetrEnc(input_channels, filters[0])

        self.up_concat3 = UnetUp(filters[3], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat2 = UnetUp(filters[2], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat1 = UnetUp(filters[1], filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        self.final = nn.Sequential(
            nn.Conv2d(filters[0], kwargs["num_classes"], kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # upsample to match 256x256
        )

        if on_gpu:
            self.encoder.cuda()
            self.up_concat3.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()

        self.apply(weights_init)

    def forward(self, inputs):
        enc1, enc2, enc3, enc4 = self.encoder(inputs)

        up3 = self.up_concat3(enc4, enc3)  # 64*64*64
        up2 = self.up_concat2(up3, enc2)  # 32*128*128
        up1 = self.up_concat1(up2, enc1)  # 16*256*256

        final = self.final(up1)
        finalize = nn.functional.softmax(final, dim=1)

        return finalize


#### ==== model with spatial transformer ==== ####
class RTUnet(UnetSuper):
    """RTUnet

    A Unet with a spatial transformer network at the beginning
    Does not produce intended outcome
    """
    def __init__(self, hparams, input_channels, min_filter, is_deconv=True, is_batchnorm=True, on_gpu=False, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [min_filter, min_filter * 2, min_filter * 4, min_filter * 8]
        self.head1 = multiHeadBlock(2, input_channels, 1,  gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd1 = forwardProcessingBlock(input_channels,  gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv1 = UnetConv(input_channels, filters[0], is_batchnorm=True, gpus=on_gpu, dropout_val=kwargs[
            "dropout_val"])
        self.head2 = multiHeadBlock(2, filters[0], 2, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd2 = forwardProcessingBlock(filters[0],  gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv2 = UnetConv(filters[0], filters[1], is_batchnorm=True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.head3 = multiHeadBlock(2, filters[1], 3, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd3 = forwardProcessingBlock(filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv3 = UnetConv(filters[1], filters[2], is_batchnorm=True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])


        # upsampling
        self.up_concat3 = UnetUp(filters[3], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat2 = UnetUp(filters[2], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat1 = UnetUp(filters[1], filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], 7, 1)
        if on_gpu:
            self.head1.cuda()
            self.head2.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.center.cuda()
            self.up_concat3.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()
        self.apply(weights_init)

    def forward(self, inputs: torch.Tensor):
        maxpool = nn.MaxPool2d(kernel_size=2)
        x_s = torch.chunk(inputs, 8, 2)
        along_x = []
        for i in x_s:
            chunks = torch.chunk(i, 8, 3)
            along_x.append(chunks)
        merge_x = []
        for chunks in along_x:
            merge_y = []
            for chunk in chunks:
                x1 = self.head1(chunk)  # 16*64*64
                y1 = self.fwd1(x1)
                z1 = self.conv1(y1)
                maxpool1 = maxpool(z1)  # 16*32*32

                x2 = self.head2(maxpool1)  # 16*64*64
                y2 = self.fwd2(x2)
                z2 = self.conv2(y2)
                maxpool2 = maxpool(z2)  # 32*16*16

                x3 = self.head3(maxpool2)  # 16*64*64
                y3 = self.fwd3(x3)
                z3 = self.conv3(y3)

                up2 = self.up_concat2(z3, z2)  # 32*32*32
                up1 = self.up_concat1(up2, z1)  # 16*64*64

                final = self.final(up1)
                finalize = nn.functional.softmax(final, dim=1)
                merge_y.append(finalize)
            merge_x.append(torch.cat(merge_y, 3))
        return torch.cat(merge_x, 2)



#### ==== Context Unet ==== ####
class ContextUnet(UnetSuper):
    """Context Unet is a U-Net with added context modules and localization modules and a different way of generating
    the higher dimension feature maps. Additionally deep_supervision elements are present, however not meaningfully
    better

    """
    def __init__(self, hparams, input_channels, is_deconv=True, is_batchnorm=True, on_gpu=False,
                 deep_supervision=True, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.deep_supervision = deep_supervision
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [16, 32, 64, 128, 256]
        self.conv1 = SimpleUnetConv(self.in_channels, filters[0], stride=1, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context1 = ContextModule(filters[0], filters[0],gpus=on_gpu)
        self.ttt2 = SimpleUnetConv(filters[0], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context2 = ContextModule(filters[1], filters[1], gpus=on_gpu)
        self.ttt3 = SimpleUnetConv(filters[1], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context3 = ContextModule(filters[2], filters[2],  gpus=on_gpu)
        self.ttt4 = SimpleUnetConv(filters[2], filters[3], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context4 = ContextModule(filters[3], filters[3], gpus=on_gpu)
        self.up_center = SimpleUnetUp(filters[3], filters[2], gpus=on_gpu)
        self.local1 = Localization(filters[3], filters[2], gpus=on_gpu)
        self.up1 = SimpleUnetUp(filters[2], filters[1], gpus=on_gpu)
        self.local2 = Localization(filters[2], filters[1], gpus=on_gpu)
        self.up2 = SimpleUnetUp(filters[1], filters[0], gpus=on_gpu)
        self.final = nn.Conv2d(filters[1], kwargs["num_classes"], 1)
        self.seg = SegmentationLayer(64, 32, 7, gpus=on_gpu)
        self.apply(weights_init)
        if on_gpu:
            self.conv1.cuda()
            self.context1.cuda()
            self.context2.cuda()
            self.context3.cuda()
            self.context4.cuda()
            self.ttt2.cuda()
            self.ttt3.cuda()
            self.ttt4.cuda()
            self.up_center.cuda()
            self.local1.cuda()
            self.local2.cuda()
            self.up1.cuda()
            self.up2.cuda()
            self.final.cuda()

    def forward(self, x):
        con1 = self.conv1(x) # 16*256*256
        son1 = self.context1(con1)
        plus1 = con1 + son1

        con2 = self.ttt2(plus1) # 32*128*128
        son2 = self.context2(con2)
        plus2 = con2 + son2

        con3 = self.ttt3(plus2) # 64*64*64
        son3 = self.context3(con3)
        plus3 = con3+son3

        con4 = self.ttt4(plus3) # 128*32*32
        son4 = self.context4(con4)
        plus4 = con4 + son4

        up_center = self.up_center(plus4) #64*64*64

        comb = torch.cat([plus3, up_center], dim=1) #128*64*64
        local1 = self.local1(comb) #64*64*64
        up1 = self.up1(local1) #32*128*128

        comb = torch.cat([plus2, up1], dim=1) #64*128*128
        local2 = self.local2(comb) #32*128*128
        up2 = self.up2(local2) #16*256*256

        comb = torch.cat([plus1, up2], dim=1) #32*256*256
        final = self.final(comb)  #7*256*256

        if self.deep_supervision:
            final = self.seg(local1, local2, final)

        return nn.functional.softmax(final, dim=1) #1*256*256

class ArchitectureOption3(UnetSuper):
    """Unet

    Basic Unet which is used for medical image segmentation and classification
    original paper: https://arxiv.org/pdf/1505.04597
    """
    def __init__(self, hparams, input_channels, is_deconv=True, is_batchnorm=False, on_gpu=False, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [8, 16, 32, 64]
        self.conv1 = UnetConv(self.in_channels, filters[0], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.multiHead1 = multiHeadBlock2(2, filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd1 = forwardProcessingBlock(filters[0],  gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv2 = UnetConv(filters[0], filters[1], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.multiHead2 = multiHeadBlock2(2, filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd2 = forwardProcessingBlock(filters[1],  gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        # upsampling
        self.up_concat2 = UnetUp(filters[2], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat1 = UnetUp(filters[1], filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], kwargs["num_classes"], 1)
        if on_gpu:
            self.conv1.cuda()
            self.multiHead1.cuda()
            self.conv2.cuda()
            self.multiHead2.cuda()
            self.conv3.cuda()
            self.fwd1.cuda()
            self.fwd2.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()
        self.apply(weights_init)

    def forward(self, inputs):
        maxpool = nn.MaxPool2d(kernel_size=2)
        conv1 = self.conv1(inputs)  # 16*256*256
        maxpool1 = maxpool(conv1)  # 16*128*128
        skip1 = self.multiHead1(conv1)
        skip1 = self.fwd1(skip1)

        conv2 = self.conv2(maxpool1)  # 32*128*128
        maxpool2 = maxpool(conv2)  # 32*64*64
        skip2 = self.multiHead1(conv2)
        skip2 = self.fwd1(skip2)

        conv3 = self.conv3(maxpool2)  # 64*64*64

        up3 = self.up_concat3(conv2)  # 64*64*64
        up2 = self.up_concat2(up3, skip2)  # 32*128*128
        up1 = self.up_concat1(up2, skip1)  # 16*256*256

        final = self.final(up1)
        finalize = nn.functional.softmax(final, dim=1)
        return finalize


    def print(self, args: torch.Tensor) -> None:
        print(args)

class SkipNet(UnetSuper):
    def __init__(self, hparams, input_channels, is_deconv=True, is_batchnorm=False, on_gpu=False, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.in_channels = input_channels
        self.gpu = on_gpu
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [16, 32, 64, 128]
        self.placeholder = False
        self.conv11 = UnetConv(self.in_channels, filters[0], is_batchnorm, gpus=on_gpu, dropout_val=kwargs[
            "dropout_val"])
        self.multiHead1 = multiHeadBlock2(2, filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd1 = forwardProcessingBlock(filters[0],  gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv12 = UnetConv(self.in_channels, filters[0], is_batchnorm, gpus=on_gpu, dropout_val=kwargs[
            "dropout_val"])
        self.conv21 = UnetConv(filters[0], filters[1], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.multiHead2 = multiHeadBlock2(2, filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd2 = forwardProcessingBlock(filters[1],  gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv22 = UnetConv(filters[0], filters[1], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv31 = UnetConv(filters[1], filters[2], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.multiHead3 = multiHeadBlock2(2, filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd3 = forwardProcessingBlock(filters[2], gpu=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv32 = UnetConv(filters[1], filters[2], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        # upsampling
        self.up_concat2 = SimpleUnetUp(filters[2], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_mh2 = multiHeadBlock2(2, filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_fwd2 = forwardProcessingBlock(filters[1],  gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_conv2 = SimpleUnetConv(filters[1], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"], stride=1)
        self.up_concat1 = SimpleUnetUp(filters[1], filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_mh1 = multiHeadBlock2(2, filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_fwd1 = forwardProcessingBlock(filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_conv1 = SimpleUnetConv(filters[1], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"], stride=1)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], kwargs["num_classes"], 1)
        if on_gpu:
            self.conv11.cuda()
            self.conv12.cuda()
            self.multiHead1.cuda()
            self.conv21.cuda()
            self.conv22.cuda()
            self.multiHead2.cuda()
            self.conv31.cuda()
            self.conv32.cuda()
            self.fwd1.cuda()
            self.fwd2.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()
            self.multiHead3.cuda()
            self.fwd3.cuda()
            self.up_mh1.cuda()
            self.up_mh2.cuda()
            self.up_fwd1.cuda()
            self.up_fwd2.cuda()
        self.apply(weights_init)

    def forward(self, inputs):
        maxpool = nn.MaxPool2d(kernel_size=2)
        ### Start Encoder
        norm1 = nn.BatchNorm2D(self.filters[0])
        norm2 = nn.BatchNorm2D(self.filters[1])
        norm3 = nn.BatchNorm2D(self.filters[2])
        if self.gpu:
            norm1.cuda()
            norm2.cuda()
            norm3.cuda()
        ### First Level
        level1_step1 = self.conv11(inputs)
        if not self.placeholder:
            level1_step2 = self.multiHead1(level1_step1)
            level1_step3 = self.fwd1(level1_step2)
            level1_step4 = norm1(self.conv12(level1_step3) + level1_step1)
        else:
            level1_step4 = norm1(self.conv12(level1_step1) + level1_step1)

        ### Second Level
        level2_input = maxpool(level1_step4)
        level2_step1 = self.conv21(level2_input)
        if not self.placeholder:
            level2_step2 = self.multiHead2(level2_step1)
            level2_step3 = self.fwd2(level2_step2)
            level2_step4 = norm2(self.conv22(level2_step3) + level2_step1)
        else:
            level2_step4 = norm2(self.conv22(level2_step1) + level2_step1)

        ### Third level
        level3_input = maxpool(level2_step4)
        level3_step1 = self.conv31(level3_input)
        if not self.placeholder:
            level3_step2 = self.multiHead3(level3_step1)
            level3_step3 = self.fwd3(level3_step2)
            level3_step4 = norm3(self.conv32(level3_step3) + level3_step1)
        else:
            level3_step4 = norm3(self.conv32(level3_step1) + level3_step1)

        ### End Encoder
        downsampling_out = level3_step4
        norm_up2 = nn.BatchNorm2D(self.filters[1])
        norm_up1 = nn.BatchNorm2D(self.filters[0])

        ###Start Decoder
        ### Second Level
        up2 = self.up_concat2(downsampling_out, level2_step4)
        if not self.placeholder:
            up2_step2 = self.up_mh2(up2)
            up2_step3 = self.up_fwd2(up2_step2)
            up2_step4 = self.up_conv2(up2_step3) + up2
        else:
            up2_step4 = self.up_conv2(up2) + up2
        up2_out = norm_up2(up2_step4)

        ### First Level
        up1 = self.up_concat1(up2_out, level1_step4)
        if not self.placeholder:
            up1_step2 = self.up_mh1(up1)
            up1_step3 = self.up_fwd1(up1_step2)
            up1_step4 = self.up_conv1(up1_step3) + up1
        else:
            up1_step4 = self.up_conv1(up1) + up1
        up1_out = norm_up1(up1_step4)

        return nn.functional.softmax(self.final(up1_out), dim=1)

