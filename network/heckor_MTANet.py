import torch
from torch import nn
from torch.nn import functional as F
from .layers import BasicConv3d, FastSmoothSeNormConv3d, RESseNormConv3d, UpConv, TwoAttention


class BaselineUNet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(BaselineUNet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))

        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


class FastSmoothSENormDeepUNet_supervision_skip_no_drop(nn.Module):
    """The model presented in the paper. This model is one of the multiple models that we tried in our experiments
    that it why it has such an awkward name."""

    def __init__(self, in_channels, n_cls, n_filters, reduction=2, return_logits=False):  # 1、2、24
        super(FastSmoothSENormDeepUNet_supervision_skip_no_drop, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters
        self.return_logits = return_logits
        #------------------------------------Encoder0---------------------------------------------------------

        self.block_0_1_1_left = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=7, stride=1, padding=3)
        self.block_0_1_2_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_0_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_0_2_1_left = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_0_2_2_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_0_2_3_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_0_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_0_3_1_left = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_0_3_2_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_0_3_3_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_0_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_0_4_1_left = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_0_4_2_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_0_4_3_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_0_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_0_5_1_left = RESseNormConv3d(8 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_0_5_2_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_0_5_3_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        # ------------------------------------Encoder1---------------------------------------------------------

        self.block_1_1_1_left = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=7, stride=1, padding=3)
        self.block_1_1_2_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_1_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_1_2_1_left = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_2_2_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_1_2_3_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,padding=1)

        self.pool_1_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_1_3_1_left = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_1_3_2_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_1_3_3_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_1_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_1_4_1_left = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_4_2_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_4_3_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)

        self.pool_1_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_1_5_1_left = RESseNormConv3d(8 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_5_2_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_5_3_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        # ------------------------------------Encoder2---------------------------------------------------------

        self.block_2_1_1_left = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=7, stride=1, padding=3)
        self.block_2_1_2_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_2_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_2_2_1_left = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_2_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_3_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_2_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_2_3_1_left = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_3_2_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_3_3_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)

        self.pool_2_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_2_4_1_left = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_2_4_2_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_2_4_3_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_2_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_2_5_1_left = RESseNormConv3d(8 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_2_5_2_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_5_3_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        #-------------------------------------------------------------------------------------------------------------------------

        self.atten_b = TwoAttention(1152)  # 1152
        self.block_b_1_right = FastSmoothSeNormConv3d(1152, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_b_2_right = FastSmoothSeNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.upconv_4 = nn.ConvTranspose3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.atten_4 = TwoAttention(8 * n_filters + 576)  # 768
        self.block_4_1_right = FastSmoothSeNormConv3d((8 + 8 + 8 + 8) * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = FastSmoothSeNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_4 = UpConv(8 * n_filters, n_filters, reduction, scale=8)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.atten_3 = TwoAttention(4 * n_filters + 288)
        self.block_3_1_right = FastSmoothSeNormConv3d((4 + 4 + 4 + 4) * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = FastSmoothSeNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_3 = UpConv(4 * n_filters, n_filters, reduction, scale=4)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.atten_2 = TwoAttention(2 * n_filters + 144)
        self.block_2_1_right = FastSmoothSeNormConv3d((2 + 2 + 2 + 2) * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = FastSmoothSeNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_2 = UpConv(2 * n_filters, n_filters, reduction, scale=2)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, 1 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.atten_1 = TwoAttention(1 * n_filters + 72)
        self.block_1_1_right = FastSmoothSeNormConv3d((1 + 1 + 1 + 1) * n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = FastSmoothSeNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        # self.conv1x1 = nn.Conv3d(1 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

        self.conv1x1 = nn.Conv3d(1 * n_filters, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        temx = x[:, 0, :, :, :]
        temy = x[:, 1, :, :, :]
        temz = x[:, 2, :, :, :]
        temx = temx.unsqueeze(1)
        temy = temy.unsqueeze(1)
        temz = temz.unsqueeze(1)

        ds0_0 = self.block_0_1_2_left(self.block_0_1_1_left(temx))  # 24,96,96
        ds1_0 = self.block_0_2_3_left(self.block_0_2_2_left(self.block_0_2_1_left(self.pool_0_1(ds0_0))))  # 48,48,48
        ds2_0 = self.block_0_3_3_left(self.block_0_3_2_left(self.block_0_3_1_left(self.pool_0_2(ds1_0))))  # 96,24,24
        ds3_0 = self.block_0_4_3_left(self.block_0_4_2_left(self.block_0_4_1_left(self.pool_0_3(ds2_0))))  # 192,12,12
        x_0 = self.block_0_5_3_left(self.block_0_5_2_left(self.block_0_5_1_left(self.pool_0_4(ds3_0))))   # 384,6,6,6

        ds0_1 = self.block_1_1_2_left(self.block_1_1_1_left(temy))
        ds1_1 = self.block_1_2_3_left(self.block_1_2_2_left(self.block_1_2_1_left(self.pool_1_1(ds0_1))))
        ds2_1 = self.block_1_3_3_left(self.block_1_3_2_left(self.block_1_3_1_left(self.pool_1_2(ds1_1))))
        ds3_1 = self.block_1_4_3_left(self.block_1_4_2_left(self.block_1_4_1_left(self.pool_1_3(ds2_1))))
        x_1 = self.block_1_5_3_left(self.block_1_5_2_left(self.block_1_5_1_left(self.pool_1_4(ds3_1))))

        ds0_2 = self.block_2_1_2_left(self.block_2_1_1_left(temz))
        ds1_2 = self.block_2_2_3_left(self.block_2_2_2_left(self.block_2_2_1_left(self.pool_2_1(ds0_2))))
        ds2_2 = self.block_2_3_3_left(self.block_2_3_2_left(self.block_2_3_1_left(self.pool_2_2(ds1_2))))
        ds3_2 = self.block_2_4_3_left(self.block_2_4_2_left(self.block_2_4_1_left(self.pool_2_3(ds2_2))))
        x_2 = self.block_2_5_3_left(self.block_2_5_2_left(self.block_2_5_1_left(self.pool_2_4(ds3_2))))

        out4 = torch.cat([x_0, x_1, x_2], dim=1)  # 1152,6,6,6
        out3 = torch.cat([ds3_0, ds3_1, ds3_2], dim=1)  # 576,12,12
        out2 = torch.cat([ds2_0, ds2_1, ds2_2], dim=1)  # 288,24,24,24
        out1 = torch.cat([ds1_0, ds1_1, ds1_2], dim=1)  # 144,48,48,48
        out0 = torch.cat([ds0_0, ds0_1, ds0_2], dim=1)  # 72,96,96,96

        out4 = self.block_b_2_right(self.block_b_1_right(self.atten_b(out4)))

        x = self.block_4_2_right(self.block_4_1_right(self.atten_4(torch.cat([self.upconv_4(out4), out3], 1))))
        sv4 = self.vision_4(x)  # 按图：sv4 = self.vision_4(self.atten_4(torch.cat([self.upconv_4(out4), out3], 1)))

        x = self.block_3_2_right(self.block_3_1_right(self.atten_3(torch.cat([self.upconv_3(x), out2], 1))))
        sv3 = self.vision_3(x)

        x = self.block_2_2_right(self.block_2_1_right(self.atten_2(torch.cat([self.upconv_2(x), out1], 1))))
        sv2 = self.vision_2(x)

        x = self.block_1_1_right(self.atten_1(torch.cat([self.upconv_1(x), out0], 1)))
        x = x + sv4 + sv3 + sv2     # 消融实验：去深度监督模块
        x = self.block_1_2_right(x)

        x = self.conv1x1(x)
        return x

        '''
        # 消融实验：去融合模块
        out4 = self.block_b_2_right(self.block_b_1_right(out4))

        x = self.block_4_2_right(self.block_4_1_right((torch.cat([self.upconv_4(out4), out3], 1))))
        sv4 = self.vision_4(x)  # 按图：sv4 = self.vision_4(self.atten_4(torch.cat([self.upconv_4(out4), out3], 1)))

        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), out2], 1)))
        sv3 = self.vision_3(x)

        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), out1], 1)))
        sv2 = self.vision_2(x)

        x = self.block_1_1_right(torch.cat([self.upconv_1(x), out0], 1))
        x = x + sv4 + sv3 + sv2
        x = self.block_1_2_right(x)

        x = self.conv1x1(x)
        return x
        '''

        # if self.return_logits:
        #     return x
        # else:
        #     if self.n_cls == 1:
        #         return torch.sigmoid(x)
        #     else:
        #         return F.softmax(x, dim=1)
