import torch
import torch.nn as nn
from torch_geometric.nn import GroupNorm3d
import torch.nn.functional as F

# 用于存储类中特征
f1, f2, f3, f4, f5, f6, f7 = 0

# 定义Spatial Pyramid Pooling层
class SpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, output_size, num_levels):
        super(SpatialPyramidPooling, self).__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.output_size = output_size
        self.pool_channels = in_channels // num_levels

        self.pools = []
        for i in range(num_levels):
            pool_size = (output_size, output_size, output_size)
            self.pools.append(nn.AdaptiveAvgPool3d(output_size))
        self.pools = nn.ModuleList(self.pools)

    def forward(self, x):
        outputs = []
        for i in range(self.num_levels):
            outputs.append(self.pools[i](x))
        spp_output = torch.cat(outputs, dim=1)
        return spp_output

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # 编码器部分
        self.encoder1 = self.conv_block(in_channels, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)

        # 中间层
        self.middle = self.conv_block(128, 256)

        # 分割任务解码器部分
        self.decoder3 = self.conv_block(256, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)

        # 分割任务输出层
        self.seg_output_layer = nn.Conv3d(64, out_channels, kernel_size=1)

        # 分类的SPP（Spatial Pyramid Pooling）和全连接层
        # 分类的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        # 分类任务输出层
        self.encoder4 = self.conv_block(256, 256)
        self.encoder5 = self.conv_block(256, 128)
        self.spp_layer = self.SpatialPyramidPooling(128, 64, 3)    # spp_channels = 64, num_level=3


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            GroupNorm3d(out_channels, 16),  # 使用Group Normalization替代BN，16是分组数量，可以根据需要调整
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            GroupNorm3d(out_channels, 16),
            nn.ReLU(inplace=True),
            MRAModule(out_channels)  # 添加 MRAModule
        )

    def forward(self, x):
        # 编码器部分
        global f1, f2, f3, f4, f5, f6, f7, f8
        enc1 = self.encoder1(x)
        if(f1 == 0):
            f1 = CSFFModule(x, enc1)
        elif enc1.shape == f1.shape:
            enc1 = enc1 + f1

        enc2 = self.encoder2(enc1)
        if(f2 == 0):
            f2 = CSFFModule(enc1, enc2)
        elif enc2.shape == f2.shape:
            enc2 = enc2 + f2

        enc3 = self.encoder3(enc2)
        if(f3 == 0):
            f3 = CSFFModule(enc2, enc3)
        elif enc3.shape == f3.shape:
            enc3 = enc3 + f3

        # enc4 = self.encoder4(enc3)
        # if(f4 == 0):
        #     f4 = CSFFModule(enc3, enc4)
        # elif enc4.shape == f4.shape:
        #     enc4 = enc4 + f4

        # 中间层
        middle = self.middle(enc3)

        # 分割解码器部分
        dec3 = self.decoder3(torch.cat([enc3, middle], 1))
        if (f6 == 0):
            f6 = CSFFModule(torch.cat([enc3, middle], 1), dec3)
        elif dec3.shape == f6.shape:
            dec3 = dec3 + f6

        dec2 = self.decoder2(torch.cat([enc2, dec3], 1))
        if (f7 == 0):
            f7 = CSFFModule(torch.cat([enc2, dec3], 1), dec2)
        elif dec3.shape == f7.shape:
            dec2 = dec2 + f7

        dec1 = self.decoder1(torch.cat([enc1, dec2], 1))
        if (f8 == 0):
            f8 = CSFFModule(torch.cat([enc1, dec2], 1), dec1)
        elif dec1.shape == f8.shape:
            dec1 = dec1 + f8

        # 分割图像的输出
        seg_output = self.seg_output_layer(dec1)

        # 分类编码器、分类的SPP和全连接层
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder4(enc4)
        spp_output = self.spp_layer(enc5)
        classification_output = self.classifier(spp_output)

        return seg_output, classification_output


# 创建级联模型
class CascadeUNet(nn.Module):
    def __init__(self, model1, model2):
        super(CascadeUNet, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        # 第一个模型的前向传播
        output1 = self.model1(x)
        concatenated_input = torch.cat([x, output1], dim=1)

        # 将第一个模型的输出作为第二个模型的输入
        output2 = self.model2(concatenated_input)

        return output2

# 定义Dice损失函数
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.0
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        return 1.0 - dice  # 返回Dice损失，1减去Dice系数

# 多尺度聚合
class MultiscaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiscaleFusion, self).__init()
        # Define the convolution layers for each scale
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding=3)

    def forward(self, x):
        # Apply convolution for each scale
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)

        # Concatenate the results along the channel dimension
        fused_feature = torch.cat((out1, out3, out5, out7), dim=1)

        return fused_feature

# 注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(pool)
        attention = torch.sigmoid(attention)
        return attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        attention = avg_out + max_out
        return attention

class MRAModule(nn.Module):
    def __init__(self, in_channels):
        super(MRAModule, self).__init__()
        self.spatial_attention = SpatialAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)
        self.multiscale_fusion = MultiscaleFusion(in_channels, in_channels)

    def forward(self, x):
        fused_feature = self.multiscale_fusion(x)
        # Apply spatial and channel attention
        spatial_attention = self.spatial_attention(fused_feature)
        channel_attention = self.channel_attention(spatial_attention)
        out = fused_feature + fused_feature * channel_attention

        return out


class CSFFModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSFFModule, self).__init()

        # 第零步：该变F_out的通道数
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 第一步: 通道拼接
        self.concat_layer = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)

        # 第二步: 1x1卷积层
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, padding=0),
            nn.GroupNorm(1, in_channels // 8),  # 使用GroupNorm，调整group的数量适应您的数据
            nn.ReLU(inplace=True)
        )

        # 第三步: 3x3卷积层
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=3, padding=1),
            nn.GroupNorm(1, in_channels // 8),  # 使用GroupNorm，调整group的数量适应您的数据
            nn.ReLU(inplace=True)
        )

        # 第四步: 3x3卷积层
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels // 8, 2, kernel_size=3, padding=1),
            nn.GroupNorm(1, 2),  # 使用GroupNorm，调整group的数量适应您的数据
            nn.ReLU(inplace=True)
        )

        # 第五步: Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, F_in, F_out):
        # 第零步: 改变F_out的通道数
        F_out = self.conv1x1(F_out)
        # 第一步: 通道拼接
        F_c = torch.cat((F_in, F_out), dim=1)
        F_c = self.concat_layer(F_c)

        # 第二步: 1x1卷积层
        F_c = self.conv1x1_2(F_c)

        # 第三步: 3x3卷积层
        F_c = self.conv3x3(F_c)

        # 第四步: 3x3卷积层
        F_c = self.conv3x3_2(F_c)

        # 第五步: Sigmoid激活函数
        F_c = self.sigmoid(F_c)

        # 获取两个通道
        w1 = F_c[:, 0:1, :, :]
        w2 = F_c[:, 1:2, :, :]

        # 输出融合后的特征
        F_first = F_in * w1 + F_out * w2

        return F_first
