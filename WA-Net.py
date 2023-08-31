import torch.nn as nn
import torch
from DWAN_lwt.dwtmodel.DWT_IDWT.DWT_IDWT_layer import DWT_2D
from DWAN_lwt.dwtmodel.waveletpro import Downsamplewave,Downsamplewave1
import torch.nn.functional as F

from torchvision.transforms import transforms

class SELayer(nn.Module):
    def __init__(self,in_channel,reduction): # reduction 减少
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel,in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel),
            nn.Sigmoid()
        )



    def forward(self,x): # torch.Size([1, 256, 64, 64])
        b,c,_,_ = x.size()
        # x1 = self.avg_pool(x) # torch.Size([1, 256, 1, 1])
        x2 = self.avg_pool(x).view(b,c) # torch.Size([1, 256])
        # print(x2.shape)
        # y1 = self.fc(x2) # torch.Size([1, 256])
        y2 = self.fc(x2).view(b,c,1,1)  # torch.Size([1, 256, 1, 1])
        # print(y2)
        y3 = y2.expand_as(x)
        # print(y3)
        y4 = x * y3
        # print(y4.shape) # torch.Size([1, 256, 64, 64])
        return y4

class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()
        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()
        # self.dwt = DWT_2D("haar")
    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        # print("x_maxpool",x_maxpool.shape)
        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # print("x_avgpool",x_avgpool.shape)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x
        return outputs

class Bottleneck(nn.Module):
    def __init__(self,in_channel,filters,stride=2):
        super(Bottleneck, self).__init__()
        c1,c2,c3 = filters
        self.out_channels = c3
        self.in_channel = in_channel
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=c1,
                               kernel_size=1, stride=stride, padding=0,bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(c1)
        # -----------------------------------------
        self.dwt = DWT_2D("haar")
        self.stride = stride

        self.conv2 = nn.Conv2d(in_channels=c1*4, out_channels=c2,
                               kernel_size=3, stride=1, bias=False, padding=1)  # unsqueeze channels
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=c3,
                               kernel_size=1, stride=1, bias=False,padding=0)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(c3)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c3, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(c3)
        )
        self.avg_conv = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(in_channels=in_channel, out_channels=c3, kernel_size=1),
                                      nn.BatchNorm2d(c3))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c3, kernel_size=1),
            nn.BatchNorm2d(c3)
        )

        self.dwt = DWT_2D("haar")
        self.conv_dwt = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c3, kernel_size=1),
            nn.BatchNorm2d(c3),
            nn.ReLU()
        )

        self.relu = nn.ReLU(inplace=True)

        self.se = SELayer(c3, 16)


    def forward(self,x):
        # print("x in:",x.shape)
        identity = x
        if self.downsample is not None:
            if self.stride == 1:
                identity = self.conv(identity)
            else:
                # identity = self.downsample(x)
                identity = self.avg_conv(x)
                # identity, _, _, _ = self.dwt(x)
                # identity = self.conv_dwt(identity)
        # print("downsample:", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        # print("conv1:", x.shape)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        ll, lh, hl, hh = self.dwt(x)
        x = torch.cat((ll, lh, hl, hh), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print("conv2:", x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print("conv3:", x.shape)
        x = self.se(x)
        # print("se:", x.shape)
        x += identity
        x = self.relu(x)
        # print("x out:", x.shape)

        return x

class Waveletatt(nn.Module):
    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution

        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave1(wavename=wavename)])
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x   #xori: torch.Size([1, 256, 56, 56])
        # print("xori:",xori.shape)
        B, C, H, W= x.shape   #B, C, H, W: 1 256 56 56
        # print("B, C, H, W:", B, C, H, W)
        x = x.view(B, H, W, C)#shape,位置互换  x.view: torch.Size([1, 56, 56, 256])
        # print("x.view:", x.shape)
        x = x.permute(0, 3, 2, 1)  #x.permute: torch.Size([1, 256, 56, 56])
        # print("x.permute:", x.shape)
        y = self.downsamplewavelet(x)  #y: torch.Size([1, 256])
        # print("y:", y.shape)
        y = self.fc(y).view(B, C, 1, 1)  # y: torch.Size([1, 256, 1, 1])
        # print("y:",y.shape)
        y = xori * y.expand_as(xori) #y: torch.Size([1, 256, 56, 56])
        # print("y:", y.shape)
        return y

class Waveletattspace(nn.Module):
    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution

        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave(wavename=wavename)])
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes*2, in_planes//2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//2, in_planes,kernel_size=1,padding= 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x
        B, C, H, W= x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        y = self.downsamplewavelet(x)
        # print("y,",y.shape)
        y = self.fc(y) # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        y = xori * y.expand_as(xori)
        return y

class SEResNet(nn.Module):
    def __init__(self,num_class):
        super(SEResNet, self).__init__()
        self.channels = 64

        self.stage1 = nn.Sequential(
            nn.Conv2d(3,self.channels,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )

        self.stage2 = nn.Sequential(
            Bottleneck(self.channels,[64,64,256],stride=1),
            Bottleneck(256,[64,64,256],stride=1),
            Bottleneck(256, [64, 64, 256], stride=1),
            # SELayer(256,16)
        )
        self.att1 = Waveletatt(in_planes=256)
        self.attspace1 = Waveletattspace(in_planes=256)
        # self.se = SELayer(256,16)
        # self.space=spatial_attention()

        self.stage3 = nn.Sequential(
            Bottleneck(256,[128,128,512],stride=2),
            Bottleneck(512, [128, 128, 512], stride=1),
            Bottleneck(512, [128, 128, 512], stride=1),
            Bottleneck(512, [128, 128, 512], stride=1),
            # SELayer(512,32)
        )

        self.stage4 = nn.Sequential(
            Bottleneck(512, [256, 256, 1024], stride=2),
            Bottleneck(1024, [256, 256, 1024], stride=1),
            Bottleneck(1024, [256, 256, 1024], stride=1),
            Bottleneck(1024, [256, 256, 1024], stride=1),
            Bottleneck(1024, [256, 256, 1024], stride=1),
            Bottleneck(1024, [256, 256, 1024], stride=1),
            # SELayer(1024,64)
        )

        self.stage5 = nn.Sequential(
            Bottleneck(1024,[512,512,2048],stride=2),
            Bottleneck(2048,[512,512,2048],stride=1),
            Bottleneck(2048, [512, 512, 2048], stride=1),
            # SELayer(2048,128)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


        self.fc = nn.Sequential(
            nn.Linear(2048,num_class)
        )
    def forward(self,x):

        x = self.stage1(x)
        # x = self.stage2(x)

        x_ = self.stage2(x)
        x = self.att1(x_)

        # x = self.att1(x_)
        x = self.attspace1(x)
        # x= self.se(x)
        # x=self.space(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        # 1 * 2048 * 7 * 7
        x = self.avgpool(x)
        x = x.view(x.shape[0],2048) # 2048
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, x_


if __name__ == '__main__':
    x = torch.randn(1,3,224,224).cuda()
    resnet50 = SEResNet(730).cuda()
    print(resnet50)
    y = resnet50(x)
    print(y.shape)
