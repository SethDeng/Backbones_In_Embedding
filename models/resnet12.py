import torch.nn as nn
import torch.nn.functional as F
# Res12Block
class Res12Block(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Res12Block, self).__init__()
        # 1 shotcut + 3 conv in Res12Block
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_1X = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(outchannel)

        self.conv_2X = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(outchannel)

        self.conv_3X = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(outchannel)

    def forward(self, x):
        # x
        residual = x
        residual = self.conv(residual)
        residual = self.bn(residual)
        # print(residual.size(), x.size())

        out = self.conv_1X(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2X(out)
        out = self.bn_2(out)
        out = self.relu(out)

        out = self.conv_3X(out)
        out = self.bn_3(out)

        # F(x) + x
        out = out + residual
        out = self.relu(out)
        out = self.maxpool(out)
        # print(out.size())

        return out

# ResNet12 backbones
class ResNet12(nn.Module):
    def __init__(self, class_num=10, block=Res12Block):
        super(ResNet12, self).__init__()
        channel_nums = [3, 64, 128, 256, 512]

        # self.conv = nn.Conv2d(3, int(channel_nums[0]), kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(channel_nums[0])
        # self.relu = nn.LeakyReLU()

        # 3 - 64
        self.layer1 = self._make_layer(block, channel_nums[0], channel_nums[1])
        # 64 - 128
        self.layer2 = self._make_layer(block, channel_nums[1], channel_nums[2])
        # 128 - 256
        self.layer3 = self._make_layer(block, channel_nums[2], channel_nums[3])
        # 256 - 512
        self.layer4 = self._make_layer(block, channel_nums[3], channel_nums[4])
        # Full connection 
        self.fc = nn.Linear(in_features=channel_nums[4], out_features=class_num)

        # initialize network parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inchannel, outchannel):
        layers = []
        layers.append(block(inchannel, outchannel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = F.avg_pool2d(x, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


