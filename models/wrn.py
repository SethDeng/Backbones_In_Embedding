import torch.nn as nn
import torch.nn.functional as F
# Res12Block
class WRNBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(WRNBlock, self).__init__()
        # 1 shotcut + 3 conv in WRNBlock
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

        return out

# WRN backbones
class WRN(nn.Module):
    def __init__(self, class_num=10, block=WRNBlock):
        super(WRN, self).__init__()
        channel_nums = [3, 128, 512]

        # 3 - 64*2(128)
        self.layer1 = self._make_layer(block, channel_nums[0], channel_nums[1])
        # 64*2 - 64*2*2(512)
        self.layer2 = self._make_layer(block, channel_nums[1], channel_nums[2])
        # Full connection 
        self.fc = nn.Linear(in_features=512, out_features=class_num)

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
        # print(x.size())
        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        out = F.avg_pool2d(x, 8)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        # print(out.size())
        
        return out


