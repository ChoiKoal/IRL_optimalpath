import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class build_c_network_9x9(nn.Module):
    def __init__(self, W, H):
        super(build_c_network_9x9, self).__init__()
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgPool = nn.AdaptiveAvgPool2d((H, W))
        self.linear = conv(512*block.expansion, 1, 1, stride=1, pad=0, bn=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # [1, 1, 81, 81] -> [1, 64, 81, 81]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgPool(out)
        out = self.linear(out).squeeze()  # [1, 512, 9, 9] -> [1, 1, 9, 9] -> [9, 9]
        return out

class build_c_network(nn.Module):
    def __init__(self, image_size = 25, conv_dim = 64):
        super(build_c_network, self).__init__()

        #self.conv0 = conv(1, conv_dim, 5, stride=1, pad=2, bn=True)
        self.conv1 = conv(1, conv_dim*2, 5, stride=2, pad=2,  bn=True)
        self.conv1_1 = conv(conv_dim*2, conv_dim*2, 5, stride=1, pad=1, bn=True)
        self.conv1_2 = conv(conv_dim*2, conv_dim*4, 5, stride=1, pad=0, bn=True)
        #self.conv1_3 = conv(conv_dim*2, conv_dim*2, 5, stride=1, pad=2, bn=True)
        self.conv2 = conv(conv_dim*4, conv_dim*4, 3, stride=1, pad=1, bn=True)
        self.conv2_1 = conv(conv_dim*4, conv_dim*4 , 3, stride=1, pad=1, bn=True)
        self.conv2_2 = conv(conv_dim * 4, conv_dim * 4, 3, stride=1, pad=0, bn=True)
        self.conv3 = conv(conv_dim*4, conv_dim*8, 1, stride=1, pad=0, bn=True)
        self.fc = conv(conv_dim*8, 1, 1, stride = 1,  pad=0, bn=True)



    def forward(self, x):  # If image_size is 64, output shape is as below.
        #out = F.elu(self.conv0(x))
        out = F.elu(self.conv1(x))  # [1, 1, 25, 25] -> [1, 128, 13, 13]
        out = F.elu(self.conv1_1(out))
        out = F.elu(self.conv1_2(out))
        #out = F.elu(self.conv1_3(out))
        out_res = out
        out = F.elu(self.conv2(out))
        out = F.elu(self.conv2_1(out))
        out = out_res + out
        out = F.elu(self.conv2_2(out))
        out = F.elu(self.conv3(out))
        out = F.elu(self.fc(out).squeeze())  # [1, 512, 5, 5] -> [1, 1, 5, 5] -> [5, 5]

        return out


class build_network(nn.Module):
    def __init__(self, W, H):
        super(build_network, self).__init__()
        self.fc0 = nn.Linear(W*W*H*H, 1600)
        self.fc0_0 = nn.Linear(1600, 1200)
        self.fc0_1 = nn.Linear(1200, 800)
        self.fc1 = nn.Linear(800, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 49)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = x.view(1, -1)
        out = F.elu(self.fc0(x))
        out = F.elu(self.fc0_0(out))
        out = F.elu(self.fc0_1(out))
        out = F.elu(self.fc1(out))
        out = F.elu(self.fc2(out))
        out = self.fc3(out)
        return out

class build_network_wideworld(nn.Module):
    def __init__(self, image_size=2401):
        super(build_network_wideworld, self).__init__()
        self.fc0_0 = nn.Linear(image_size, 1600)
        self.fc0 = nn.Linear(1600, 1200)
        self.fc0_2 = nn.Linear(1200, 800)
        self.fc1 = nn.Linear(800, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 49)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = x.view(1, -1)
        out = F.elu(self.fc0_0(x))
        out = F.elu(self.fc0(out))
        out = F.elu(self.fc0_2(out))
        out = F.elu(self.fc1(out))
        out = F.elu(self.fc2(out))
        out = self.fc3(out)
        return out

class build_network_c_wideworld(nn.Module):
    def __init__(self, image_size = 25, conv_dim = 32):
        super(build_network_c_wideworld, self).__init__()

        self.conv0 = conv(1, conv_dim, 9, stride = 1, pad = 4, bn = True) #(49*49)
        self.conv0_0 = conv(conv_dim, conv_dim, 9, stride=2, pad=4, bn=True) # (25*25)
        self.conv1 = conv(conv_dim, conv_dim * 2, 5, stride=2, pad=2, bn=True)
        self.conv1_1 = conv(conv_dim * 2, conv_dim * 2, 5, stride=1, pad=1, bn=True)
        self.conv1_2 = conv(conv_dim * 2, conv_dim * 2, 5, stride=1, pad=0, bn=True)
        self.conv2 = conv(conv_dim * 2, conv_dim * 4, 3, stride=1, pad=1, bn=True)
        # self.conv2_1 = conv(conv_dim, conv_dim, 3, stride=1, pad=0, bn=True)
        self.conv2_2 = conv(conv_dim * 4, conv_dim * 4, 3, stride=1, pad=1, bn=True)
        self.conv2_3 = conv(conv_dim * 4, conv_dim * 4, 3, stride=1, pad=1, bn=True)
        self.conv3 = conv(conv_dim * 4, conv_dim * 8, 1, stride=1, pad=0, bn=True)
        self.fc = conv(conv_dim * 8, 1, 1, stride=1, pad=0, bn=True)

    def forward(self, x):  # If image_size is 64, output shape is as below.
        out = F.elu(self.conv0(x))
        out = F.elu(self.conv0_0(out))
        out = F.elu(self.conv1(out))
        out = F.elu(self.conv1_1(out))
        out = F.elu(self.conv1_2(out))
        out = F.elu(self.conv2(out))
        # out = F.elu(self.conv2_1(out))
        out = F.elu(self.conv2_2(out))
        out = F.elu(self.conv2_3(out))
        out = F.elu(self.conv3(out))
        out = F.elu(self.fc(out).squeeze())

        return out