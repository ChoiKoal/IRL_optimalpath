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

class build_c_network(nn.Module):
    def __init__(self, image_size = 25, conv_dim = 64):
        super(build_c_network, self).__init__()

        #self.conv0 = conv(1, conv_dim, 5, stride=1, pad=2, bn=True)
        self.conv1 = conv(1, conv_dim*2, 5, stride=2, pad=2,  bn=True)
        self.conv1_1 = conv(conv_dim*2, conv_dim*2, 5, stride = 1, pad=1, bn=True)
        self.conv1_2 = conv(conv_dim*2, conv_dim*4, 5, stride = 1, pad=0, bn=True)
        #self.conv1_3 = conv(conv_dim*2, conv_dim*2, 5, stride=1, pad=2, bn=True)
        self.conv2 = conv(conv_dim*4, conv_dim*4, 3, stride=1, pad=1, bn=True)
        self.conv2_1 = conv(conv_dim*4, conv_dim*4 , 3, stride=1, pad=1, bn=True)
        self.conv2_2 = conv(conv_dim * 4, conv_dim * 4, 3, stride=1, pad=0, bn=True)
        self.conv3 = conv(conv_dim*4, conv_dim*8, 1, stride=1, pad=0, bn=True)
        self.fc = conv(conv_dim*8, 1, 1, stride = 1,  pad=0, bn=True)



    def forward(self, x):  # If image_size is 64, output shape is as below.
        #out = F.elu(self.conv0(x))
        out = F.elu(self.conv1(x))
        out = F.elu(self.conv1_1(out))
        out = F.elu(self.conv1_2(out))
        #out = F.elu(self.conv1_3(out))
        out_res = out
        out = F.elu(self.conv2(out))
        out = F.elu(self.conv2_1(out))
        out = out_res + out
        out = F.elu(self.conv2_2(out))
        out = F.elu(self.conv3(out))
        out = F.elu(self.fc(out).squeeze())

        return out


class build_network(nn.Module):
    def __init__(self, image_size=625):
        super(build_network, self).__init__()
        self.fc1 = nn.Linear(image_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 25)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = x.view(1, -1)
        out = F.elu(self.fc1(x))
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