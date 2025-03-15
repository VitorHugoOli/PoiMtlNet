import torch
from torch import nn

from configs.globals import DEVICE
from configs.model import ModelConfig


class CategoryPoiNet(nn.Module):
    def __init__(self, num_classes):
        super(CategoryPoiNet, self).__init__()

        self.linear = nn.Linear(256, 484)
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.linear(x)
        b, seq, dim = x.shape
        dim = int(dim ** 0.5)
        x = x.view(b, seq, dim, dim)

        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.view(b, seq, -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out

    @staticmethod
    def reshape_output(out, y):
        y = y.view(-1)
        return out, y

    @staticmethod
    def reshape_output_old(out, y):
        one_hot = torch.eye(ModelConfig.NUM_CLASSES).to(DEVICE)
        return out, one_hot[y]

