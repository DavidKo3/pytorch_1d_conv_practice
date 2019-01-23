import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv1d(28, 10, 5)
        # self.pool1 = nn.
        self.conv2 = nn.Conv1d(10, 20, 5)
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(400, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, input):
        print("1.size : " , input.shape)
        print("self.conv1(input): ", self.conv1(input).shape)
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        print("4.size :", x.shape)
        x = F.relu(self.fc1(x))
        # print("5.size :", x.shape)
        x = F.relu(self.fc2(x))
        # print("6.size :", x.shape)
        # x = self.pool1(F.relu(self.conv1(input)))
        # print("2.size : ", x.shape)
        # x = self.pool2(F.relu(self.conv2(x)))
        # # print("3.size :", x.shape)

        return x


