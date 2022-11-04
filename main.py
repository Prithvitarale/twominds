import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.conv0 = nn.Conv2d(3, 8, 3, 1)
        self.conv1 = nn.Conv2d(8, 16, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(3936256, 1024)
        # self.fc1 = nn.Linear(2284800, 1024)
        # self.fc2 = nn.Linear(1024, 256)
        # self.fc3 = nn.Linear(256, 40)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.dropout1(x)
        x = F.relu(x)
        # x = self.conv4(x)
        # x = self.conv4_bn(x)
        # x = self.conv5(x)
        # x = self.conv5_bn(x)
        x = t.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.relu(x)


    def get_features(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = self.dropout1(x)
        x = F.leaky_relu(x)
        # x = self.pool(x)
        # x = self.conv3_bn(x)
        # x = self.conv4(x)

        # x = self.conv4_bn(x)

        # x = t.flatten(x, 1)
        # x = self.fc1(x)
        # x = t.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        return x