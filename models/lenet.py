# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init

# class LeNet(torch.nn.Module):

#     def __init__(self):
#         super(LeNet, self).__init__()

#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2),
#             torch.nn.Dropout(p=1 - 0.7))

#         self.layer2 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2),
#             torch.nn.Dropout(p=1 - 0.7))

#         self.layer3 = torch.nn.Sequential(
#             torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
#             torch.nn.Dropout(p=1 - 0.7))

#         self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
#         torch.nn.init.xavier_uniform(self.fc1.weight)
#         self.layer4 = torch.nn.Sequential(
#             self.fc1,
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=1 - 0.7))

#         self.fc2 = torch.nn.Linear(625, 10, bias=True)
#         torch.nn.init.xavier_uniform(self.fc2.weight)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = out.view(out.size(0), -1)   # Flatten them for FC
#         features = out
#         out = self.fc1(out)
#         out = self.fc2(out)
#         return out, features

import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        features = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(x))

        return y, features

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @property
    def name(self):
        return ('lenet')