import torch
import torch.nn as nn
import torch.nn.functional as F

# models/simple_net.py
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10) # 10 output classes
        )

    def forward(self, x):
        return self.net(x)

class MNISTNet(nn.Module):
    def __init__(self, *kwargs):
        super(MNISTNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3)
        self.fc1 = nn.Linear(in_features=5 * 5 * 5, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)

        # Optional: Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x