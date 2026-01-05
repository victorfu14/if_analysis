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
        
        # 1. First Convolutional Layer
        # Input: 1 channel (grayscale), Output: 32 channels, Kernel: 3x3
        # Image size changes: 28x28 -> 26x26
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        
        # 2. Second Convolutional Layer
        # Input: 32 channels, Output: 64 channels, Kernel: 3x3
        # Image size changes (after pool): 13x13 -> 11x11
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        # 3. Fully Connected (Dense) Layers
        # We need to calculate the input size for the first linear layer.
        # The math: 
        #   - Start: 28x28
        #   - Conv1 (3x3 kernel, no pad): 26x26
        #   - MaxPool (2x2): 13x13
        #   - Conv2 (3x3 kernel, no pad): 11x11
        #   - MaxPool (2x2): 5x5
        # Final shape per image: 64 channels * 5 * 5 pixels = 1600 features
        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10) # 10 digits

        # Optional: Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Pass through Conv1 -> ReLU -> MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Pass through Conv2 -> ReLU -> MaxPool
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten: 
        # Convert the 3D tensor (Batch, 64, 5, 5) to 2D (Batch, 1600)
        x = torch.flatten(x, 1)
        
        # Pass through Dense Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x