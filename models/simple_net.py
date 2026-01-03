import torch.nn as nn

# models/simple_net.py
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10) # 10 output classes
        )
    
    def forward(self, x):
        return self.net(x)