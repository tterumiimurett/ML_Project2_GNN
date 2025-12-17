import torch
import torch.nn.functional as F
from torch.nn import Linear

class MLP_Baseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, is_multilabel=False):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.dropout = dropout
        self.is_multilabel = is_multilabel

        # 输入层
        self.layers.append(Linear(in_channels, hidden_channels))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(Linear(hidden_channels, hidden_channels))
        
        # 输出层
        self.layers.append(Linear(hidden_channels, out_channels))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.is_multilabel:
            return x
        return F.log_softmax(x, dim=-1)