import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn import Linear

def mlp_collate_fn(batch):
    xs = torch.stack([b.x for b in batch])
    ys = torch.stack([b.y for b in batch])
    return MLPSample(xs, ys)

class MLPSample:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.batch_size = x.size(0)
    
    def to(self, device):
        return MLPSample(self.x.to(device), self.y.to(device))
    
class MLP_Dataset(Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        node_idx = self.indices[idx]
        x = self.data.x[node_idx]
        y = self.data.y[node_idx]
        return MLPSample(x, y)

class MLP_Baseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.dropout = dropout

        # 输入层
        self.layers.append(Linear(in_channels, hidden_channels))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(Linear(hidden_channels, hidden_channels))
        
        # 输出层
        self.layers.append(Linear(hidden_channels, out_channels))

    def forward(self, batch):
        x = batch.x if isinstance(batch, MLPSample) else batch
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=-1)