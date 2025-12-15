import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import Node2Vec
from torch.utils.data import Dataset
from pdb import set_trace as st

def node2vec_mlp_collate_fn(batch):
    xs = torch.stack([b.x for b in batch])
    ys = torch.stack([b.y for b in batch])
    edge_embed = torch.stack([b.edge_embed for b in batch])
    return Node2Vec_MLPSample(xs, ys, edge_embed)

class Node2Vec_MLPSample:
    def __init__(self, x, y, edge_embed):
        self.x = x
        self.y = y
        self.edge_embed = edge_embed
        self.batch_size = x.size(0)
    
    def to(self, device):
        return Node2Vec_MLPSample(self.x.to(device), self.y.to(device), self.edge_embed.to(device))
    
class Node2Vec_MLP_Dataset(Dataset):
    def __init__(self, data, indices, edge_embed):
        self.data = data
        self.indices = indices
        self.edge_embed = edge_embed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        node_idx = self.indices[idx]
        x = self.data.x[node_idx]
        y = self.data.y[node_idx]
        edge_embed = self.edge_embed[node_idx]
        return Node2Vec_MLPSample(x, y, edge_embed)

def load_Node2Vec_embeddings(data, device):
    return Node2Vec(
        edge_index=data.edge_index,   # 全图
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
        num_nodes=data.num_nodes,
    ).to(device)

class Node2Vec_MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout: float):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.dropout = dropout

        in_channels = in_channels + 128 # 增加 Node2Vec 嵌入维度

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
        
        return F.log_softmax(x, dim=-1)