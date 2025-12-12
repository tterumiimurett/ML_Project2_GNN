import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout

        # 输入层
        mlp_initial = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        self.convs.append(GINConv(mlp_initial, train_eps=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # 隐藏层
        for _ in range(num_layers - 2):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
            self.convs.append(GINConv(mlp, train_eps=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        mlp_last = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        self.convs.append(GINConv(mlp_last, train_eps=False))


        # 输出层
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)
