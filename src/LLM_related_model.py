# 步骤1：加载BERT模型和分词器（中文/英文可选，ogbn-arxiv是英文论文，选bert-base-uncased）
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import Linear, ReLU, Sequential, Parameter
from torch_geometric.nn import GCNConv
from pdb import set_trace as st


import torch
from typing import Iterator, Tuple

class GPUBatchIterator:
    def __init__(
        self,
        data, 
        lm_embeddings: torch.Tensor, 
        indices: torch.Tensor,    # (M,)   on GPU or CPU
        batch_size: int,
        shuffle: bool = True,
    ):
        assert data.x.is_cuda and data.y.is_cuda and lm_embeddings.is_cuda, "x and y and lm_embedding must already be on GPU"
        self.x = data.x
        self.lm_embeddings = lm_embeddings
        self.y = data.y.squeeze()
        self.indices = indices.to(data.x.device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = indices.size(0)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.shuffle:
            perm = torch.randperm(self.num_samples, device=self.indices.device)
            idx = self.indices[perm]
        else:
            idx = self.indices

        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield self.x[batch_idx], self.lm_embeddings[batch_idx], self.y[batch_idx]

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

class MLP_Pretrain(torch.nn.Module): 
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

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                last_hidden_state = x
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=-1), last_hidden_state

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    pbar = tqdm(loader, desc="Training", ncols=100)
    for (x, lm_embedding, y) in pbar:
        batch_size = x.shape[0]

        optimizer.zero_grad()
        out, _ = model(lm_embedding)
        
        loss = F.nll_loss(out[:batch_size], y)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss) * batch_size
        total_correct += int(out[:batch_size].argmax(dim=-1).eq(y).sum())
        total_nodes += batch_size
        
        pbar.set_postfix({
            'Train Loss': f'{total_loss/total_nodes:.4f}',
            'Train Acc': f'{total_correct/total_nodes:.4f}'
        })

    return total_loss / total_nodes, total_correct / total_nodes

@torch.no_grad()
def eval(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    pbar = tqdm(loader, desc=f"Evaluating", ncols=100)
    for (x, lm_embedding, y) in pbar:
        batch_size = x.shape[0]
        
        out, _ = model(lm_embedding)
        
        loss = F.nll_loss(out[:batch_size], y)
        total_loss += float(loss) * batch_size
        total_correct += int(out[:batch_size].argmax(dim=-1).eq(y).sum())
        total_nodes += batch_size
        
        pbar.set_postfix({
            'Loss': f'{total_loss/total_nodes:.4f}',
            'Acc': f'{total_correct/total_nodes:.4f}'
        })

    return total_loss/total_nodes, total_correct / total_nodes

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.mlps = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout
        self.log_alpha = Parameter(torch.tensor(0.0))
        self.log_beta  = Parameter(torch.tensor(0.0))

        # 输入层
        mlp_initial = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
        )
        self.mlps.append(mlp_initial)
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # 隐藏层
        for _ in range(num_layers - 2):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
            self.mlps.append(mlp)
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        mlp_last = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        self.mlps.append(mlp_last)
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # 输出层
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, batch):
        alpha = torch.exp(self.log_alpha)
        beta  = torch.exp(self.log_beta)
        st()
        x = batch.x[:, :128]
        lm_embed = batch.x[:, 128:]
        x = alpha * x + beta * lm_embed
        edge_index = batch.edge_index
        for i, conv in enumerate(self.convs):
            x = self.mlps[i](x)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)
