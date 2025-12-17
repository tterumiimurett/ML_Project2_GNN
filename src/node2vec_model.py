import torch
from torch_geometric.nn import Node2Vec

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