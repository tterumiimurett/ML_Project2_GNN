import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.node2vec_model import load_Node2Vec_embeddings
from argparse import ArgumentParser
from ogb.nodeproppred import PygNodePropPredDataset
import torch
from tqdm import tqdm
from collections import deque
import numpy as np
from pathlib import Path

DATA_ROOT = "./dataset"

def main():
    parser = ArgumentParser(description="Train Node2Vec Pretraining stage on OGBN-Datasets")
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Dataset name (default: ogbn-arxiv)(Other options: ogbn-products, ogbn-proteins)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = Path(f"results/Node2Vec_Pretrain/{args.dataset}/node2vec_embeddings.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = PygNodePropPredDataset(name=args.dataset, root=DATA_ROOT)
    data = dataset[0]
    
    node2vec = load_Node2Vec_embeddings(data, device)

    optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=0.01)

    loader = node2vec.loader(
        batch_size=128,
        shuffle=True,
        num_workers=4,
    )

    node2vec.train()

    window = 5
    std_threshold = 1e-3
    loss_queue = deque(maxlen=window)

    pbar = tqdm(range(1, 101), desc="Training Node2Vec", ncols=100)
    for epoch in pbar:
        total_loss = 0

        for pos_rw, neg_rw in loader:
            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)

            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:03d}, Loss: {total_loss / len(loader):.4f}")
        loss_queue.append(total_loss / len(loader))

        if len(loss_queue) == window and np.std(loss_queue) < std_threshold:
            print("Early stopping triggered.")
            save_path = save_path.with_stem("node2vec_embeddings_converged_at_epoch_" + str(epoch))
            break

    node2vec.eval()
    for p in node2vec.parameters():
        p.requires_grad = False

    emb = node2vec.embedding.weight.detach()
    torch.save(emb, save_path)

if __name__ == "__main__":
    main()