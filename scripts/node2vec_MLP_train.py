import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.mlp_model import MLP_Baseline
from src.plot_utils import save_results, plot_loss_curve, plot_accuracy_curve
from src.train_utils import train_fullbatch, eval_fullbatch, train_fullbatch_multilabel, eval_fullbatch_multilabel
from ogb.nodeproppred import PygNodePropPredDataset
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from glob import glob

DATA_ROOT = "./dataset"

def main():
    parser = argparse.ArgumentParser(description="MLP Baseline Training Script")
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Dataset name (default: ogbn-arxiv)(Other options: ogbn-products, ogbn-proteins)')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels (default: 256)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (default: 0.4)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs (default: 50)')
    # parser.add_argument('--batch_size', type=int, default=1024, help='Batch size (default: 1024)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (default: cuda if available else cpu)')
    
    args = parser.parse_args()

    # Load dataset
    dataset = PygNodePropPredDataset(name=args.dataset, root=DATA_ROOT)
    data = dataset[0]

    node2vec_embed_path = glob(f"results/Node2Vec_Pretrain/{args.dataset}/node2vec_embeddings_*.pt")

    assert len(node2vec_embed_path) == 1, f"Expected one Node2Vec embedding file, found {len(node2vec_embed_path)}."

    node2vec_embed = torch.load(
        node2vec_embed_path[0],
        map_location="cpu"
    )

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    device = torch.device(args.device)

    x = torch.cat([data.x, node2vec_embed], dim=-1).to(device) if args.dataset != "ogbn-proteins" else node2vec_embed.to(device)
    y = data.y.squeeze().to(device) if args.dataset != "ogbn-proteins" else data.y.to(device, dtype=float)

    if args.dataset == 'ogbn-proteins':
        train_fullbatch_fn = train_fullbatch_multilabel
        eval_fullbatch_fn = eval_fullbatch_multilabel
    else:
        train_fullbatch_fn = train_fullbatch
        eval_fullbatch_fn = eval_fullbatch

    model = MLP_Baseline(
        in_channels=x.shape[1],
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        is_multilabel=(args.dataset == 'ogbn-proteins')
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_path = Path(f"results/Node2Vec_MLP/{args.dataset}/best_model.pt")

    best_val_acc = 0
    test_acc_at_best_val = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_fullbatch_fn(model, x, y, train_idx, optimizer)
        val_loss, val_acc = eval_fullbatch_fn(model, x, y, valid_idx)
        test_loss, test_acc = eval_fullbatch_fn(model, x, y, test_idx)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            test_acc_at_best_val = test_acc
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} "
            f"(Best Val Acc: {best_val_acc:.4f}, Test Acc at Best Val: {test_acc_at_best_val:.4f})")


    print("\n--- Training Finished ---")
    print(f"Best validation accuracy of {best_val_acc:.4f} achieved at epoch {best_epoch}.")
    
    print("\n--- Loading best model for final evaluation ---")
    model.load_state_dict(torch.load(checkpoint_path))
    
    _, final_val_acc = eval_fullbatch_fn(model, x, y, valid_idx)
    _, final_test_acc = eval_fullbatch_fn(model, x, y, test_idx)

    print(f'Final Results (from best model at epoch {best_epoch}):')
    print(f'  Validation Accuracy: {final_val_acc:.4f}')
    print(f'  Test Accuracy: {final_test_acc:.4f}')

    # 保存结果并绘制学习曲线
    best_epoch_metrics = {
        'best_epoch': best_epoch,
        'val_acc': final_val_acc,
        'test_acc': final_test_acc,
    }
    save_prefix = save_results(history, args, best_epoch_metrics, f"Node2Vec_MLP/{args.dataset}")
    plot_loss_curve(history, save_prefix)
    plot_accuracy_curve(history, save_prefix)


if __name__ == "__main__":
    main()
        


