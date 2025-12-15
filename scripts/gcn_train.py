import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
import sys

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset

from pathlib import Path

from src.gcn_model import GCN
from src.train_utils import train, eval
from src.plot_utils import save_results, plot_loss_curve, plot_accuracy_curve

# --- 训练与评估函数 ---

DATA_ROOT = "./dataset"

# --- 主函数 ---

def main():
    parser = argparse.ArgumentParser(description='OGBN-Products GIN training')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Dataset name (default: ogbn-arxiv)(Other options: ogbn-products, ogbn-proteins)')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size for training')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty)')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # --- 数据加载与预处理 ---
    print("Loading dataset...")
    dataset = PygNodePropPredDataset(name=args.dataset, root=DATA_ROOT, transform=T.ToUndirected())
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']

    print("Setting up data loaders...")
    # 邻域采样是处理大图的关键，[15, 10, 5] 表示为3层GNN，分别采样15, 10, 5个邻居
    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=[15, 10, 5][:args.num_layers],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12
    )

    # 验证和测试时，batch_size可以设得更大以加速评估
    subdivision_batch_size = args.batch_size * 6 
    val_loader = NeighborLoader(data, input_nodes=split_idx['valid'], num_neighbors=[15, 10, 5][:args.num_layers],
                              batch_size=subdivision_batch_size, num_workers=12)
    test_loader = NeighborLoader(data, input_nodes=split_idx['test'], num_neighbors=[15, 10, 5][:args.num_layers],
                               batch_size=subdivision_batch_size, num_workers=12)

    print("Initializing model...")
    model = GCN(
        in_channels=data.x.size(-1),
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- 训练设置 ---
    # 定义一个固定的检查点文件名
    checkpoint_path = Path(f"results/GCN/{args.dataset}/best_model.pt")

    # 初始化一个字典来存储训练过程中的指标
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    
    best_val_acc = 0.0
    best_epoch = 0

    print("Start training...")
    for epoch in range(1, 1 + args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        # 记录训练指标
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 每个epoch都进行评估
        val_loss, val_acc = eval(model, val_loader, device)
        test_loss, test_acc = eval(model, test_loader, device)
        
        # 记录评估指标
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        # 检查是否是最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_path.parent.mkdir(parents = True, exist_ok = True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    -> New best model saved at epoch {epoch} with Val Acc: {val_acc:.4f}")

    print("\n--- Training Finished ---")
    print(f"Best validation accuracy of {best_val_acc:.4f} achieved at epoch {best_epoch}.")
    
    print("\n--- Loading best model for final evaluation ---")
    model.load_state_dict(torch.load(checkpoint_path))
    
    _, final_val_acc = eval(model, val_loader, device)
    _, final_test_acc = eval(model, test_loader, device)

    print(f'Final Results (from best model at epoch {best_epoch}):')
    print(f'  Validation Accuracy: {final_val_acc:.4f}')
    print(f'  Test Accuracy: {final_test_acc:.4f}')

    # 保存结果并绘制学习曲线
    best_epoch_metrics = {
        'best_epoch': best_epoch,
        'val_acc': final_val_acc,
        'test_acc': final_test_acc,
    }
    save_prefix = save_results(history, args, best_epoch_metrics, f"GCN/{args.dataset}")
    plot_loss_curve(history, save_prefix)
    plot_accuracy_curve(history, save_prefix)


if __name__ == "__main__":
    main()
