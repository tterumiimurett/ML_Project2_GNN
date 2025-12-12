import argparse
import sys
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset

from models import GIN

# --- 训练与评估函数 ---

def train(model, train_loader, optimizer, device):
    """
    执行一个epoch的训练
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    pbar = tqdm(train_loader, desc="Training", ncols=100)
    for batch in pbar:
        batch = batch.to(device)
        # NeighborLoader产生的batch中，只有部分节点是需要预测的中心节点
        batch_size = batch.batch_size
        
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        
        # 计算损失和准确率时，只使用中心节点
        loss = F.nll_loss(out[:batch_size], batch.y[:batch_size].squeeze())
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss) * batch_size
        total_correct += int(out[:batch_size].argmax(dim=-1).eq(batch.y[:batch_size].squeeze()).sum())
        total_nodes += batch_size
        
        pbar.set_postfix({
            'Loss': f'{total_loss/total_nodes:.4f}',
            'Acc': f'{total_correct/total_nodes:.4f}'
        })

    return total_loss / total_nodes, total_correct / total_nodes


@torch.no_grad()
def test(model, data, loader, device, split_idx):
    """
    在验证集或测试集上评估模型。
    现在返回 (准确率, 平均损失)。
    """
    model.eval()
    
    out_all = []
    # 收集中心节点的真实标签
    y_true_all = data.y[split_idx].squeeze()

    for batch in tqdm(loader, desc="Evaluating", ncols=100):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        out_all.append(out[:batch.batch_size].cpu())
        
    out_all = torch.cat(out_all, dim=0)
    
    # 计算损失
    loss = F.nll_loss(out_all, y_true_all)
    
    # 计算准确率
    y_pred = out_all.argmax(dim=-1)
    correct = y_pred.eq(y_true_all).sum().item()
    total = split_idx.numel()
    
    return correct / total, float(loss)

# --- 结果处理函数 ---

def save_results(history, args, best_epoch_metrics):
    """
    将训练历史、超参数和最佳epoch结果保存到JSON文件。
    """
    # 使用时间戳创建唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/gin_results_{timestamp}.json"
    
    # 将超参数和训练历史结合起来，以便完整记录
    results_data = {
        'args': vars(args),
        'best_epoch_metrics': best_epoch_metrics,
        'history': history
    }
    
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=4)
        
    print(f"Results saved to {filename}")
    # 返回对应的图片文件名前缀
    return filename.replace(".json", "")

def plot_loss_curve(history, save_path_prefix):
    """
    绘制损失学习曲线并保存图片。
    """
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], 
             label='Train Loss', linestyle='-')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], 
             label='Validation Loss', linestyle='-')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = f"{save_path_prefix}_loss.png"
    plt.savefig(save_path)
    print(f"Loss curve plot saved to {save_path}")
    plt.close()

def plot_accuracy_curve(history, save_path_prefix):
    """
    绘制准确率学习曲线并保存图片。
    """
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(history['train_acc']) + 1), history['train_acc'], 
             label='Train Accuracy', linestyle='-')
    plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], 
             label='Validation Accuracy', linestyle='-')
    plt.plot(range(1, len(history['test_acc']) + 1), history['test_acc'], 
             label='Test Accuracy', linestyle='-')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = f"{save_path_prefix}_accuracy.png"
    plt.savefig(save_path)
    print(f"Accuracy curve plot saved to {save_path}")
    plt.close()

# --- 主函数 ---

def main():
    parser = argparse.ArgumentParser(description='OGBN-Products GIN training')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty)')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # --- 数据加载与预处理 ---
    print("Loading dataset...")
    dataset = PygNodePropPredDataset(name='ogbn-products', root='./data', transform=T.ToUndirected())
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
    model = GIN(
        in_channels=data.x.size(-1),
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- 训练设置 ---
    # 定义一个固定的检查点文件名
    checkpoint_path = "results/best_model.pt"

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
        val_acc, val_loss = test(model, data, val_loader, device, split_idx['valid'])
        test_acc, test_loss = test(model, data, test_loader, device, split_idx['test'])
        
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
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    -> New best model saved at epoch {epoch} with Val Acc: {val_acc:.4f}")

    print("\n--- Training Finished ---")
    print(f"Best validation accuracy of {best_val_acc:.4f} achieved at epoch {best_epoch}.")
    
    print("\n--- Loading best model for final evaluation ---")
    model.load_state_dict(torch.load(checkpoint_path))
    
    final_val_acc, _ = test(model, data, val_loader, device, split_idx['valid'])
    final_test_acc, _ = test(model, data, test_loader, device, split_idx['test'])

    print(f'Final Results (from best model at epoch {best_epoch}):')
    print(f'  Validation Accuracy: {final_val_acc:.4f}')
    print(f'  Test Accuracy: {final_test_acc:.4f}')

    # 保存结果并绘制学习曲线
    best_epoch_metrics = {
        'best_epoch': best_epoch,
        'val_acc': final_val_acc,
        'test_acc': final_test_acc,
    }
    save_prefix = save_results(history, args, best_epoch_metrics)
    plot_loss_curve(history, save_prefix)
    plot_accuracy_curve(history, save_prefix)


if __name__ == "__main__":
    main()
