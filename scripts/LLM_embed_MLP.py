import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer, AutoModel
from ogb.nodeproppred import NodePropPredDataset
from pathlib import Path
import argparse

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


from src.gcn_model import GCN
from src.LLM_related_model import train, eval, MLP_Pretrain, LLM_Embed_Dataset, LLM_Enbed_collate_fn
from src.train_utils import GPUBatchIterator
from src.plot_utils import save_results, plot_loss_curve, plot_accuracy_curve
from pdb import set_trace as st
from LLM.utils import get_ogbn_arxiv_data
from torch.utils.data import DataLoader

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="MLP Baseline Training Script")
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Dataset name (default: ogbn-arxiv)(Other options: ogbn-products, ogbn-proteins)')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels (default: 256)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (default: 0.4)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size (default: 1024)')
    parser.add_argument('--use_full_batch', action='store_true', help='Use full-batch training (default: False)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (default: cuda if available else cpu)')
    
    args = parser.parse_args()

    MODEL_ID = "/ssd/common/LLMs/Qwen3-Embedding-0.6B"

    DATA_ROOT = "./dataset"

    # 初始化BERT
    # Load model directly
    device = torch.device(args.device)

    paper_texts, labels, split_idx, original_label_mapping = get_ogbn_arxiv_data()

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=DATA_ROOT, transform=T.ToUndirected())

    lm_embed_path = Path("results/Qwen3_Embed/Qwen3_embedding_06b.pt")

    if not lm_embed_path.exists():
        model = SentenceTransformer(
            MODEL_ID,
            model_kwargs={"device_map": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
        )
        model.eval()
        print("Generating Qwen embeddings...")
        lm_embed_path.parent.mkdir(parents=True, exist_ok=True)
        lm_embeddings = []
        for i in tqdm(range(0, len(paper_texts), 100), desc = "Generating Qwen embeddings"):
            batch_texts = paper_texts[i:i + 100]
            batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
            lm_embeddings.append(batch_embeddings)
        lm_embeddings = torch.cat(lm_embeddings, dim=0).to("cuda")
        torch.save(lm_embeddings, lm_embed_path)
    else:
        lm_embeddings = torch.load(lm_embed_path).to(device)

    data = dataset[0].to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    train_loader = GPUBatchIterator(data.x, data.y.squeeze(), train_idx, args.batch_size, shuffle=True, extra_x=lm_embeddings)
    valid_loader = GPUBatchIterator(data.x, data.y.squeeze(), valid_idx, args.batch_size, shuffle=False, extra_x=lm_embeddings)
    test_loader = GPUBatchIterator(data.x, data.y.squeeze(), test_idx, args.batch_size, shuffle=False, extra_x=lm_embeddings)

    # train_set = LLM_Embed_Dataset(lm_embeddings, data.y, train_idx)
    # valid_set = LLM_Embed_Dataset(lm_embeddings, data.y, valid_idx)
    # test_set = LLM_Embed_Dataset(lm_embeddings, data.y, test_idx)

    # train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=LLM_Enbed_collate_fn, shuffle=True, num_workers=4)
    # valid_loader = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=LLM_Enbed_collate_fn, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=LLM_Enbed_collate_fn, shuffle=False, num_workers=4)


    model = MLP_Pretrain(
        in_channels = lm_embeddings.shape[1], 
        hidden_channels = args.hidden_channels, 
        out_channels = dataset.num_classes, 
        num_layers = args.num_layers, 
        dropout = args.dropout
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- 训练设置 ---
    # 定义一个固定的检查点文件名
    checkpoint_path = Path(f"results/LLM_embed/MLP/best_model.pt")

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
        val_loss, val_acc = eval(model, valid_loader, device)
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
    
    _, final_val_acc = eval(model, valid_loader, device)
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
    save_prefix = save_results(history, args, best_epoch_metrics, f"LLM_embed/MLP")
    plot_loss_curve(history, save_prefix)
    plot_accuracy_curve(history, save_prefix)


if __name__ == "__main__":
    main()
