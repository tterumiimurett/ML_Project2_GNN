from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime

def save_results(history, args, best_epoch_metrics, exp_name):
    """
    将训练历史、超参数和最佳epoch结果保存到JSON文件。
    """
    # 使用时间戳创建唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(f"results/{exp_name}/results_{timestamp}.json")

    filename.parent.mkdir(parents=True, exist_ok=True)
    
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
    return filename.with_suffix("")

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