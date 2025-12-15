from utils import get_ogbn_arxiv_data, load_llm_and_tokenizer, Arxiv_Dataset, eval
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path

from pdb import set_trace as st

def main():

    parser = ArgumentParser(description="LLM Zero-shot Classification on OGBN-Arxiv")
    parser.add_argument('--save', action = "store_true", help="Whether to save detailed results as JSON")
    parser.add_argument('--save_path', type=Path, default=Path("results/LLM_ZeroShot/Results.jsonl"), help="Path to save detailed results JSON")
    args = parser.parse_args()
    print(args)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    SUBSET_SIZE = 512 # 设置为 None 来运行完整的测试集
    BATCH_SIZE = 128
    
    MODEL_ID = "/ssd/common/LLMs/Qwen3-8B" 
    
    paper_texts, labels, split_idx, original_label_mapping = get_ogbn_arxiv_data()

    
    print("--- Dataset Statistics ---")
    print(f"  Train samples: {len(split_idx['train'])}")
    print(f"  Valid samples: {len(split_idx['valid'])}")
    print(f"  Test samples:  {len(split_idx['test'])}")
    print("--------------------------\n")
    
    # 创建一份干净、统一的类别名称用于Prompt和评估
    clean_label_mapping = {}
    clean_categories_list = []
    for idx, name in sorted(original_label_mapping.items()):
        clean_name = name.replace('arxiv ', '').strip() 
        if ' ' in clean_name:
            parts = clean_name.split(' ', 1)
            # 确保格式类似 "cs.AI"
            clean_name = f"{parts[0].lower()}.{parts[1].upper()}"
        clean_label_mapping[idx] = clean_name
        clean_categories_list.append(clean_name)
    
    test_indices = split_idx['test']
    
    if SUBSET_SIZE is not None:
        test_indices = test_indices[:SUBSET_SIZE]

    # 加载模型
    model, tokenizer = load_llm_and_tokenizer(MODEL_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_dataset = Arxiv_Dataset(paper_texts, labels, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    save_json = []
    print(f"Starting inference on {len(test_indices)} samples using {MODEL_ID} with batch size {BATCH_SIZE}...")
    
    correct_predictions, total_predictions = eval(model, tokenizer, clean_label_mapping, clean_categories_list, test_loader, args)

    # 报告结果
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print("\n--- Evaluation Complete ---")
    print(f"Model: {MODEL_ID} (Non-Thinking Mode)")
    print(f"Test samples: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")

if __name__ == "__main__":
    main()
