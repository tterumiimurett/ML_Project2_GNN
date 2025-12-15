from utils import get_ogbn_arxiv_data, load_llm_and_tokenizer, Arxiv_Dataset, eval, train
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path

from pdb import set_trace as st
from collections import Counter, defaultdict

from pathlib import Path

confusion_counter = Counter()
confusion_examples = defaultdict(list)

def generate_rule_for_confusion(
    model, tokenizer,
    true_label, pred_label,
    examples
):

    prompt = f"""
The model often confuses papers from category {true_label} with {pred_label}.

Below are some example papers (title + abstract):

{"".join(examples)}

Please write ONE short, general, actionable rule
that helps distinguish {true_label} from {pred_label}.
The rule should focus on high-level characteristics,
not specific datasets or paper names.

Return only ONE sentence.
"""
    message = [[{"role": "user", "content": prompt}]]
    text_inputs = tokenizer.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True).to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=500, 
        pad_token_id=tokenizer.pad_token_id
    )
    responses_texts = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return responses_texts[0].strip()

def main():

    parser = ArgumentParser(description="LLM Zero-shot Classification on OGBN-Arxiv")
    parser.add_argument('--save', action = "store_true", help="Whether to save detailed results as JSON")
    parser.add_argument('--save_path', type=Path, default=Path("results/LLM_Context_Learning/Results.jsonl"), help="Path to save detailed results JSON")
    args = parser.parse_args()
    print(args)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    TRAIN_SUBSET_SIZE = 5000
    SUBSET_SIZE = 512 # 设置为 None 来运行完整的测试集
    BATCH_SIZE = 32
    
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
    
    
    test_indices = split_idx['test'][:SUBSET_SIZE] if SUBSET_SIZE else split_idx['test']
    train_indices = split_idx['train'][:TRAIN_SUBSET_SIZE] if TRAIN_SUBSET_SIZE else split_idx['train']

    # 加载模型
    model, tokenizer = load_llm_and_tokenizer(MODEL_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Arxiv_Dataset(paper_texts, labels, train_indices)
    test_dataset = Arxiv_Dataset(paper_texts, labels, test_indices)

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # randomly_pick 8 samples from train_loader as In-context examples

    examples = [train_dataset[i] for i in range(8)]

    example_book = []
    for paper_text, label in examples:
        label_name = clean_label_mapping[label.item()]
        example_book.append(f"Title and Abstract: {paper_text}\nTrue Category: {label_name}\n\n")
    
    # confusion_counter, confusion_examples = train(model, tokenizer, clean_label_mapping, clean_categories_list, train_loader)

    # playbook = []
    # TOP_K = 10  # 只处理最常见的 5 个 confusion pair

    # for (true, pred), _ in confusion_counter.most_common(TOP_K):
    #     examples = confusion_examples[(true, pred)]
    #     rule = generate_rule_for_confusion(
    #         model, tokenizer,
    #         true, pred,
    #         examples
    #     )
    #     playbook.append(f"[{true} vs {pred}] {rule}")

    save_json = []
    print(f"Starting inference on {len(test_indices)} samples using {MODEL_ID} with batch size {BATCH_SIZE}...")
    
    correct_predictions, total_predictions = eval(model, tokenizer, clean_label_mapping, clean_categories_list, test_loader, args, example_book=example_book)

    # 报告结果
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print("\n--- Evaluation Complete ---")
    print(f"Model: {MODEL_ID} (Non-Thinking Mode)")
    print(f"Test samples: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")

if __name__ == "__main__":
    main()
