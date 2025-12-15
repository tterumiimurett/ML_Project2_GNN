import re
import torch
import functools 
import os
import pandas as pd
from ogb.nodeproppred import NodePropPredDataset
from tqdm import tqdm
import json
# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from collections import Counter, defaultdict

import numpy as np

DATA_ROOT = "./dataset"

class Arxiv_Dataset(Dataset):
    def __init__(self, paper_texts, labels, indices):
        self.paper_texts = paper_texts
        self.labels = labels
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.paper_texts[actual_idx], self.labels[actual_idx]

def get_ogbn_arxiv_data():
    """
    下载并预处理 ogbn-arxiv 数据集。
    此函数使用 'ogb' 官方加载器获取图数据，并手动下载一个独立的文件来获取
    原始的标题和摘要文本，然后将它们映射在一起。
    """
    dataset_dir = os.path.join(DATA_ROOT, 'ogbn_arxiv')
    
    # 使用 OGB 加载器加载主数据集
    print("Loading OGB dataset structure (graph, labels, splits)...")
    # OGB 需要 weights_only=False（或者不设置该参数，默认为False）
    # 新版 torch（如 2.3+）可能默认为 True，导致 OGB 加载失败
    original_torch_load = torch.load
    try:
        torch.load = functools.partial(original_torch_load, weights_only=False)
        dataset = NodePropPredDataset(name='ogbn-arxiv', root=DATA_ROOT)
    finally:
        torch.load = original_torch_load
    
    split_idx = dataset.get_idx_split()
    _, labels = dataset[0]
    labels = labels.reshape(-1)

    # 下载并加载原始文本数据 (标题 + 摘要)
    raw_text_url = 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz'
    raw_text_path = os.path.join(dataset_dir, 'raw', 'titleabs.tsv.gz')

    if not os.path.exists(raw_text_path):
        print(f"Raw text file not found at {raw_text_path}. Downloading...")
        os.makedirs(os.path.dirname(raw_text_path), exist_ok=True)
        try:
            import requests
            with requests.get(raw_text_url, stream=True) as r:
                r.raise_for_status()
                with open(raw_text_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Download via requests failed: {e}. Please check connection/URL.")
            raise RuntimeError(f"Download failed for {raw_text_url}")
            
        if not os.path.exists(raw_text_path) or os.path.getsize(raw_text_path) < 1000:
            raise RuntimeError(f"Download failed for {raw_text_url}")

    print("Loading raw text data (titleabs.tsv.gz)...")
    titleabs_df = pd.read_csv(raw_text_path, sep='\t', header=None, names=['paper_id', 'title', 'abstract'], compression='gzip')
    
    # 将原始文本映射到节点索引
    nodeidx2paperid_path = os.path.join(dataset_dir, 'mapping', 'nodeidx2paperid.csv.gz')
    nodeidx2paperid_df = pd.read_csv(nodeidx2paperid_path, header=None, names=['paper_id'], compression='gzip')
    
    print("Mapping paper IDs to text (using fast zip method)...")
    titles = titleabs_df['title'].fillna('')
    abstracts = titleabs_df['abstract'].fillna('')
    paper_ids = titleabs_df['paper_id']
    
    # 构建从 paper_id 到文本的字典
    paperid2text = {
        pid: f"Title: {t}\n\nAbstract: {a}" 
        for pid, t, a in zip(paper_ids, titles, abstracts)
    }

    # 创建最终的论文文本列表
    num_nodes = labels.shape[0]
    paper_texts = [''] * num_nodes
    for node_idx in range(num_nodes):
        paper_id = nodeidx2paperid_df['paper_id'].iloc[node_idx]
        paper_texts[node_idx] = paperid2text.get(paper_id, "Title: \n\nAbstract: ")

    # 加载标签映射
    label_mapping_path = os.path.join(dataset_dir, 'mapping', 'labelidx2arxivcategeory.csv.gz')
    label_mapping_df = pd.read_csv(label_mapping_path, compression='gzip')
    label_mapping = dict(zip(label_mapping_df['label idx'], label_mapping_df['arxiv category']))
    
    return paper_texts, labels, split_idx, label_mapping


def load_llm_and_tokenizer(model_id="Qwen/Qwen3-8B"): 
    """
    加载量化的LLM及其分词器。
    使用 BitsAndBytes (BNB) 4-bit 量化，以便在消费级GPU上运行。
    """
    print(f"Loading model: {model_id}...")
    
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True, 
    #     bnb_4bit_compute_dtype=torch.float16
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side='left' # 明确指定左填充
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer

def extract_prediction(generated_text, all_categories):
    """
    从模型的输出中提取预测的类别名称。
    `all_categories` 应该是一个干净的类别名称列表, 例如 ['cs.AI', 'cs.AR']。
    """
    first_line = generated_text.split('\n')[0].strip()
    
    for category in all_categories:
        # 使用正则表达式和 re.IGNORECASE (忽略大小写) 来进行匹配
        if re.search(r'\b' + re.escape(category) + r'\b', first_line, re.IGNORECASE):
            return category
            
    for category in all_categories:
        if re.search(r'\b' + re.escape(category) + r'\b', generated_text, re.IGNORECASE):
            return category
            
    return None # 如果找不到任何有效类别，返回 None

def get_clean_label_mapping(original_label_mapping):
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
    return clean_label_mapping, clean_categories_list

def build_prompt(paper_text, label_mapping, playbook=None, example_book=None):
    """
    为LLM构建 Zero-shot 分类提示词 (prompt)
    """
    try:
        title, abstract = paper_text.split('\n\nAbstract: ')
        title = title.replace('Title: ', '')
    except ValueError:
        title = "N/A"
        abstract = "N/A"
        if paper_text.startswith("Title: "):
            title = paper_text.replace('Title: ', '').strip()
        if '\n\nAbstract: ' in paper_text:
             abstract = paper_text.split('\n\nAbstract: ')[-1]

    # 将标签字典转换为有序列表
    category_list = [f"{idx}. {name}" for idx, name in label_mapping.items()]
    category_list_str = "\n".join(category_list)

    playbook_text = ""
    if playbook:
        playbook_text = (
            "The following classification heuristics were learned "
            "from training data and may help:\n"
            + "\n".join(f"- {r}" for r in playbook)
            + "\n\n"
        )
    
    example_text = ""
    if example_book:
        example_text = (
            "Here are some examples of papers and their true categories:\n\n"
            + "".join(example_book)
            + "\n"
        )


    prompt = f"""You are an expert academic assistant. Your task is to classify a given scientific paper into one of the following 40 arXiv categories.

Here are the possible categories:
{category_list_str}

{playbook_text}

{example_text}

Now, classify the following paper based on its title and abstract. You must respond with only the full category name from the list above and nothing else.

Title: {title}
Abstract: {abstract}

Important: Respond with one of the exact category names listed above.
Category:"""
    return prompt

def eval(model, tokenizer, clean_label_mapping, clean_categories_list, loader, args, playbook=None, example_book=None):
    # 运行推理和评估
    correct_predictions = 0
    total_predictions = 0

    for i, (batch_paper_texts, batch_labels) in enumerate(tqdm(loader, desc="Evaluating")):
        prompts = [build_prompt(paper_text, clean_label_mapping, playbook, example_book) for paper_text in batch_paper_texts]
        batch_true_labels = [clean_label_mapping[label.item()] for label in batch_labels]

        messages_batch = [[{"role": "user", "content": p}] for p in prompts]

        text_inputs = tokenizer.apply_chat_template(
            messages_batch, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=10, 
            pad_token_id=tokenizer.pad_token_id
        )

        responses_texts = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted_label_names = [extract_prediction(response_text, clean_categories_list) for response_text in responses_texts]

        prediction_grade_array = np.array([(predicted_label_name.lower() == true_label.lower()) if predicted_label_name else False for predicted_label_name, true_label in zip(predicted_label_names, batch_true_labels)]).astype(int)

        correct_predictions += prediction_grade_array.sum()
        total_predictions += len(batch_true_labels)

        if args.save:
            with open(args.save_path, 'a') as f:
                for j, response_text in enumerate(responses_texts):
                    json_line = {
                        "prompt": prompts[j],
                        "true_label": batch_true_labels[j],
                        "model_output": response_text
                    }
                    f.write(json.dumps(json_line) + "\n")
    return correct_predictions, total_predictions


def train(model, tokenizer, clean_label_mapping, clean_categories_list, loader):
    confusion_counter = Counter()
    confusion_examples = defaultdict(list)

    for i, (batch_paper_texts, batch_labels) in enumerate(tqdm(loader, desc="Training")):
        prompts = [build_prompt(paper_text, clean_label_mapping) for paper_text in batch_paper_texts]
        batch_true_labels = [clean_label_mapping[label.item()] for label in batch_labels]

        messages_batch = [[{"role": "user", "content": p}] for p in prompts]

        text_inputs = tokenizer.apply_chat_template(
            messages_batch, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=10, 
            pad_token_id=tokenizer.pad_token_id
        )

        responses_texts = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted_label_names = [extract_prediction(response_text, clean_categories_list) for response_text in responses_texts]

        for prediced_label_name, true_label_name, paper_text in zip(predicted_label_names, batch_true_labels, batch_paper_texts):
            if prediced_label_name and prediced_label_name.lower() != true_label_name.lower():
                confusion_counter[(true_label_name, prediced_label_name)] += 1
                if len(confusion_examples[(true_label_name, prediced_label_name)]) < 5:
                    confusion_examples[(true_label_name, prediced_label_name)].append(paper_text)

    return confusion_counter, confusion_examples