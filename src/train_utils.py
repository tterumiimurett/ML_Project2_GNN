from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import torch

@torch.no_grad()
def multilabel_micro_f1(logits, targets, threshold=0.5):
    # logits: [N, 112], targets: [N, 112] float 0/1
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()

    tp = (pred * targets).sum()
    fp = (pred * (1 - targets)).sum()
    fn = ((1 - pred) * targets).sum()

    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
    return f1.item()

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    pbar = tqdm(loader, desc="Training", ncols=100)
    for batch in pbar:
        batch = batch.to(device)
        batch_size = batch.batch_size

        y = batch.y[:batch_size].squeeze()
        optimizer.zero_grad()
        out = model(batch)
        
        loss = F.nll_loss(out[:batch_size], y)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss) * batch_size
        total_correct += int(out[:batch_size].argmax(dim=-1).eq(y).sum())
        total_nodes += batch_size
        
        pbar.set_postfix({
            'Train Loss': f'{total_loss/total_nodes:.4f}',
            'Train Acc': f'{total_correct/total_nodes:.4f}'
        })

    return total_loss / total_nodes, total_correct / total_nodes

def train_protrein(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    pbar = tqdm(loader, desc="Training", ncols=100)
    for batch in pbar:
        batch = batch.to(device)
        batch_size = batch.batch_size

        y = batch.y[:batch_size].squeeze()
        optimizer.zero_grad()
        out = model(batch)
        loss_fn = BCEWithLogitsLoss(reduction="mean")  
        loss = loss_fn(out[:batch_size], y)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss) * batch_size
        total_correct += int(out[:batch_size].argmax(dim=-1).eq(y).sum())
        total_nodes += batch_size
        
        pbar.set_postfix({
            'Train Loss': f'{total_loss/total_nodes:.4f}',
            'Train Acc': f'{total_correct/total_nodes:.4f}'
        })

    return total_loss / total_nodes, total_correct / total_nodes


def train_fullbatch(model, x, y, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(x)                      # [N, C]
    loss = F.nll_loss(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

    acc = (
        out[train_idx].argmax(dim=-1)
        == y[train_idx]
    ).float().mean()

    return loss.item(), acc.item()

def train_fullbatch_protein(model, x, y, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(x)                       # [N,112] logits
    loss_fn = BCEWithLogitsLoss(reduction="mean")
    loss = loss_fn(out[train_idx], y[train_idx].float())

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        f1 = multilabel_micro_f1(out[train_idx], y[train_idx])

    return loss.item(), f1

@torch.no_grad()
def eval(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    pbar = tqdm(loader, desc=f"Evaluating", ncols=100)
    for batch in pbar:
        batch = batch.to(device)
        batch_size = batch.batch_size
        y = batch.y[:batch.batch_size].squeeze()
        
        out = model(batch)
        
        loss = F.nll_loss(out[:batch_size], y)
        
        total_loss += float(loss) * batch_size
        total_correct += int(out[:batch_size].argmax(dim=-1).eq(y).sum())
        total_nodes += batch_size
        
        pbar.set_postfix({
            'Loss': f'{total_loss/total_nodes:.4f}',
            'Acc': f'{total_correct/total_nodes:.4f}'
        })

    return total_loss/total_nodes, total_correct / total_nodes

@torch.no_grad()
def eval_protein(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    pbar = tqdm(loader, desc=f"Evaluating", ncols=100)
    for batch in pbar:
        batch = batch.to(device)
        batch_size = batch.batch_size
        y = batch.y[:batch.batch_size].squeeze()
        
        out = model(batch)
        loss_fn = BCEWithLogitsLoss(reduction="mean")   
        loss = loss_fn(out[:batch_size], y)
        
        total_loss += float(loss) * batch_size
        total_correct += int(out[:batch_size].argmax(dim=-1).eq(y).sum())
        total_nodes += batch_size
        
        pbar.set_postfix({
            'Loss': f'{total_loss/total_nodes:.4f}',
            'Acc': f'{total_correct/total_nodes:.4f}'
        })

    return total_loss/total_nodes, total_correct / total_nodes

@torch.no_grad()
def eval_fullbatch(model, x, y, idx):
    model.eval()
    out = model(x)

    loss = F.nll_loss(out[idx], y[idx])
    acc = (out[idx].argmax(dim=-1) == y[idx]).float().mean()

    return loss.item(), acc.item()

@torch.no_grad()
def eval_fullbatc_protein(model, x, y, idx):
    model.eval()
    out = model(x)
    loss_fn = BCEWithLogitsLoss(reduction="mean")  
    loss = loss_fn(out[idx], y[idx])
    f1 = multilabel_micro_f1(out[idx], y[idx])

    return loss.item(), f1