from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import torch
from typing import Iterator, Tuple, Optional

class GPUBatchIterator:
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        shuffle: bool = True,
        extra_x: Optional[torch.Tensor] = None,
    ):
        self.x = x
        self.y = y
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.extra_x = extra_x
        self.num_samples = indices.size(0)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        if self.shuffle:
            perm = torch.randperm(self.num_samples, device=self.indices.device)
            idx = self.indices[perm]
        else:
            idx = self.indices

        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            if self.extra_x is not None:
                yield self.x[batch_idx], self.extra_x[batch_idx], self.y[batch_idx]
            else:
                yield self.x[batch_idx], self.y[batch_idx]

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

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
        if isinstance(batch, tuple):
            x, y = batch[0], batch[-1]
            batch_size = x.shape[0]
            out = model(x)
        else:
            batch = batch.to(device)
            batch_size = batch.batch_size
            y = batch.y[:batch_size].squeeze()
            out = model(batch)

        optimizer.zero_grad()
        
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

def train_multilabel(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_f1 = 0
    total_steps = 0

    pbar = tqdm(loader, desc="Training", ncols=100)
    for batch in pbar:
        if isinstance(batch, tuple): # GPUBatchIterator yields tuples
             x, y = batch[0], batch[-1]
             batch_size = x.shape[0]
             out = model(x)
        else: # NeighborLoader yields Data objects
             batch = batch.to(device)
             batch_size = batch.batch_size
             y = batch.y[:batch_size].squeeze()
             out = model(batch)

        optimizer.zero_grad()
        loss_fn = BCEWithLogitsLoss(reduction="mean")  
        loss = loss_fn(out[:batch_size], y.float())
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss)
        total_f1 += multilabel_micro_f1(out[:batch_size], y)
        total_steps += 1
        
        pbar.set_postfix({
            'Train Loss': f'{total_loss/total_steps:.4f}',
            'Train F1': f'{total_f1/total_steps:.4f}'
        })

    return total_loss / total_steps, total_f1 / total_steps


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

def train_fullbatch_multilabel(model, x, y, train_idx, optimizer):
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
        if isinstance(batch, tuple):
            x, y = batch[0], batch[-1]
            batch_size = x.shape[0]
            out = model(x)
        else:
            batch = batch.to(device)
            batch_size = batch.batch_size
            y = batch.y[:batch_size].squeeze()
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
def eval_multilabel(model, loader, device):
    model.eval()
    total_loss = 0
    total_f1 = 0
    total_steps = 0

    pbar = tqdm(loader, desc=f"Evaluating", ncols=100)
    for batch in pbar:
        if isinstance(batch, tuple):
            x, y = batch[0], batch[-1]
            batch_size = x.shape[0]
            out = model(x)
        else:
            batch = batch.to(device)
            batch_size = batch.batch_size
            y = batch.y[:batch_size].squeeze()
            out = model(batch)
        
        loss_fn = BCEWithLogitsLoss(reduction="mean")   
        loss = loss_fn(out[:batch_size], y.float())
        
        total_loss += float(loss)
        total_f1 += multilabel_micro_f1(out[:batch_size], y)
        total_steps += 1
        
        pbar.set_postfix({
            'Loss': f'{total_loss/total_steps:.4f}',
            'F1': f'{total_f1/total_steps:.4f}'
        })

    return total_loss/total_steps, total_f1 / total_steps

@torch.no_grad()
def eval_fullbatch(model, x, y, idx):
    model.eval()
    out = model(x)

    loss = F.nll_loss(out[idx], y[idx])
    acc = (out[idx].argmax(dim=-1) == y[idx]).float().mean()

    return loss.item(), acc.item()

@torch.no_grad()
def eval_fullbatch_multilabel(model, x, y, idx):
    model.eval()
    out = model(x)
    loss_fn = BCEWithLogitsLoss(reduction="mean")  
    loss = loss_fn(out[idx], y[idx].float())
    f1 = multilabel_micro_f1(out[idx], y[idx])

    return loss.item(), f1