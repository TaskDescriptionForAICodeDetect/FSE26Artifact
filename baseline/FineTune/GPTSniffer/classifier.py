from typing import List, Dict, Any
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from pathlib import Path
import os
from tqdm import tqdm


class CodeDataset(Dataset):
    def __init__(self, input_ids: List[List[int]], attention_mask: List[List[int]], labels: List[int]):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name: str = "microsoft/codebert-base", num_labels: int = 2):
        super().__init__()
        def _resolve_local_path(name: str) -> str:
            try:
                p = Path(name)
                candidates = [
                    p,
                    Path(__file__).resolve().parent / name,
                    Path(__file__).resolve().parents[2] / Path(name).name,
                    Path.cwd() / name,
                ]
                for c in candidates:
                    try:
                        if c.exists():
                            return str(c.resolve())
                    except Exception:
                        continue
            except Exception:
                pass
            return name

        resolved = _resolve_local_path(model_name)
        if not Path(resolved).exists():
            candidates_hint = [
                str(Path(model_name)),
                str((Path(__file__).resolve().parent / model_name)),
                str((Path(__file__).resolve().parents[2] / Path(model_name).name)),
                str((Path.cwd() / model_name)),
            ]
            raise FileNotFoundError(
                "[GPTSniffer] Local model directory not found, please confirm the path. Tried: " + ", ".join(candidates_hint)
            )
        print(f"[GPTSniffer] Resolved model path: {resolved}")
        self.backbone = AutoModel.from_pretrained(resolved, local_files_only=True)
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return logits, loss


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = "microsoft/codebert-base",
    epochs: int = 2,
    lr: float = 2e-5,
    device: str = None,
    log_interval: int = 50,
) -> CodeBERTClassifier:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeBERTClassifier(model_name=model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    print(f"[GPTSniffer] Training on device: {device}")
    print(f"[GPTSniffer] Model: {model_name}")
    print(f"[GPTSniffer] Train batches: {num_train_batches}, Val batches: {num_val_batches}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % log_interval == 0 or (step + 1) == num_train_batches:
                avg_loss = running_loss / (step + 1)
                pbar.set_postfix({"train_loss": f"{avg_loss:.4f}"})

        model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            val_running_loss = 0.0
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits, vloss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                preds = logits.argmax(dim=-1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                if vloss is not None:
                    val_running_loss += vloss.item()
        acc = correct / max(total, 1)
        val_loss = val_running_loss / max(num_val_batches, 1)
        print(f"[GPTSniffer] Epoch {epoch+1}/{epochs} - train_loss: {running_loss / max(num_train_batches,1):.4f} | val_loss: {val_loss:.4f} | val_acc: {acc:.4f}")

    return model

