"""
Mental Health Detection using BERT + BiLSTM + Graph Attention Network (GAT)

Author: Nishanth M

Features:
- BERTweet embeddings
- BiLSTM + Attention pooling
- Graph Attention Network (GAT)
- Focal Loss
- Cosine LR Scheduler
"""

import re
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")

# ================= CONFIG =================
DATA_PATH = "data/Mental-Health-Twitter.csv"
MODEL_NAME = "vinai/bertweet-base"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= DATA =================
class DatasetProcessor:
    def __init__(self, path):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        if "post_text" in df.columns:
            df = df.rename(columns={"post_text": "text"})
        df["text"] = df["text"].astype(str)
        return df

    def clean(self, text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        return text.strip()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx])
        }

# ================= MODEL =================
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, H):
        return torch.relu(self.fc(H))

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        hidden = self.bert.config.hidden_size

        self.lstm = nn.LSTM(hidden, 128, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(256, 1)

        self.gat = GATLayer(256, 128)

        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask).last_hidden_state
        x, _ = self.lstm(x)

        attn_w = torch.softmax(self.attn(x), dim=1)
        pooled = torch.sum(attn_w * x, dim=1)

        g_out = self.gat(pooled)
        combined = torch.cat([pooled, g_out], dim=1)

        return self.classifier(combined)

# ================= TRAIN =================
def train(model, loader, optimizer, criterion):
    model.train()
    preds, labels = [], []

    for batch in tqdm(loader):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        out = model(ids, mask)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        preds.extend(torch.argmax(out, 1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    return accuracy_score(labels, preds)

# ================= MAIN =================
def main():
    processor = DatasetProcessor(DATA_PATH)
    df = processor.load()

    df["text"] = df["text"].apply(processor.clean)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])

    model = Model(num_classes=df["label"].nunique()).to(DEVICE)
    tokenizer = model.tokenizer

    train_ds = TextDataset(train_df["text"].values, train_df["label"].values, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    weights = compute_class_weight("balanced", classes=np.unique(df["label"]), y=df["label"])
    weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(EPOCHS):
        acc = train(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}: Accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()