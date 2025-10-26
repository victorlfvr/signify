# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in tqdm(dataloader, desc="Train", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total * 100
    return avg_loss, acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, gts = [], []

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Eval", leave=False):
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

            preds.extend(out.argmax(1).cpu().numpy())
            gts.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = correct / total * 100
    return avg_loss, acc, np.array(preds), np.array(gts)


def compute_metrics(preds, gts, class_names):
    cm = confusion_matrix(gts, preds)
    report = classification_report(gts, preds, target_names=class_names, digits=3)
    return cm, report
