# train.py
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from data_loader import load_npys, preprocess_data, create_dataloaders
from models import BiMambaModel


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    start = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.cpu().numpy())
    elapsed = time.time() - start
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1, elapsed


def model_size_mb(model):
    param_size  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1e6


def main():
    # ===== 路径配置 =====
    data_paths  = [
        '../sub_frame_driving_data/frame4_diff_data_driving_01.npy',
        '../sub_frame_driving_data/frame4_diff_data_driving_02.npy',
    ]
    label_paths = [
        '../sub_frame_driving_data/frame4_diff_labels_driving_01.npy',
        '../sub_frame_driving_data/frame4_diff_labels_driving_02.npy',
    ]

    # ===== 读取 + 预处理 =====
    data_x, data_y = load_npys(data_paths, label_paths)
    # 保持与原脚本一致：添加通道维，按需可选归一化/标准化
    data_x = data_x.astype(np.float32)
    # 可选：norm='minmax' 或 'zscore'
    data_x = preprocess_data(data_x[:, :, None, ...], norm=None)

    # ===== 超参 =====
    in_channels = 1
    cnn_out_dim = 128
    mamba_dim   = 128
    num_classes = 2
    batch_size  = 2048
    epochs      = 100
    lr          = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== DataLoader =====
    train_loader, val_loader, test_loader = create_dataloaders(
        data_x, data_y, batch_size=batch_size, splits=(0.7, 0.15, 0.15)
    )

    # ===== 模型与优化器 =====
    model = BiMambaModel(in_channels, cnn_out_dim, mamba_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ===== 训练 =====
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc, prec, rec, f1, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f} | "
              f"Val Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    # ===== 测试评估 =====
    acc, prec, rec, f1, infer_time = evaluate(model, test_loader, device)
    print("\n--- Final Test ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Inference Time (s): {infer_time:.2f}")
    print(f"Model Size (MB): {model_size_mb(model):.2f}")


if __name__ == "__main__":
    main()
