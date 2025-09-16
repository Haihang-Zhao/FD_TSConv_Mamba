# data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class NPYDataset(Dataset):
    """
    data_x: numpy array, shape (N, T, C, H, W) or (N, T, H, W)
    data_y: numpy array, shape (N,) or (N,1)
    """
    def __init__(self, data_x, data_y):
        # 确保有通道维
        if data_x.ndim == 4:  # (N, T, H, W) -> (N, T, 1, H, W)
            data_x = data_x[:, :, None, ...]
        self.data = data_x.astype(np.float32)
        labels = data_y.squeeze()
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)  # (T, C, H, W)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def preprocess_data(x, norm: str | None = None, eps: float = 1e-8):
    """
    可选归一化/标准化（按每样本每时间步在 H,W 上逐通道处理）
    norm: None | 'minmax' | 'zscore'
    """
    if norm is None:
        return x
    if x.ndim != 5:
        raise ValueError("Expected shape (N, T, C, H, W) after channel insertion.")

    if norm.lower() == 'minmax':
        x_min = x.min(axis=(3, 4), keepdims=True)
        x_max = x.max(axis=(3, 4), keepdims=True)
        x = (x - x_min) / (x_max - x_min + eps)
    elif norm.lower() == 'zscore':
        x_mean = x.mean(axis=(3, 4), keepdims=True)
        x_std = x.std(axis=(3, 4), keepdims=True) + eps
        x = (x - x_mean) / x_std
    else:
        raise ValueError("norm should be None | 'minmax' | 'zscore'")
    return x.astype(np.float32)


def load_npys(data_paths: list[str], label_paths: list[str]):
    """读取多份 .npy 并在样本维拼接"""
    xs = [np.load(p) for p in data_paths]
    ys = [np.load(p) for p in label_paths]
    data_x = np.concatenate(xs, axis=0)
    data_y = np.concatenate(ys, axis=0)
    return data_x, data_y


def create_dataloaders(data_x, data_y, batch_size=2048, splits=(0.7, 0.15, 0.15), seed=42, num_workers=0):
    """划分数据并生成 DataLoader"""
    dataset = NPYDataset(data_x, data_y)
    n_total = len(dataset)
    n_train = int(splits[0] * n_total)
    n_val = int(splits[1] * n_total)
    n_test = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader
