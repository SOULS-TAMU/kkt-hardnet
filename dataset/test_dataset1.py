import torch
import numpy as np
from torch.utils.data import Dataset


# ------------------- DATASET -------------------
class BDataset(Dataset):
    def __init__(self, sol_fn, b_range=(0.76, 2.0), n_samples=1000):
        b_vals = np.random.uniform(*b_range, size=(n_samples,))
        x_vals = np.array([sol_fn(bi) for bi in b_vals])  # shape (N, 2)

        self.b_vals = torch.tensor(b_vals, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        self.x_vals = torch.tensor(x_vals, dtype=torch.float32)              # (N, 2)

    def __len__(self):
        return len(self.b_vals)

    def __getitem__(self, idx):
        return self.b_vals[idx], self.x_vals[idx]