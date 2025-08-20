import torch
from torch.utils.data import Dataset


class ModelDataset(Dataset):
    def __init__(self, param_df, var_df, obj_param_df):
        """
        param_df: pandas DataFrame of parameters (input)
        var_df: pandas DataFrame of variables (target)
        """
        assert len(param_df) == len(var_df), "Mismatched number of rows"
        
        self.x = torch.tensor(param_df.values, dtype=torch.float32)
        self.y = torch.tensor(var_df.values, dtype=torch.float32)
        self.obj_param_y = torch.tensor(obj_param_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.obj_param_y[idx]
