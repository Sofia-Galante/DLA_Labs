import torch
import numpy as np
from torch.utils.data import Dataset

class TargetTransform:
    def __init__(self, width, height):
        self.w = width
        self.h = height
    
    def __call__(self, y):
        return torch.from_numpy(np.full((self.w, self.h), y, dtype=np.int64))
    
class DatasetWrapper(Dataset):
    def __init__(self, dataset, size, n_cols, n_rows):
        self.dataset = dataset
        n = len(dataset)
        rng = np.random.default_rng()
        self.indexes = rng.choice(n, (size, n_cols, n_rows))
        
    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, idx):
        image_indexes = self.indexes[idx]
        x_sample = list()
        y_sample = list()
        for row in image_indexes:
            x_row = list()
            y_row = list()
            for col in row:
                x, y = self.dataset[col]
                x_row.append(x)
                y_row.append(y)
            x_sample.append(torch.cat(x_row, dim=1))
            y_sample.append(torch.cat(y_row, dim=0))
        return torch.cat(x_sample, dim=2), torch.cat(y_sample, dim=1)
