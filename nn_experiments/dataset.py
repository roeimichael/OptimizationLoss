import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class StudentDataset(Dataset):
    def __init__(self, X, y, group_ids):
        self.X = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        self.y = torch.LongTensor(y.values if hasattr(y, 'values') else y)
        self.group_ids = torch.LongTensor(group_ids.values if hasattr(group_ids, 'values') else group_ids)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'label': self.y[idx],
            'group_id': self.group_ids[idx]
        }
