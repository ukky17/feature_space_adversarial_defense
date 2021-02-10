import numpy as np

import torch
from torch.utils.data import DataLoader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            out_data = self.transform(out_data)
        return out_data, out_label

def get_representations(imgs1, imgs2, model, device, batch_size=400):
    dataloader = CustomDataset(imgs1, imgs2)
    dataloader = DataLoader(dataloader, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for i, (x1, x2) in enumerate(dataloader):
            y1 = model(x1.to(device)).detach().cpu().numpy()
            y2 = model(x2.to(device)).detach().cpu().numpy()
            if i == 0:
                data1, data2 = y1, y2
            else:
                data1 = np.concatenate([data1, y1], axis=0)
                data2 = np.concatenate([data2, y2], axis=0)
    return data1, data2
