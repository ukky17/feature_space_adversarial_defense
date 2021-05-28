import numpy as np

import torch
import torchvision.datasets as dst
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

def data_loader(dataset, batch_size):
    if dataset == 'stl10':
        transform_test = tfs.Compose([tfs.ToTensor(),
                                      tfs.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))])
        data_test = dst.STL10('data/STL10', download=True, split='test',
                                transform=transform_test)
        dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
        labelStr = ["airplane", "bird", "car", "cat", "deer", "dog",
                    "horse", "monkey", "ship", "truck"]
        n_class, img_size = 10, (3, 96, 96)
    return dataloader, labelStr, n_class, img_size

def select_correct(dataloader, model_Z, device):
    data_dict = dict()
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            out = model_Z.forward(data.to(device))
            pred = torch.argmax(out, 1).detach().cpu().numpy()

            data = data.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            data = data[pred == target]
            target = target[pred == target]

            if i == 0:
                data_dict['x1'] = data
                data_dict['target'] = target
            else:
                data_dict['x1'] = np.concatenate([data_dict['x1'], data], axis=0)
                data_dict['target'] = np.concatenate([data_dict['target'], target], axis=0)
    return data_dict
