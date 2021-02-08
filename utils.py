import os
import json
from tqdm import tqdm

import numpy as np
from scipy import io
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.datasets as dst
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import kornia

def dist(x1, x2):
    """ L2 distance between two arrays """
    return (np.sum((x1 - x2) ** 2)) ** 0.5

def dist_torch(x1, x2):
    """ L2 distance between two arrays """
    return (torch.sum((x1 - x2) ** 2)) ** 0.5

def mse_np(x1, x2):
    return np.sum((x1 - x2) ** 2) / np.prod(np.shape(x1))

def zncc_torch(x, y):
    """ zreo-mean correlation between two images
    x: (1, 3, x, y), y: (1, 3, x, y)
    """

    xx = x - torch.mean(x)
    yy = y - torch.mean(y)
    return torch.sum(xx * yy) / \
           torch.sqrt(torch.sum(xx ** 2)) / torch.sqrt(torch.sum(yy ** 2))

def get_edges_torch(img_tensor):
    """
    input: (n, 3, x, y)
    output: (n, 1, x, y)
    """

    return kornia.filters.Sobel()(kornia.color.rgb_to_grayscale(img_tensor))

def edge_zncc_torch(x, y):
    """ x, y: (1, 3, x, y) """

    return zncc_torch(get_edges_torch(x), get_edges_torch(y))

def edge_ss_torch(x, y):
    """ x, y: (1, 3, x, y) """

    return torch.sum((get_edges_torch(x) - get_edges_torch(y)) ** 2)

def corr_torch(x1, x2):
    v1 = x1 - torch.mean(x1)
    v2 = x2 - torch.mean(x2)

    var1 = torch.sqrt(torch.sum(v1 ** 2)) + 1e-10
    var2 = torch.sqrt(torch.sum(v2 ** 2)) + 1e-10
    return torch.sum(v1 * v2) / (var1 * var2)

# define the data_loader
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

class ToCaffeTensor(object):
    def __call__(self, imageTensor):
        return imageTensor[[2,1,0], :, :] * 255

# load imagenet-val dataset
def load_imagenet(data_len):
    path = '../data/ILSVRC2014/'
    img_filenames = np.sort(os.listdir(path + 'ILSVRC2012_img_val'))
    x_val = []
    for i in tqdm(range(data_len)):
        f = path + 'ILSVRC2012_img_val/' + img_filenames[i]
        img = Image.open(f).convert("RGB")
        x_val.append(img.copy())
        img.close()

    # label
    f = path + 'ILSVRC2014_devkit/data/'
    f += 'ILSVRC2014_clsloc_validation_ground_truth.txt'
    gt = pd.read_csv(f, header=None).values
    meta_clsloc = io.loadmat(path + 'ILSVRC2014_devkit/data/meta_clsloc.mat')
    class_index = pd.read_json(path + 'imagenet_class_index.json').loc[0, :]
    y_val = np.zeros(data_len, dtype=int)
    for i in tqdm(range(data_len)):
        idx = gt[i][0]
        wnid = meta_clsloc['synsets'][0][idx-1][1][0]
        y_val[i] = int(class_index[class_index == wnid].index[0])

    # label to str
    class_index = json.load(open(path + 'imagenet_class_index.json', 'r'))
    labels = {int(key):value for (key, value) in class_index.items()}
    return x_val, y_val, labels

def create_imagenet_loader(x_val, y_val, batch_size, basetype):
    # stack at data loader
    if basetype == 'caffe':
        normalize = tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[1,1,1])
        transform = tfs.Compose([tfs.Resize(256), tfs.CenterCrop(227),
                                 tfs.ToTensor(), normalize, ToCaffeTensor()])
    elif basetype == 'torch':
        normalize = tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        transform = tfs.Compose([tfs.Resize(256), tfs.CenterCrop(227),
                                 tfs.ToTensor(), normalize])
    DL = CustomDataset(x_val, y_val, transform=transform)
    DL = DataLoader(DL, batch_size=batch_size, shuffle=False)
    return DL

def deprocess_image(x):
    _x = np.copy(x)
    # normalize tensor: center on 0., ensure std is 0.1
    _x -= _x.mean()
    _x /= _x.std() + 1e-5
    return _x

def deprocess_image2(x):
    """utility function to convert a float array into a valid uint8 image.

    # Arguments
        x: A numpy-array representing the generated image.

    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    _x = np.copy(x)

    # normalize tensor: center on 0., ensure std is 0.25
    _x -= _x.mean()
    _x /= (_x.std() + 1e-5)
    _x *= 0.25

    # clip to [0, 1]
    _x += 0.5
    _x = np.clip(_x, 0, 1)

    # convert to RGB array
    _x *= 255
    _x = np.clip(_x, 0, 255).astype('uint8')
    return _x

# for ImageNet denormalization
class Denormalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class Clip:
    def __init__(self):
        return

    def __call__(self, tensor):
        t = tensor.clone()
        t[t>1] = 1
        t[t<0] = 0
        return t
