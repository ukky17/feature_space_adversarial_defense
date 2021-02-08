""" evaluate a smoothed classifier on a dataset
"""
import os
import sys
from time import time
import datetime
import shutil
import argparse
import json
from collections import Counter
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.datasets as dst
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

import model_loader as model_loader
import utils as utils

def random_dir_torch(dim):
    # ref: https://qiita.com/mh-northlander/items/a2e643cf62317f129541

    with torch.no_grad():
        while True:
            vec = torch.randn(dim, device=device, requires_grad=False,
                              dtype=torch.float)
            r = torch.norm(vec).item()
            if r != 0.0:
                return vec / r

def random_noise(model, img, d_list, bound, n_iter):
    """
    d_list: list of the L2 norm of the noise
    """

    counters = []
    for dd in d_list:
        # create new images
        img_news = []
        for _ in range(n_iter):
            _noise = random_dir_torch(img.shape).detach().cpu().numpy() * dd
            img_new = img + _noise

            if bound[0] != -float('inf'):
                img_new = np.where(img_new >= bound[0], img_new, img - _noise)
            elif bound[1] != float('inf'):
                img_new = np.where(img_new <= bound[1], img_new, img - _noise)

            img_news.append(img_new)

        # dataloader
        img_news = np.array(img_news)
        _DL = utils.CustomDataset(img_news, np.zeros(n_iter))
        _DL = DataLoader(_DL, batch_size=400, shuffle=False)

        with torch.no_grad():
            classes = []
            for (img_new, _) in _DL:
                img_new = img_new.to(device)
                out = model(img_new)
                c = torch.argmax(out, 1).detach().cpu().numpy().tolist()
                classes += c

            counters.append(Counter(classes))
    return counters

def geometry(x1, x2, d_list, base_classifier, bound):
    d_mean = np.mean(np.sum((x1 - x2) ** 2, axis=(1, 2, 3)) ** 0.5)
    print(d_mean)

    DL = utils.CustomDataset(x1, x2)
    DL = DataLoader(DL, batch_size=1, shuffle=False)

    props = np.zeros((4, len(x1), len(d_list)))
    for idx, (_x1, _x2) in tqdm(enumerate(DL)):
        _c1 = torch.argmax(base_classifier(_x1.to(device)), dim=1).item()
        _c2 = torch.argmax(base_classifier(_x2.to(device)), dim=1).item()

        if _c1 == _c2:
            props[:, idx, :] = np.nan
            continue

        _x1 = _x1.detach().cpu().numpy()[0]
        _x2 = _x2.detach().cpu().numpy()[0]

        counters1 = random_noise(base_classifier, _x1, d_list, bound, n_iter=100)
        counters2 = random_noise(base_classifier, _x2, d_list, bound, n_iter=100)

        props[0, idx] = [counter[_c1] / 100 for counter in counters1]
        props[1, idx] = [counter[_c2] / 100 for counter in counters2]
        props[2, idx] = [counter[_c1] / 100 for counter in counters2]
        props[3, idx] = [counter[_c2] / 100 for counter in counters1]

    labels = ["x: ori, cla: ori", "x: adv, cla: adv", "x: adv, cla: ori"]
    n_nonnan = np.sum(np.isnan(props[0, :, 0]) == 0)
    for (i, col, fmt, label) in zip([0, 1, 2], ['m', 'c', 'c'], ['', '', '--'],
                                    labels):
        plt.errorbar(d_list,
                     np.nanmean(props[i], axis=0),
                     yerr=np.nanstd(props[i], axis=0) / np.sqrt(n_nonnan),
                     c=col, fmt=fmt, label=label)
    plt.axvline(d_mean, color='k')
    plt.title(params.space + ' space')
    plt.xlabel('epsilon')
    plt.ylabel('Class freq')
    plt.legend()
    plt.savefig(save_dir + params.space + '.png')
    plt.close()
    np.save(save_dir + 'topology_AEn_bounded_' + params.space, props)

def rpr_plotter(muls, counters1, counters2, cla1, cla2):
    plt.figure(figsize=(8, 6))

    ind = np.arange(len(counters1))
    ticks = [str(m) for m in muls]

    for j, counters, center in zip([0, 1], [counters1, counters2],
                                   ['orig', 'adv']):
        bars = np.zeros((3, len(counters)))
        for i, counter in enumerate(counters):
            v1 = counter[cla1]
            v2 = counter[cla2]
            v3 = sum(list(counter.values())) - v1 - v2
            bars[:, i] = [v1, v2, v3]

        ax = plt.subplot(1, 2, j+1)
        plot1 = ax.bar(ind, bars[0], color='m')
        plot2 = ax.bar(ind, bars[1], bottom=bars[0], color='c')
        plot3 = ax.bar(ind, bars[2], bottom=bars[0] + bars[1], color='gray')
        ax.set_xticks(ind)
        ax.set_xticklabels(ticks)
        ax.set_xlabel('Distance (d * )')
        ax.set_ylabel('Class %')
        ax.legend([plot1, plot2, plot3], ['Ori class', 'Adv class', 'Others'])
        ax.set_title('Center: ' + center)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stl10',
                        choices=['cifar10', 'imagenet', 'stl10'])
    parser.add_argument('--basenet', type=str, default='VGG_stl')
    parser.add_argument('--model_path', type=str, default='models/vgg_stl.pth')
    parser.add_argument('--target_layer', type=str, default='27')

    parser.add_argument('--filename', type=str)
    parser.add_argument('--data_path', type=str,
                        default='generated/200916_2/c0.5_layer27_variables.npy')
    parser.add_argument('--n_data', type=int, default=32)
    parser.add_argument('--space', type=str, default='input',
                        choices=['input', 'hidden'])
    params = parser.parse_args()

    device = torch.device('cuda')

    # save settings
    save_dir = 'generated/' + params.filename + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy('topology_AEn.py', save_dir)

    # images and targets
    d = np.load(params.data_path, allow_pickle=True).item()

    # model
    Z, g, model3 = model_loader.load_encoder(params.basenet, params.target_layer,
                                             params.model_path, device)
    for param in Z.parameters():
        param.requires_grad = False
    for param in g.parameters():
        param.requires_grad = False
    for param in model3.parameters():
        param.requires_grad = False
    Z.eval()
    g.eval()
    model3.eval()

    if params.space == 'input':
        base_classifier = Z
        d_list = np.arange(0, 101, 5)
        x1 = d['datas'][:params.n_data]
        x2 = d['x2'][:params.n_data]
        # x1 = np.array([_x1.detach().cpu().numpy() for _x1 in d['x1'][:params.n_data]])
        # x2 = np.array([_x2.detach().cpu().numpy() for _x2 in d['x2'][:params.n_data]])
        bound = (-float('inf'), float('inf'))
    elif params.space == 'hidden':
        base_classifier = model3

        if params.basenet == 'VGG_stl':
            # d_list = np.arange(0, 3100, 100)
            d_list = np.arange(0, 1050, 50)
        elif params.basenet == 'ResNet50_stl':
            d_list = np.arange(0, 81, 4)
        elif params.basenet == 'ResNet101_stl':
            d_list = np.arange(0, 201, 10)
        elif params.basenet == 'ResNet152_stl':
            d_list = np.arange(0, 241, 12)

        bound = (0, float('inf'))

        _DL = utils.CustomDataset(d['datas'][:params.n_data], d['x2'][:params.n_data])
        # tmp1 = np.array([_x1.detach().cpu().numpy() for _x1 in d['x1'][:params.n_data]])
        # tmp2 = np.array([_x2.detach().cpu().numpy() for _x2 in d['x2'][:params.n_data]])
        # _DL = utils.CustomDataset(tmp1, tmp2)
        _DL = DataLoader(_DL, batch_size=400, shuffle=False)
        for i, (_x1, _x2) in enumerate(_DL):
            _y1 = g(_x1.to(device)).detach().cpu().numpy()
            _y2 = g(_x2.to(device)).detach().cpu().numpy()
            if i == 0:
                x1, x2 = _y1, _y2
            else:
                x1 = np.concatenate([x1, _y1], axis=0)
                x2 = np.concatenate([x2, _y2], axis=0)

    geometry(x1, x2, d_list, base_classifier, bound)
