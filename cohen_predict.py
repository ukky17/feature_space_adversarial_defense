""" evaluate a smoothed classifier on a dataset
"""
import os
import sys
import math
import shutil
import argparse
import json
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

def _sample_noise(x, base_classifier, sigma, bound, num, n_class):
    def _count_arr(arr, length):
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    with torch.no_grad():
        counts = np.zeros(n_class, dtype=int)
        for _ in range(math.ceil(num / params.batch_size)):
            this_batch_size = min(params.batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device=device) * sigma
            batch_new = batch + noise

            if bound[0] != -float('inf'):
                batch_new = torch.where(batch_new >= bound[0], batch_new, batch - noise)
            elif bound[1] != float('inf'):
                batch_new = torch.where(batch_new <= bound[1], batch_new, batch - noise)

            predictions = base_classifier(batch_new).argmax(1)
            counts += _count_arr(predictions.detach().cpu().numpy(), n_class)
        return counts

def main(base_classifier, x1, x2, sigma, bound, n_class):
    DL = utils.CustomDataset(x1, x2)
    DL = DataLoader(DL, batch_size=1, shuffle=False)

    is_accurate = np.zeros((4, len(x1)))
    for idx, (_x1, _x2) in enumerate(DL):
        _x1, _x2 = _x1.to(device), _x2.to(device)
        _c1 = torch.argmax(base_classifier(_x1), dim=1).item()
        _c2 = torch.argmax(base_classifier(_x2), dim=1).item()

        if _c1 == _c2:
            is_accurate[:, idx] = [np.nan, np.nan, np.nan, np.nan]
            continue

        counts1 = _sample_noise(_x1, base_classifier, sigma, bound, params.N0,
                                n_class)
        pred1 = counts1.argmax().item()
        counts2 = _sample_noise(_x2, base_classifier, sigma, bound, params.N0,
                                n_class)
        pred2 = counts2.argmax().item()

        is_accurate[:, idx] = [int(pred1 == _c1), int(pred2 == _c2),
                               int(pred2 == _c1), int(pred1 == _c2)]
    return is_accurate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stl10',
                        choices=['cifar10', 'imagenet', 'stl10'])
    parser.add_argument('--basenet', type=str, default='VGG_stl')
    parser.add_argument('--model_path', type=str, default='models/vgg_stl.pth')
    parser.add_argument('--target_layer', type=str, default='27')
    parser.add_argument("--batch_size", type=int, default=400)

    parser.add_argument('--filename', type=str)
    parser.add_argument('--data_path', type=str,
                        default='generated/200916_2/c0.5_layer27_variables.npy')
    parser.add_argument('--n_data', type=int, default=128)
    parser.add_argument('--space', type=str, default='input',
                        choices=['input', 'hidden'])
    parser.add_argument("--N0", type=int, default=100)

    params = parser.parse_args()

    device = torch.device('cuda')

    # save settings
    save_dir = 'generated/' + params.filename + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy('cohen_predict.py', save_dir)

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

    if params.dataset == 'cifar10':
        n_class, img_size = 10, 32
    elif params.dataset == 'stl10':
        n_class, img_size = 10, 96
    elif params.dataset == 'imagenet':
        n_class, img_size = 1000, 224

    if params.space == 'input':
        base_classifier = Z
        sigmas = np.arange(0, 0.75, 0.05)
        x1 = d['datas'][:params.n_data]
        x2 = d['x2'][:params.n_data]
        # x1 = np.array([_x1.detach().cpu().numpy() for _x1 in d['x1'][:params.n_data]])
        # x2 = np.array([_x2.detach().cpu().numpy() for _x2 in d['x2'][:params.n_data]])
        bound = (-float('inf'), float('inf'))
    elif params.space == 'hidden':
        base_classifier = model3

        if params.basenet == 'VGG_stl':
            sigmas = np.arange(0, 5.25, 0.25)
        elif params.basenet == 'ResNet50_stl':
            sigmas = np.arange(0, 0.42, 0.02)
        elif params.basenet == 'ResNet101_stl':
            sigmas = np.arange(0, 1.05, 0.05)
        elif params.basenet == 'ResNet152_stl':
            sigmas = np.arange(0, 1.26, 0.06)

        bound = (0, float('inf'))

        _DL = utils.CustomDataset(d['datas'][:params.n_data],
                                  d['x2'][:params.n_data])
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

    accurate_all = np.zeros((4, len(x1), len(sigmas)))
    for s in tqdm(range(len(sigmas))):
        is_accurate = main(base_classifier, x1, x2, sigmas[s], bound, n_class)
        accurate_all[:, :, s] = is_accurate

    ratio = 100 * np.nansum(accurate_all, axis=1) / \
            np.sum(np.isnan(accurate_all[0]) == 0, axis=0)

    # plot
    fig = plt.figure()
    plt.plot(sigmas, ratio[0], label='x: ori, cla: ori', c='m')
    plt.plot(sigmas, ratio[1], label='x: adv, cla: adv', c='c')
    plt.plot(sigmas, ratio[2], '--', label='x: adv, cla: ori', c='c')
    plt.ylim([0, 105])
    plt.xlabel('Sigma')
    plt.ylabel('% correct')
    plt.legend()
    plt.savefig(save_dir + 'smoothing_' + params.space + '.png')
    plt.close()
    np.save(save_dir + 'cohen_predict_' + params.space, accurate_all)
