import os
import math
import shutil
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model_loader
import data
import utils

def sample_noise(x, classifier, sigma, lb, num, n_class):
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

            if lb == 0:
                batch_new = torch.where(batch_new >= 0, batch_new, batch - noise)

            preds = torch.argmax(classifier(batch_new), 1)
            counts += _count_arr(preds.detach().cpu().numpy(), n_class)
        return counts

def main(classifier, data1, data2, sigma, lb, n_class=10):
    DL = utils.CustomDataset(data1, data2)
    DL = DataLoader(DL, batch_size=1, shuffle=False)

    is_accurate = np.zeros(len(data1))
    for idx, (d1, d2) in enumerate(DL):
        d1, d2 = d1.to(device), d2.to(device)
        c1 = torch.argmax(classifier(d1), 1).item()
        c2 = torch.argmax(classifier(d2), 1).item()

        if c1 == c2:
            is_accurate[idx] = np.nan
            continue

        counts2 = sample_noise(d2, classifier, sigma, lb, params.N0, n_class)
        pred2 = counts2.argmax().item()

        is_accurate[idx] = [int(pred2 == c1)]
    return is_accurate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stl10')
    parser.add_argument('--basenet', type=str, choices=['VGG_stl', 'ResNet50_stl'])
    parser.add_argument('--model_path', type=str)

    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--target_layer', type=str)
    parser.add_argument('--batch_size', type=int, default=400)

    parser.add_argument('--n_data', type=int)
    parser.add_argument('--space', type=str, choices=['input', 'hidden'])
    parser.add_argument('--N0', type=int, default=100)

    params = parser.parse_args()

    device = torch.device('cuda')

    # save settings
    save_dir = 'results/' + params.exp_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy('smoothing_predict.py', save_dir)

    # images and targets
    data_dict = np.load(save_dir + 'data_dict.npy', allow_pickle=True).item()

    # model
    Z, g, h = model_loader.load_encoder(params.basenet,
                                        params.target_layer,
                                        params.model_path,
                                        device)

    if params.space == 'input':
        classifier = Z
        sigmas = np.arange(0, 0.75, 0.05)
        lb = -float('inf')

        data1 = data_dict['x1'][:params.n_data]
        data2 = data_dict['x2'][:params.n_data]

    elif params.space == 'hidden':
        classifier = h
        if params.basenet == 'VGG_stl':
            sigmas = np.arange(0, 5.25, 0.25)
        elif params.basenet == 'ResNet50_stl':
            sigmas = np.arange(0, 0.42, 0.02)
        lb = 0

        tmp_DL = utils.CustomDataset(data_dict['x1'][:params.n_data],
                                     data_dict['x2'][:params.n_data])
        tmp_DL = DataLoader(tmp_DL, batch_size=400, shuffle=False)
        for i, (x1, x2) in enumerate(tmp_DL):
            y1 = g(x1.to(device)).detach().cpu().numpy()
            y2 = g(x2.to(device)).detach().cpu().numpy()
            if i == 0:
                data1, data2 = y1, y2
            else:
                data1 = np.concatenate([data1, y1], axis=0)
                data2 = np.concatenate([data2, y2], axis=0)

    accurate_all = np.zeros((len(data1), len(sigmas)))
    for s in tqdm(range(len(sigmas))):
        accurate_all[:, s] = main(classifier, data1, data2, sigmas[s], lb)

    # defense success rate
    ratio = np.zeros(len(sigmas))
    for s in range(len(sigmas)):
        ratio[s] = 100 * np.sum(accurate_all[:, s] == 1) / \
                    np.sum(np.isnan(accurate_all[:, s]) == 0)

    # plot and save
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.plot(sigmas, ratio, '--', linewidth=1)
    ax.set_ylim([0, 105])
    ax.set_xlabel('Sigma')
    ax.set_ylabel('% correct')
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_title(title)
    plt.savefig(save_dir + 'smoothing_' + params.space + '.png')
    plt.close()
    np.save(save_dir + 'cohen_predict_' + params.space, accurate_all)
