import os
from collections import Counter
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

def random_dir_torch(dim):
    with torch.no_grad():
        while True:
            vec = torch.randn(dim, device=device, dtype=torch.float)
            r = torch.norm(vec).item()
            if r != 0.0:
                return vec / r

def random_noise(model, img, radii, lb, n_iter):
    counters = []
    for dd in radii:
        imgs_noise = []
        for _ in range(n_iter):
            noise = random_dir_torch(img.shape).detach().cpu().numpy() * dd
            img_noise = img + noise

            if lb == 0:
                img_noise = np.where(img_noise >= 0, img_noise, img - noise)

            imgs_noise.append(img_noise)

        # dataloader
        imgs_noise = np.array(imgs_noise)
        DL = utils.CustomDataset(imgs_noise, np.zeros(n_iter))
        DL = DataLoader(DL, batch_size=400, shuffle=False)

        with torch.no_grad():
            classes = []
            for (img_noise, _) in _DL:
                preds = torch.argmax(model(img_noise.to(device)), 1)
                classes += preds.detach().cpu().numpy().tolist()

            counters.append(Counter(classes))
    return counters

def main(data1, data2, radii, classifier, lb):
    dist_mean = np.mean(np.sum((data1 - data2) ** 2, axis=(1, 2, 3)) ** 0.5)
    print(dist_mean)

    DL = utils.CustomDataset(data1, data2)
    DL = DataLoader(DL, batch_size=1, shuffle=False)

    props = np.zeros((4, len(x1), len(radii)))
    for idx, (d1, d2) in tqdm(enumerate(DL)):
        c1 = torch.argmax(classifier(d1.to(device)), 1).item()
        c2 = torch.argmax(classifier(d2.to(device)), 1).item()

        if c1 == c2:
            props[:, idx, :] = np.nan
            continue

        d1 = d1.detach().cpu().numpy()[0]
        d2 = d2.detach().cpu().numpy()[0]

        counters1 = random_noise(classifier, d1, radii, lb, n_iter=100)
        counters2 = random_noise(classifier, d2, radii, lb, n_iter=100)

        props[0, idx] = [counter[_c1] for counter in counters1]
        props[1, idx] = [counter[_c2] for counter in counters2]
        props[2, idx] = [counter[_c1] for counter in counters2]
        props[3, idx] = [counter[_c2] for counter in counters1]

    plt = plt.figure()
    ax = plt.subplot(1, 1, 1)
    labels = ["x: ori, cla: ori", "x: adv, cla: adv", "x: adv, cla: ori"]
    for (i, col, fmt, label) in zip([0, 1, 2], ['m', 'c', 'c'], ['', '', '--'],
                                    labels):
        ax.plot(radii, np.nanmean(props[i], axis=0), col+fmt, label=label, linewidth=1)
    ax.axvline(dist_mean, color='k', linewidth=1)
    ax.set_ylim([0, 105])
    ax.set_xlabel('r')
    ax.set_ylabel('Class freq (%)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_title(params.space + ' space')
    ax.legend()
    plt.savefig(save_dir + 'geometry_' + params.space + '.png')
    plt.close()
    np.save(save_dir + 'geometry_' + params.space, props)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stl10')
    parser.add_argument('--basenet', type=str, choices=['VGG_stl', 'ResNet50_stl'])
    parser.add_argument('--model_path', type=str)

    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--target_layer', type=str)

    parser.add_argument('--n_data', type=int)
    parser.add_argument('--space', type=str, choices=['input', 'hidden'])

    params = parser.parse_args()

    device = torch.device('cuda')

    # save settings
    save_dir = 'results/' + params.exp_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy('geometry.py', save_dir)

    # images and targets
    data_dict = np.load(save_dir + 'data_dict.npy', allow_pickle=True).item()

    # model
    Z, g, h = model_loader.load_encoder(params.basenet,
                                        params.target_layer,
                                        params.model_path,
                                        device)

    if params.space == 'input':
        classifier = Z
        radii = np.arange(0, 101, 5)
        lb = -float('inf')

        data1 = d['x1'][:params.n_data]
        data2 = d['x2'][:params.n_data]

    elif params.space == 'hidden':
        classifier = h
        if params.basenet == 'VGG_stl':
            radii = np.arange(0, 1050, 50)
        elif params.basenet == 'ResNet50_stl':
            radii = np.arange(0, 81, 4)
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

    main(data1, data2, radii, classifier, lb)
