import os
from collections import Counter
import shutil
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import model_loader
import data
import utils

def random_dir_torch(dim, device):
    while True:
        vec = torch.randn(dim, device=device)
        r = torch.norm(vec).item()
        if r != 0.0:
            return vec / r

def random_noise(model, data, radii, lb, n_iter, device):
    with torch.no_grad():
        counters = []
        for r in radii:
            # add noise into `data`
            data_noise = torch.zeros((n_iter, ) + tuple(data.shape), device=device)
            for i in range(n_iter):
                noise = random_dir_torch(tuple(data.shape), device) * r
                data_new = data + noise

                if lb == 0:
                    data_new = torch.maximum(data_new, torch.zeros(1, device=device))

                data_noise[i] = data_new

            # predict and count
            preds = torch.argmax(model(data_noise), 1).detach().cpu().numpy()
            counters.append(Counter(preds.tolist()))

    return counters

def main(data1, data2, radii, classifier, lb, device):
    dist_mean = np.mean(np.sum((data1 - data2) ** 2, axis=(1, 2, 3)) ** 0.5)

    DL = utils.CustomDataset(data1, data2)
    DL = DataLoader(DL, batch_size=1, shuffle=False)

    props = np.zeros((3, len(data1), len(radii)))
    for idx, (d1, d2) in tqdm(enumerate(DL)):
        d1, d2 = d1.to(device), d2.to(device)
        with torch.no_grad():
            pred1 = torch.argmax(classifier(d1), 1).item()
            pred2 = torch.argmax(classifier(d2), 1).item()

        if pred1 == pred2:
            props[:, idx, :] = np.nan
            continue

        counters1 = random_noise(classifier, d1[0], radii, lb, 100, device)
        counters2 = random_noise(classifier, d2[0], radii, lb, 100, device)

        props[0, idx] = [counter[pred1] for counter in counters1]
        props[1, idx] = [counter[pred2] for counter in counters2]
        props[2, idx] = [counter[pred1] for counter in counters2]

    return props, dist_mean

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
        radii = np.arange(0, 101, 5)
        props, dist_mean = main(data_dict['x1'][:params.n_data],
                                data_dict['x2'][:params.n_data],
                                radii=radii, classifier=Z, lb=-float('inf'),
                                device=device)

    elif params.space == 'hidden':
        if params.basenet == 'VGG_stl':
            radii = np.arange(0, 2100, 100)
        elif params.basenet == 'ResNet50_stl':
            radii = np.arange(0, 201, 10)

        data1, data2 = utils.get_representations(data_dict['x1'][:params.n_data],
                                                 data_dict['x2'][:params.n_data],
                                                 g, device)

        props, dist_mean = main(data1, data2, radii=radii, classifier=h, lb=0,
                                device=device)

    # plot
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    labels = ['Center: x1, Class: x1', 'Center: x2, Class: x2',
              'Center: x2, Class: x1']
    for (i, fmt, label) in zip([0, 1, 2], ['m', 'c', 'c--'], labels):
        ax.plot(radii, np.nanmean(props[i], axis=0), fmt, label=label)
    ax.axvline(dist_mean, color='k')
    ax.set_ylim([0, 105])
    ax.set_xlabel('r')
    ax.set_ylabel('Class freq (%)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.legend()
    plt.savefig(save_dir + 'geometry_' + params.space + '.png')
    plt.close()
    np.save(save_dir + 'geometry_' + params.space, props)
