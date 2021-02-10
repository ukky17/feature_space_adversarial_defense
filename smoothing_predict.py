import os
import math
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

def sample_noise(x, classifier, sigma, lb, num, batch_size, device, n_class):
    def _count_arr(arr, length):
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    with torch.no_grad():
        counts = np.zeros(n_class, dtype=int)
        for _ in range(math.ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device=device) * sigma
            batch_new = batch + noise

            if lb == 0:
                batch_new = torch.where(batch_new >= 0, batch_new, batch - noise)

            preds = torch.argmax(classifier(batch_new), 1)
            counts += _count_arr(preds.detach().cpu().numpy(), n_class)
        return counts

def main(data1, data2, sigmas, classifier, lb, params, device, n_class=10):
    results = np.zeros((len(data1), len(sigmas)))
    ratio = np.zeros(len(sigmas))
    for s in tqdm(range(len(sigmas))):
        DL = utils.CustomDataset(data1, data2)
        DL = DataLoader(DL, batch_size=1, shuffle=False)

        for idx, (d1, d2) in enumerate(DL):
            d1, d2 = d1.to(device), d2.to(device)
            with torch.no_grad():
                pred1 = torch.argmax(classifier(d1), 1).item()
                pred2 = torch.argmax(classifier(d2), 1).item()

            if pred1 == pred2:
                is_correct[idx] = np.nan
                continue

            counts2 = sample_noise(d2, classifier, sigmas[s], lb, params.N0,
                                   params.batch_size, device, n_class)
            sm_pred2 = counts2.argmax().item()
            results[idx, s] = int(sm_pred2 == pred1)

        ratio[s] = 100 * np.sum(results[:, s] == 1) / np.sum(np.isnan(results[:, s]) == 0)

    return results, ratio

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
        results, ratio = main(data_dict['x1'][:params.n_data],
                              data_dict['x2'][:params.n_data],
                              sigmas=np.arange(0, 0.75, 0.05),
                              classifier=Z,
                              lb=-float('inf'),
                              params=params, device=device, n_class=10)

    elif params.space == 'hidden':
        if params.basenet == 'VGG_stl':
            sigmas = np.arange(0, 5.25, 0.25)
        elif params.basenet == 'ResNet50_stl':
            sigmas = np.arange(0, 0.42, 0.02)

        data1, data2 = utils.get_representations(data_dict['x1'][:params.n_data],
                                                 data_dict['x2'][:params.n_data],
                                                 g, device)

        results, ratio = main(data1, data2, sigmas=sigmas, classifier=h, lb=0,
                              params=params, device=device, n_class=10)

    # plot
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.plot(sigmas, ratio)
    ax.set_ylim([0, 105])
    ax.set_xlabel('Sigma')
    ax.set_ylabel('% correct')
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    plt.savefig(save_dir + 'smoothing_predict_' + params.space + '.png')
    plt.close()
    np.save(save_dir + 'smoothing_predict_' + params.space, results)
