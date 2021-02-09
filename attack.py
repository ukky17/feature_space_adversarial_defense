import os
import sys
import time
import shutil
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import model_loader
import data
import utils

def smoothing(classifier, x, sigma, N0, n_class):
    outs = torch.zeros(N0, x.size()[0], n_class, device=device)
    for i in range(N0):
        noise = torch.randn_like(x, device=device) * sigma
        outs[i] = nn.Softmax()(classifier(x + noise))

    return torch.mean(outs, 0)

def train_oneepoch(dataloader, params, device):
    cum_loss = [0] * len(loss_names)
    x2_all = np.zeros_like(data_dict['x1'])

    zero_tensor = torch.zeros(1, device=device)
    for batch_idx, (x1, label) in enumerate(dataloader):
        x1, label = x1.to(device), label.to(device)

        index = label.cpu().view(-1, 1)
        label_onehot = torch.zeros(x1.size()[0], n_class)
        label_onehot.scatter_(1, index, 1)
        label_onehot = label_onehot.to(device)

        # update Generator
        optimizer_G.zero_grad()

        x2 = G(x1)

        output1 = Z(x2)
        output2 = smoothing(Z, x2, params.sigma, params.N0, n_class)
        fake_logit = D(x2)
        real1 = torch.max(torch.mul(output1, label_onehot), 1)[0]
        other1 = torch.max(torch.mul(output1, (1-label_onehot))-label_onehot*10000, 1)[0]
        real2 = torch.max(torch.mul(output2, label_onehot), 1)[0]
        other2 = torch.max(torch.mul(output2, (1-label_onehot))-label_onehot*10000, 1)[0]

        loss_hidden = torch.mean((g(x2) - g(x1)) ** 2)
        loss_class = torch.mean(torch.maximum(real1 - other1, zero_tensor))
        loss_smooth = torch.mean(torch.maximum(real2 - other2, zero_tensor))
        loss_adv = -torch.mean(fake_logit)
        loss_G = loss_hidden + params.lambda_c * loss_class + \
                 params.lambda_adv * loss_adv + \
                 params.lambda_smooth * loss_smooth

        loss_G.backward()
        optimizer_G.step()

        # update Discriminator
        optimizer_D.zero_grad()

        fake_logit = D(x2.detach())
        real_logit = D(x1)
        loss_D = nn.ReLU()(1.0 - real_logit).mean() + nn.ReLU()(1.0 + fake_logit).mean()

        loss_D.backward()
        optimizer_D.step()

        # stock loss
        loss_list = [loss_hidden, loss_class, loss_adv, loss_smooth, loss_G, loss_D]
        for i, l in enumerate(loss_list):
            cum_loss[i] += l.item() * len(x1)

        # stock x2
        idx = batch_idx * params.batch_size
        x2_all[idx: idx + len(x2)] = x2.detach().cpu().numpy()

    for i in range(len(cum_loss)):
        cum_loss[i] /= len(x2_all)
    return cum_loss, x2_all

def plot_ex(x1, x2, epoch, device):
    # predict on x1 and x2
    with torch.no_grad():
        out1 = Z(torch.from_numpy(x1).to(device))
        pred1 = torch.argmax(out1, 1).detach().cpu().numpy()
        out2 = Z(torch.from_numpy(x2).to(device))
        pred2 = torch.argmax(out2, 1).detach().cpu().numpy()

    # plot
    plt.figure(figsize=(20, 20))
    for i in range(min(40, len(x1))):
        for j, imgs, title, pred in zip(range(2), [x1, x2], ['x1', 'x2'], [pred1, pred2]):
            img = np.transpose(imgs[i], (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())
            ax = plt.subplot(10, 8, 2 * i + j + 1)
            ax.imshow(img)
            ax.set_title(title + ': ' + labelStr[pred[i]])
            plt.axis('off')
    plt.savefig(save_dir + 'img_epoch' + str(epoch) + '.png')
    plt.close()

def get_acc(data_dict, device):
    DL = utils.CustomDataset(data_dict['x2'], data_dict['target'])
    DL = DataLoader(DL, batch_size=params.batch_size, shuffle=False)

    with torch.no_grad():
        correct = 0
        for (x2, output) in DL:
            x2, label = x2.to(device), output.to(device)
            pred = torch.argmax(Z(x2), 1)
            correct += torch.sum(label.eq(idx)).item()

    return correct / len(data_dict['x1'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['stl10'])
    parser.add_argument('--basenet', type=str, choices=['VGG_stl', 'ResNet50_stl'])
    parser.add_argument('--model_path', type=str)

    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--target_layer', type=str)

    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--lambda_c', type=float)
    parser.add_argument('--lambda_adv', type=float)
    parser.add_argument('--lambda_smooth', type=float)

    parser.add_argument('--lr_G', type=float, default=0.0001)
    parser.add_argument('--lr_D', type=float, default=0.0004)
    parser.add_argument('--lr_decay', type=float, default=0.96)
    parser.add_argument('--lr_decay_epoch', type=int, default=64)

    parser.add_argument('--N0', default=10, type=int)
    parser.add_argument('--sigma', default=0.15, type=float)

    params = parser.parse_args()
    print(params)
    print()

    device = torch.device('cuda')

    # save settings
    save_dir = 'results/' + params.exp_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy('attack.py', save_dir)

    # images and targets
    data_full, labelStr, n_class, img_size = data.data_loader(params.dataset,
                                                              params.batch_size)

    # model
    Z, g, h = model_loader.load_encoder(params.basenet,
                                        params.target_layer,
                                        params.model_path,
                                        device)
    G, D = model_loader.load_generator(img_size[1], device)

    # select only successfully-classified images
    data_dict = data.select_correct(data_full, Z, device)

    DL = utils.CustomDataset(data_dict['x1'], data_dict['target'])
    DL = DataLoader(DL, batch_size=params.batch_size, shuffle=False)

    # optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=params.lr_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()),
                             lr=params.lr_D, betas=(0.0, 0.9))

    loss_names = ['hidden', 'class', 'GAN_G', 'smooth', 'total_G', 'total_D']
    losses = np.zeros((params.epochs, len(loss_names)))
    print(loss_names)
    for epoch in range(params.epochs):
        start = time.time()
        loss, x2_all = train_oneepoch(DL, params, device)
        run_time = time.time() - start

        losses[epoch] = loss
        print('Epoch {}, Time: {:.2f} s'.format(epoch, run_time))
        print(('  Loss' + ': {:.6f}' * len(loss)).format(*loss))

        if epoch % params.lr_decay_epoch == 0:
            params.lr_D *= params.lr_decay
            params.lr_G *= params.lr_decay

        if epoch % 64 == 0:
            # plot the first 40 images and save weights
            plot_ex(data_dict['x1'][:40], x2_all[:40], epoch, device)

    plot_ex(data_dict['x1'][:40], x2_all[:40], epoch, device)

    # save
    data_dict['x2'] = x2_all
    np.save(save_dir + 'data_dict', data_dict)

    # accuracy
    acc, avg_distort = get_acc(data_dict, device)
    print('accuracy: {}'.format(acc))
