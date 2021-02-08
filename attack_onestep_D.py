import os
import sys
import time
import shutil
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.datasets as dst
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

import model_loader
import utils

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class GeneratorEncDec_SN(nn.Module):
    def __init__(self, img_size, mid_fc_type, channels=3):
        super(GeneratorEncDec_SN, self).__init__()

        def downsample(in_feat, out_feat, normalize=True, dropout=0.0):
            layers = [nn.utils.spectral_norm(
                        nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            return layers

        def upsample(in_feat, out_feat, normalize=True, dropout=0.0):
            layers = [nn.utils.spectral_norm(
                    nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU(inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            return layers

        self.encoder = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            *downsample(512, 512, normalize=False))

        self.mid_fc_type = mid_fc_type
        if self.mid_fc_type == 'fc':
            self.neck_size = img_size // 32
            self.mid_size = 512 * self.neck_size ** 2
            self.fc = nn.Sequential(
                nn.Linear(self.mid_size, self.mid_size),
                nn.Linear(self.mid_size, self.mid_size))
        elif self.mid_fc_type == 'channel_wise_fc':
            self.fc = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(512, 512, 1)))

        self.decoder = nn.Sequential(
            *upsample(512, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            *upsample(64, 64),
            nn.utils.spectral_norm(nn.Conv2d(64, channels, 3, 1, 1)),
            nn.Tanh())

    def forward(self, x):
        out = self.encoder(x) # (512, 3, 3)
        if self.mid_fc_type == 'fc':
            out = out.view(out.shape[0], -1) # (512*3*3)
            out = self.fc(out) # (512*3*3)
            out = out.view(out.shape[0], 512, self.neck_size, self.neck_size) # (512, 3, 3)
        elif self.mid_fc_type == 'channel_wise_fc':
            out = self.fc(out)
        return self.decoder(out)

class Discriminator_SN(nn.Module):
    def __init__(self, img_size):
        super(Discriminator_SN, self).__init__()

        def discriminator_block(in_feat, out_feat):
            layers = [nn.utils.spectral_norm(
                          nn.Conv2d(in_feat, out_feat, 4, 2, 1)),
                      nn.LeakyReLU(0.1, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(512 * ds_size ** 2, 1)))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

def _sample_noise(x, base_classifier, sigma, num, n_class):
    outs = torch.zeros(num, x.size()[0], n_class, device=device)
    for i in range(num):
        noise = torch.randn_like(x, device=device) * sigma
        outs[i] = nn.Softmax()(base_classifier(x + noise))

    return torch.mean(outs, 0)

def cohen_predict(classifier, x, sigma, N0, n_class):
    """ x: cuda tensor """

    # noise = torch.randn_like(x, device=device, requires_grad=False) * sigma
    # return classifier(x + noise)
    return _sample_noise(x, classifier, sigma, N0, n_class)

def train_oneepoch(dataloader, params):
    cum_loss = [0] * 5
    x2_all = np.zeros_like(d['datas'])

    for batch_idx, (x1, label) in enumerate(dataloader):
        x1 = x1.to(device)
        label = label.to(device)

        index = label.cpu().view(-1, 1)
        label_onehot = torch.zeros(x1.size()[0], n_class)
        label_onehot.scatter_(1, index, 1)
        label_onehot = label_onehot.to(device)

        ones = torch.ones((x1.size()[0], 1), dtype=torch.float32, device=device)
        zeros = torch.zeros((x1.size()[0], 1), dtype=torch.float32, device=device)
        zero_t = torch.zeros(1, device=device)

        ## update Generator
        optimizer_g.zero_grad()

        if params.dataset == 'stl10':
            x2 = net_g(x1)
        elif params.dataset == 'cifar10':
            x2 = (net_g(x1) + 1) / 2

        output1 = Z(x2)
        output2 = cohen_predict(Z, x2, params.cohen_sigma, params.cohen_N0, n_class)
        fake_logit = net_d(x2)
        real1 = torch.max(torch.mul(output1, label_onehot), 1)[0]
        other1 = torch.max(torch.mul(output1, (1-label_onehot))-label_onehot*10000, 1)[0]
        real2 = torch.max(torch.mul(output2, label_onehot), 1)[0]
        other2 = torch.max(torch.mul(output2, (1-label_onehot))-label_onehot*10000, 1)[0]

        loss_hidden = torch.mean((g(x2) - g(x1)) ** 2)
        loss_target = torch.mean(torch.max(real1 - other1, zero_t))
        loss_smooth = torch.mean(torch.max(real2 - other2, zero_t))
        loss_adv = -torch.mean(fake_logit)
        loss_g = loss_hidden + params.lambda_c * loss_target + \
                 params.lambda_smooth * loss_smooth + \
                 params.lambda_adv * loss_adv

        loss_g.backward()
        optimizer_g.step()

        ## update Discriminator
        optimizer_d.zero_grad()

        fake_logit = net_d(x2.detach())
        real_logit = net_d(x1)
        loss_d = nn.ReLU()(1.0 - real_logit).mean() + nn.ReLU()(1.0 + fake_logit).mean()

        loss_d.backward()
        optimizer_d.step()

        # stock loss
        loss_list = [loss_hidden, loss_target, loss_adv, loss_g, loss_d]
        for i, _loss in enumerate(loss_list):
            cum_loss[i] += _loss.item() * len(x1)

        # stock x2
        _idx = batch_idx * params.batch_size
        x2_all[_idx: _idx + len(x2)] = x2.detach().cpu().numpy()

    for i in range(len(cum_loss)):
        cum_loss[i] /= len(x2_all)
    return cum_loss, x2_all

def plotter(_x1_all, _x2_all, _epoch, device):
    _c = Z(torch.from_numpy(_x1_all).to(device))
    t_x1_all = torch.argmax(_c, 1).detach().cpu().numpy()
    _c = Z(torch.from_numpy(_x2_all).to(device))
    t_x2_all = torch.argmax(_c, 1).detach().cpu().numpy()
    plt.figure(figsize=(20, 20))
    for i in range(min(40, len(_x1_all))):
        for j, _imgs, _title, c in zip(range(2), [_x1_all, _x2_all],
                                       ["x1", "x2"], [t_x1_all, t_x2_all]):
            _img = np.transpose(_imgs[i], (1, 2, 0))
            _img = (_img - _img.min()) / (_img.max() - _img.min())
            ax = plt.subplot(10, 8, 2 * i + j + 1)
            ax.imshow(_img)
            ax.set_title(_title + '  ' + labelStr[c[i]])
            plt.axis('off')
    # plt.tight_layout()
    plt.savefig(save_dir + 'c' + str(params.lambda_c) + '_layer' + str(params.target_layer)\
                + '_epoch' + str(_epoch) + '.png')
    plt.close()

def acc_under_attack(x1_all, x2_all, labels, device):
    # prepare the dataloader
    DL = utils.CustomDataset(x2_all, labels)
    DL = DataLoader(DL, batch_size=params.batch_size, shuffle=False)

    correct = 0
    for (x2, output) in DL:
        x2, label = x2.to(device), output.to(device)

        with torch.no_grad():
            _, idx = torch.max(Z(x2), 1)
        correct += torch.sum(label.eq(idx)).item()

    distort = (np.sum((x2_all - x1_all) ** 2)) ** 0.5
    return correct / len(x1_all), distort / len(x1_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stl10',
                        choices=['cifar10', 'stl10'])
    parser.add_argument('--basenet', type=str, default='VGG_stl',
                        choices=['VGG_cifar', 'VGG_stl',
                                 'ResNet50_stl', 'ResNet101_stl', 'ResNet152_stl'])
    parser.add_argument('--model_path', type=str, default='models/vgg_stl.pth')

    parser.add_argument('--filename', type=str)
    parser.add_argument('--target_layer', type=str, default='27')

    parser.add_argument('--epochs', type=int, default=1000) # 5000
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--lambda_c', type=float, default=0.5)
    parser.add_argument('--lambda_adv', type=float, default=0.5)
    parser.add_argument('--lambda_smooth', type=float, default=0.5)

    parser.add_argument('--mid_fc_type', type=str, default='channel_wise_fc',
                        choices=['none', 'fc', 'channel_wise_fc'])

    parser.add_argument('--optimizer_type', type=str, default='Adam')
    parser.add_argument('--lr_g', type=float, default=0.0001)
    parser.add_argument('--lr_d', type=float, default=0.0004)
    parser.add_argument('--lr_decay', type=float, default=0.96)
    parser.add_argument('--lr_decay_epoch', type=int, default=64)

    parser.add_argument('--cohen_N0', default=10, type=int)
    parser.add_argument('--cohen_sigma', default=0.25, type=float)

    params = parser.parse_args()
    print(params)

    device = torch.device('cuda')

    # save settings
    save_dir = 'generated/' + params.filename + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy('attack_onestep_D.py', save_dir)

    # images and targets cifar10: [0, 1], stl10: [-1, 1]
    if params.dataset == 'cifar10':
        transform_test = tfs.Compose([tfs.ToTensor()])
        data_test = dst.CIFAR10('data/cifar10/', download=False, train=False,
                                transform=transform_test)
        dataloader = DataLoader(data_test, batch_size=params.batch_size, shuffle=False)
        labelStr = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                    "frog", "horse", "ship", "truck"]
        n_class, img_size = 10, 32
    elif params.dataset == 'stl10':
        transform_test  = tfs.Compose([tfs.ToTensor(),
                                       tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])
        data_test = dst.STL10('data/STL10', download=True, split='test',
                                transform=transform_test)
        dataloader = DataLoader(data_test, batch_size=params.batch_size, shuffle=False)
        labelStr = ["airplane", "bird", "car", "cat", "deer", "dog",
                    "horse", "monkey", "ship", "truck"]
        n_class, img_size = 10, 96

    # model
    Z, g, model3 = model_loader.load_encoder(params.basenet, params.target_layer,
                                             params.model_path, device)
    net_d = Discriminator_SN(img_size)
    net_g = GeneratorEncDec_SN(img_size, params.mid_fc_type)
    net_d.to(device)
    net_d.apply(weights_init_normal)
    net_g.to(device)
    net_g.apply(weights_init_normal)
    for param in Z.parameters():
        param.requires_grad = False
    for param in g.parameters():
        param.requires_grad = False
    for param in model3.parameters():
        param.requires_grad = False
    Z.eval()
    g.eval()
    model3.eval()

    # select only successfully-classified images
    d = dict()
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            out = Z.forward(data.to(device))
            data = data.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            y = torch.argmax(out, 1).detach().cpu().numpy()
            data = data[y == target]
            target = target[y == target]

            if i == 0:
                d['datas'] = data
                d['targets'] = target
            else:
                d['datas'] = np.concatenate([d['datas'], data], axis=0)
                d['targets'] = np.concatenate([d['targets'], target], axis=0)

    print('Original x1: ', d['datas'].shape)
    print('Target: ', d['targets'].shape)

    DL = utils.CustomDataset(d['datas'], d['targets'])
    DL = DataLoader(DL, batch_size=params.batch_size, shuffle=False)

    # prepare optimizers
    if params.optimizer_type == 'Adam':
        optimizer_g = optim.Adam(net_g.parameters(), lr=params.lr_g, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, net_d.parameters()),
                                 lr=params.lr_d, betas=(0.0,0.9))
    elif params.optimizer_type == 'RMSprop':
        optimizer_g = optim.RMSprop(net_g.parameters(), lr=params.lr_g)
        optimizer_d = optim.RMSprop(net_d.parameters(), lr=params.lr_d)

    loss_names = ['hidden', 'target', 'adv', 'g', 'd']
    losses = np.zeros((params.epochs, len(loss_names)))
    print()
    print(loss_names)
    for epoch in range(params.epochs):
        start = time.time()
        l1, x2_all = train_oneepoch(DL, params)
        run_time = time.time() - start

        losses[epoch, :] = l1
        print('Epoch {}, Time: {:.2f} s'.format(epoch, run_time))
        print('  Loss: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(\
                l1[0], l1[1], l1[2], l1[3], l1[4]))

        if epoch % params.lr_decay_epoch == 0:
            params.lr_d *= params.lr_decay
            params.lr_g *= params.lr_decay

        if epoch % 64 == 0:
            # plot the first 40 images and save weights
            plotter(d['datas'][:40], x2_all[:40], epoch, device)

    # plot the first 40 images and save weights
    plotter(d['datas'][:40], x2_all[:40], epoch, device)

    # save
    d['x2'] = x2_all
    np.save(save_dir + 'c' + str(params.lambda_c) + '_layer' + str(params.target_layer)\
            + '_variables', d)

    # accuracy
    acc, avg_distort = acc_under_attack(d['datas'], d['x2'], d['targets'], device)
    print("c, test accuracy, distort: {}, {}, {}".format(params.lambda_c, acc, avg_distort))
