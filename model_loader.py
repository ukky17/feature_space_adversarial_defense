import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data

# import generator
import model

def load_encoder(basenet, target_layer, model_path, device):
    if basenet == 'VGG_cifar':
        model1 = model.VGG_cifar()
        model1 = nn.DataParallel(model1, device_ids=range(1))
        _model2, _model3 = model.VGG_cifar2, model.VGG_cifar3
    elif basenet == 'VGG_stl':
        model1 = model.VGG_stl()
        _model2, _model3 = model.VGG_stl2, model.VGG_stl3
    elif basenet == 'ResNet50_stl':
        model1 = model.ResNet_stl(50)
        _model2, _model3 = model.ResNet_stl2, model.ResNet_stl3
    elif basenet == 'ResNet101_stl':
        model1 = model.ResNet_stl(101)
        _model2, _model3 = model.ResNet_stl2, model.ResNet_stl3
    elif basenet == 'ResNet152_stl':
        model1 = model.ResNet_stl(152)
        _model2, _model3 = model.ResNet_stl2, model.ResNet_stl3
    elif basenet == 'CaffeNet':
        model1 = model.CaffeNet()
        _model2, _model3 = model.CaffeNet2, model.CaffeNet3
    elif basenet == 'AlexNet':
        model1 = model.AlexNet()
        _model2, _model3 = model.AlexNet2, model.AlexNet3

    model1.load_state_dict(torch.load(model_path))
    model1.to(device)
    model1.eval()

    model2 = _model2(model1, target_layer)
    model3 = _model3(model1, target_layer)

    return model1, model2, model3

def load_generator(basenet, target_layer, device):
    if basenet == 'CaffeNet':
        if target_layer in {'conv3', 'conv4'}:
            gen = generator.DeePSiMConv34()
        elif target_layer == 'pool5':
            gen = generator.DeePSiMPool5()
        elif target_layer in {'fc6', 'fc7'}:
            gen = generator.DeePSiMFc()

        model_path = 'models/deepsim_pytorch/XDREAM/' + target_layer + '.pt'

    elif basenet == 'VGG_cifar':
        if target_layer == '22':
            gen = generator.DeePSiM_VGG_cifar22()
            model_path = 'models/deepsim_pytorch/VGG_cifar/200622_2.pth'

    elif basenet == 'VGG_stl':
        if target_layer == '22':
            gen = generator.DeePSiM_VGG_stl22()
            model_path = 'models/deepsim_pytorch/VGG_stl/net_g_epoch127.pth'

    gen.load_state_dict(torch.load(model_path))
    gen.to(device)
    gen.eval()
    return gen
