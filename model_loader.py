import torch
import torch.nn as nn

from model import classifier, gan

def load_encoder(basenet, target_layer, model_path, device):
    if basenet == 'VGG_stl':
        model_Z = classifier.VGG_stl()
        get_model_g, get_model_h = classifier.VGG_stl2, classifier.VGG_stl3
    elif basenet == 'ResNet50_stl':
        model_Z = classifier.ResNet_stl(50)
        get_model_g, get_model_h = classifier.ResNet_stl2, classifier.ResNet_stl3

    model_Z.load_state_dict(torch.load(model_path))
    model_Z.to(device)
    model_Z.eval()
    for param in model_Z.parameters():
        param.requires_grad = False

    model_g = get_model_g(model_Z, target_layer)
    model_g.to(device)
    model_g.eval()
    for param in model_g.parameters():
        param.requires_grad = False

    model_h = get_model_h(model_Z, target_layer)
    model_h.to(device)
    model_h.eval()
    for param in model_h.parameters():
        param.requires_grad = False

    return model_Z, model_g, model_h

def load_generator(img_size, device):
    model_G = gan.GeneratorEncDec(img_size)
    model_G.to(device)
    model_G.apply(gan.weights_init_normal)

    model_D = gan.Discriminator(img_size)
    model_D.to(device)
    model_D.apply(gan.weights_init_normal)
    return model_G, model_D
