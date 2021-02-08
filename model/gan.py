import torch
import torch.nn as nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class GeneratorEncDec(nn.Module):
    def __init__(self, img_size, channels=3):
        super(GeneratorEncDec, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.utils.spectral_norm(
                        nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.utils.spectral_norm(
                    nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.encoder = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            *downsample(512, 512, normalize=False))

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
        out = self.encoder(x)
        out = self.fc(out)
        return self.decoder(out)

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()

        def discriminator_block(in_feat, out_feat):
            layers = [nn.utils.spectral_norm(
                        nn.Conv2d(in_feat, out_feat, 4, 2, 1)),
                      nn.LeakyReLU(0.1, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512))

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(512 * ds_size ** 2, 1)))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)
