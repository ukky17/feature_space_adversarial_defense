import torch
import torch.nn as nn

from layer import Noise

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
       512, 512, 512, 'M']

class VGG_cifar(nn.Module):
    def __init__(self):
        super(VGG_cifar, self).__init__()
        self.classifier = nn.Linear(512, 10)
        self.features = self._make_layers(cfg)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# input --> hidden layer
class VGG_cifar2(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(VGG_cifar2, self).__init__()
        features = list(model1.module.features)
        self.features = nn.ModuleList(features).eval()
        self.target_layer = int(target_layer)

    def forward(self, x):
        if self.target_layer == -1:
            return x

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.target_layer:
                return x

# hidden layer --> output layer
class VGG_cifar3(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(VGG_cifar3, self).__init__()
        features = list(model1.module.features)
        self.features = nn.ModuleList(features).eval()
        self.target_layer = int(target_layer)
        self.classifier = model1.module.classifier

    def forward(self, x):
        for ii, model in enumerate(self.features):
            if ii > self.target_layer:
                x = model(x)

        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out

class VGG_stl(nn.Module):
    def __init__(self):
        super(VGG_stl, self).__init__()
        self.features = self._make_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

# input --> hidden layer
class VGG_stl2(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(VGG_stl2, self).__init__()
        features = list(model1.features)
        self.features = nn.ModuleList(features).eval()
        self.target_layer = int(target_layer)

    def forward(self, x):
        if self.target_layer == -1:
            return x

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.target_layer:
                return x

# hidden layer --> output layer
class VGG_stl3(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(VGG_stl3, self).__init__()
        features = list(model1.features)
        self.features = nn.ModuleList(features).eval()
        self.avgpool = model1.avgpool
        classifier = list(model1.classifier)
        self.classifier = nn.ModuleList(classifier).eval()
        self.target_layer = int(target_layer)

    def forward(self, x):
        for ii, model in enumerate(self.features):
            if ii > self.target_layer:
                x = model(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        for ii, model in enumerate(self.classifier):
            x = model(x)
        return x

class VGG_noise_stl(nn.Module):
    def __init__(self, noise_layer, noise_std):
        super(VGG_noise_stl, self).__init__()
        self.noise_layer = noise_layer
        self.noise_std = noise_std

        self.features = self._make_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x

        if self.noise_layer != -1:
            layers.insert(self.noise_layer, Noise(self.noise_std))
        return nn.Sequential(*layers)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_stl(nn.Module):
    def __init__(self, n_layers, block=Bottleneck, num_classes=10,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet_stl, self).__init__()

        if n_layers == 50:
            layers = [3, 4, 6, 3]
        elif n_layers == 101:
            layers = [3, 4, 23, 3]
        elif n_layers == 152:
            layers = [3, 8, 36, 3]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# input --> hidden layer
class ResNet_stl2(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(ResNet_stl2, self).__init__()
        self.model1 = model1
        self.target_layer = target_layer

    def forward(self, x):
        if self.target_layer == '-1':
            return x

        x = self.model1.conv1(x)
        x = self.model1.bn1(x)
        x = self.model1.relu(x)
        x = self.model1.maxpool(x)

        x = self.model1.layer1(x)
        if self.target_layer == 'layer1':
            return x
        x = self.model1.layer2(x)
        if self.target_layer == 'layer2':
            return x
        x = self.model1.layer3(x)
        if self.target_layer == 'layer3':
            return x
        x = self.model1.layer4(x)
        if self.target_layer == 'layer4':
            return x

        x = self.model1.avgpool(x)
        if self.target_layer == 'avgpool':
            return x
        x = torch.flatten(x, 1)
        x = self.model1.fc(x)

        return x

# hidden layer --> output layer
class ResNet_stl3(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(ResNet_stl3, self).__init__()
        self.model1 = model1
        self.target_layer = target_layer

    def forward(self, x):
        if self.target_layer in {'-1'}:
            x = self.model1.conv1(x)
            x = self.model1.bn1(x)
            x = self.model1.relu(x)
            x = self.model1.maxpool(x)

            x = self.model1.layer1(x)

        if self.target_layer in {'-1', 'layer1'}:
            x = self.model1.layer2(x)
        if self.target_layer in {'-1', 'layer1', 'layer2'}:
            x = self.model1.layer3(x)
        if self.target_layer in {'-1', 'layer1', 'layer2', 'layer3'}:
            x = self.model1.layer4(x)

        if self.target_layer in {'-1', 'layer1', 'layer2', 'layer3', 'layer4'}:
            x = self.model1.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model1.fc(x)

        return x

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# input --> hidden layer
class AlexNet2(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(AlexNet2, self).__init__()
        features = list(model1.features)
        classifier = list(model1.classifier)
        self.features = nn.ModuleList(features).eval()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.ModuleList(classifier).eval()
        self.target_layer = target_layer
        self.mapper1 = {'input': -1, 'conv1': 2, 'conv2': 5, 'conv3': 7,
                        'conv4': 9, 'pool5': 12}
        self.mapper2 = {'fc6': 2, 'fc7': 5, 'fc8': 6}

    def forward(self, x):
        if self.target_layer == 'input':
            return x

        for ii, model in enumerate(self.features):
            x = model(x)
            if self.target_layer in self.mapper1 and \
                ii == self.mapper1[self.target_layer]:
                return x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        for ii, model in enumerate(self.classifier):
            x = model(x)
            if self.target_layer in self.mapper2 and \
                ii == self.mapper2[self.target_layer]:
                return x

# hidden layer --> output layer
class AlexNet3(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(AlexNet3, self).__init__()
        features = list(model1.features)
        classifier = list(model1.classifier)
        self.features = nn.ModuleList(features).eval()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.ModuleList(classifier).eval()
        self.target_layer = target_layer
        self.mapper1 = {'input': -1, 'conv1': 2, 'conv2': 5, 'conv3': 7,
                        'conv4': 9, 'pool5': 12}
        self.mapper2 = {'fc6': 2, 'fc7': 5, 'fc8': 6}

    def forward(self, x):
        if self.target_layer in self.mapper1:
            for ii, model in enumerate(self.features):
                if ii > self.mapper1[self.target_layer]:
                    x = model(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            for ii, model in enumerate(self.classifier):
                x = model(x)
            return x

        elif self.target_layer in self.mapper2:
            for ii, model in enumerate(self.classifier):
                if ii > self.mapper2[self.target_layer]:
                    x = model(x)
            return x

class CaffeNet(nn.Module):
    def __init__(self):
        super(CaffeNet, self).__init__()
        self.conv1 = self.__conv(2, name='conv1',
                                 in_channels=3, out_channels=96,
                                 kernel_size=11, stride=4, groups=1)
        self.conv2 = self.__conv(2, name='conv2',
                                 in_channels=96, out_channels=256,
                                 kernel_size=5, stride=1, padding=2, groups=2)
        self.conv3 = self.__conv(2, name='conv3',
                                 in_channels=256, out_channels=384,
                                 kernel_size=3, stride=1, padding=1, groups=1)
        self.conv4 = self.__conv(2, name='conv4',
                                 in_channels=384, out_channels=384,
                                 kernel_size=3, stride=1, padding=1, groups=2)
        self.conv5 = self.__conv(2, name='conv5',
                                 in_channels=384, out_channels=256,
                                 kernel_size=3, stride=1, padding=1, groups=2)
        self.fc6_1 = self.__dense(name = 'fc6_1',
                                  in_features=9216, out_features=4096)
        self.fc7_1 = self.__dense(name = 'fc7_1',
                                  in_features=4096, out_features=4096)
        self.fc8_1 = self.__dense(name = 'fc8_1',
                                  in_features=4096, out_features=1000)

        self.features = [
            self.conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0),
            self.conv2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0),
            self.conv3,
            nn.ReLU(inplace=True),
            self.conv4,
            nn.ReLU(inplace=True),
            self.conv5,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]

        self.classifier = [
            self.fc6_1,
            nn.ReLU(inplace=True),
            self.fc7_1,
            nn.ReLU(inplace=True),
            self.fc8_1
        ]

    def forward(self, x):
        for feat in self.features:
            x = feat(x)
        x = x.view(x.size(0), -1)
        for cla in self.classifier:
            x = cla(x)
        return x

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        return layer

# input --> hidden layer
class CaffeNet2(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(CaffeNet2, self).__init__()
        self.features = model1.features
        self.classifier = model1.classifier
        self.target_layer = target_layer
        self.mapper1 = {'input': -1, 'norm1': 3, 'norm2': 7, 'conv3': 9,
                        'conv4': 11, 'pool5': 14}
        self.mapper2 = {'fc6': 1, 'fc7': 3, 'fc8': 4}

    def forward(self, x):
        if self.target_layer == 'input':
            return x

        for ii, model in enumerate(self.features):
            x = model(x)
            if self.target_layer in self.mapper1 and \
                ii == self.mapper1[self.target_layer]:
                return x

        x = x.view(x.size(0), -1)

        for ii, model in enumerate(self.classifier):
            x = model(x)
            if self.target_layer in self.mapper2 and \
                ii == self.mapper2[self.target_layer]:
                return x

# hidden layer --> output layer
class CaffeNet3(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(CaffeNet3, self).__init__()
        self.features = model1.features
        self.classifier = model1.classifier
        self.target_layer = target_layer
        self.mapper1 = {'input': -1, 'norm1': 3, 'norm2': 7, 'conv3': 9,
                        'conv4': 11, 'pool5': 14}
        self.mapper2 = {'fc6': 1, 'fc7': 3, 'fc8': 4}

    def forward(self, x):
        if self.target_layer in self.mapper1:
            for ii, model in enumerate(self.features):
                if ii > self.mapper1[self.target_layer]:
                    x = model(x)

            x = x.view(x.size(0), -1)

            for ii, model in enumerate(self.classifier):
                x = model(x)
            return x

        elif self.target_layer in self.mapper2:
            for ii, model in enumerate(self.classifier):
                if ii > self.mapper2[self.target_layer]:
                    x = model(x)
            return x
