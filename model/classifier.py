import torch
import torch.nn as nn

class VGG_stl(nn.Module):
    def __init__(self):
        super(VGG_stl, self).__init__()

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
               512, 512, 512, 'M', 512, 512, 512, 'M']
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
    def __init__(self, model_full, target_layer):
        super(VGG_stl2, self).__init__()
        features = list(model_full.features)
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
    def __init__(self, model_full, target_layer):
        super(VGG_stl3, self).__init__()
        features = list(model_full.features)
        self.features = nn.ModuleList(features).eval()
        self.avgpool = model_full.avgpool
        classifier = list(model_full.classifier)
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
    def __init__(self, model_full, target_layer):
        super(ResNet_stl2, self).__init__()
        self.model_full = model_full
        self.target_layer = target_layer

    def forward(self, x):
        if self.target_layer == '-1':
            return x

        x = self.model_full.conv1(x)
        x = self.model_full.bn1(x)
        x = self.model_full.relu(x)
        x = self.model_full.maxpool(x)

        x = self.model_full.layer1(x)
        if self.target_layer == 'layer1':
            return x
        x = self.model_full.layer2(x)
        if self.target_layer == 'layer2':
            return x
        x = self.model_full.layer3(x)
        if self.target_layer == 'layer3':
            return x
        x = self.model_full.layer4(x)
        if self.target_layer == 'layer4':
            return x

        x = self.model_full.avgpool(x)
        if self.target_layer == 'avgpool':
            return x
        x = torch.flatten(x, 1)
        x = self.model_full.fc(x)

        return x

# hidden layer --> output layer
class ResNet_stl3(torch.nn.Module):
    def __init__(self, model_full, target_layer):
        super(ResNet_stl3, self).__init__()
        self.model_full = model_full
        self.target_layer = target_layer

    def forward(self, x):
        if self.target_layer in {'-1'}:
            x = self.model_full.conv1(x)
            x = self.model_full.bn1(x)
            x = self.model_full.relu(x)
            x = self.model_full.maxpool(x)

            x = self.model_full.layer1(x)

        if self.target_layer in {'-1', 'layer1'}:
            x = self.model_full.layer2(x)
        if self.target_layer in {'-1', 'layer1', 'layer2'}:
            x = self.model_full.layer3(x)
        if self.target_layer in {'-1', 'layer1', 'layer2', 'layer3'}:
            x = self.model_full.layer4(x)

        if self.target_layer in {'-1', 'layer1', 'layer2', 'layer3', 'layer4'}:
            x = self.model_full.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model_full.fc(x)

        return x
