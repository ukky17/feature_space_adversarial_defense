# Official implementation of "Brain-inspired noise in higher layers for adversarial robustness"

## Requirements

```
PyTorch >= 1.7
```

## Run

|  Code                     |  Description
|  ----                     | ----
|  `attack_vgg16.sh`        |  Create feature-space adversarial examples on VGG16
|  `attack_resnet50.sh`     |  Create feature-space adversarial examples on ResNet50
|  `geometry.sh`            |  Run geometry analysis
|  `smoothing_predict.sh`   |  Run randomized smoothing

## Code Credit
* Classifier implementation: [torchvision package](https://github.com/pytorch/vision/tree/master/torchvision/models)
* GAN implementation: [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
* Randomized smoothing: [Official implementation](https://github.com/locuslab/smoothing)
* C&W attack: [Random self-ensemble](https://github.com/xuanqing94/RobustNet/blob/master/attack.py)
