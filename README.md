# Official implementation of "Adversarial attacks and defenses using feature-space stochasticity"

## What's included in this repository?
* Attack to create feature-space adversarial examples (Section 4.1)
* Geometrical analysis (Section 4.2)
* Defense experiments (Section 4.3)

All experiments are performed on VGG16 and ResNet50 classifier using STL-10 dataset.

## Requirements

```
PyTorch >= 1.7
```

## Run

|  Code                     |  Description
|  ----                     | ----
|  `attack_vgg16.sh`        |  Create feature-space adversarial examples on VGG16 classifier
|  `attack_resnet50.sh`     |  Create feature-space adversarial examples on ResNet50 classifier
|  `geometry.sh`            |  Run geometry analysis
|  `smoothing_predict.sh`   |  Run defense experiments

## Code References
* Classifier implementation: [torchvision package](https://github.com/pytorch/vision/tree/master/torchvision/models)
* GAN implementation: [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
* Randomized smoothing: [Official implementation](https://github.com/locuslab/smoothing)
* C&W attack: [Random self-ensemble](https://github.com/xuanqing94/RobustNet/blob/master/attack.py)

## Citation

**Adversarial attacks and defenses using feature-space stochasticity**. Jumpei Ukita and Kenichi Ohki, Neural Networks, 2023. [Paper link](https://www.sciencedirect.com/science/article/abs/pii/S0893608023004422)
