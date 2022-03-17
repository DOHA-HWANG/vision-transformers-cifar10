#!/bin/bash

# python train_vit_cifar10.py --patch 2 --lr 1e-4 --aug --n_epochs 200 # vit-patchsize-2

# python train_vit_cifar10.py --net vit --lr 1e-4 # train with pretrained vit

# python train_vit_cifar10.py --net vit_timm_base --lr 1e-4 # train with pretrained vit

# python train_vit_cifar10.py --net vit_timm_small --lr 1e-4 # train with pretrained vit

python train_vit_cifar10.py --net deit --lr 1e-4 
# python train_vit_cifar10.py --net deit --lr 1e-4 --teacher-model regnety_160 --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth # train with pretrained vit

# python train_vit_cifar10.py --net vit_timm_large --lr 1e-4 # train with pretrained vit

# python train_vit_cifar10.py --net convmixer --aug --n_epochs 200 # train with convmixer

# python train_vit_cifar10.py --net res18 # resnet18

#python train_vit_cifar10.py --net res50 --aug --n_epochs 200 # resnet18+randaug
