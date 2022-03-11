# Reference Codes
# https://github.com/kentaroy47/vision-transformers-cifar10
# https://github.com/FrancescoSaverioZuppichini/ViT
# https://github.com/lucidrains/vit-pytorch


#Lib import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from torchvision.utils import save_image

from models import *
from models.vit import ViT
from utils import progress_bar
from models.convmixer import ConvMixer

from dataset import my_Cifar10

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='256')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='50')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--convkernel', default='8', type=int)
parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')
# parser.add_argument('--dataset', default="cifar10")

args = parser.parse_args()

# Use wandb for visualize & debug
# User guide(Korean): https://greeksharifa.github.io/references/2020/06/10/wandb-usage/
# take in args
import wandb
watermark = "{}_lr{}".format(args.net, args.lr)
if args.amp:
    watermark += "_useamp"

wandb.init(project="cifar10-challange",
           name=watermark)
wandb.config.update(args)

# Use albumentations for image augmentations
# User guide(Korean): https://hoya012.github.io/blog/albumentation_tutorial/
if args.aug:
    import albumentations
bs = int(args.bs)
imsize = int(args.size)

use_amp = args.amp

if args.net=="vit_timm_large":
    size = 384
elif args.net=="vit_timm_small" or args.net=="vit_timm_base":
    size = 224
else:
    size = imsize

# Load dataset
train_dataset, test_dataset, train_dataloader, test_dataloader = my_Cifar10(imageSize=size, aug=args.aug)

print('train_dataset', len(train_dataset))
print('test_dataset', len(test_dataset))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',  'frog', 'horse', 'ship', 'truck')

# Check sample image
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

print(images.shape)
img1 = images[0]
print('label', classes[labels[0]])
save_image(img1, "./visualize/cifar10_sample1_{}.png".format(classes[labels[0]]))


# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = 32,
    patch_size = args.patch,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm_large" or args.net=="vit_timm_base" or args.net=="vit_timm_small":
    import timm
    print("Available Vision Transformer Models: ")
    print(timm.list_models("vit*"))
    if args.net=="vit_timm_base":
        net = timm.create_model("vit_base_patch16_224", pretrained=True)
    elif args.net=="vit_timm_small":
        net = timm.create_model("vit_small_patch16_224", pretrained=True)
    elif args.net=="vit_timm_large":
        net = timm.create_model("vit_large_patch16_384", pretrained=True)

    net.head = nn.Linear(net.head.in_features, 10)
    

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net) # make parallel
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

 
# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  


# use cosine or reduce LR on Plateau scheduling
if not args.cos:
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

if args.cos:
    wandb.config.scheduler = "cosine"
else:
    wandb.config.scheduler = "ReduceLROnPlateau" 
    

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

wandb.watch(net)
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    if args.cos:
        scheduler.step(epoch-1)
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

# writeout wandb
wandb.save("wandb_{}.h5".format(args.net))
