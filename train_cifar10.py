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

from timm.models import create_model

from models import *
from models.vit import ViT
from utils import progress_bar
from models.convmixer import ConvMixer

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup
from dataset import my_Cifar10
from distillation_loss import DistillationLoss
# from models.CIFAR10.custom_models_cifar10 import resnet50

import pdb

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')  # store_true : False
parser.add_argument('--amp', action='store_true', help='enable AMP training')
# parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', type=str, default='vit')
parser.add_argument('--bs', type=int, default='256')
parser.add_argument('--size', type=int, default="32")
parser.add_argument('--classes', type=int, default="10")
parser.add_argument('--hidden_dim', type=int, default="512")
parser.add_argument('--encoder_blocks', type=int, default="6")
parser.add_argument('--mha_head_cnt', type=int, default="8")
parser.add_argument('--n_epochs', type=int, default='50')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--convkernel', default='8', type=int)
parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')

# Distillation parameters
parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                    help='Name of teacher model to train (default: "regnety_160"')
parser.add_argument('--teacher-path', type=str, default='')
parser.add_argument('--distillation-type', default='hard', choices=['none', 'soft', 'hard'], type=str, help="")
parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# smooding
parser.add_argument('--smoothing', action='store_true', help='use smoothing') 

# check quantization, not implemented for ViT
parser.add_argument('--is_quant', type=int, default=0, help='0: no quant or 1: quant') 

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
print('aug: ', args.aug)
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
    image_size = args.size,
    patch_size = args.patch,
    num_classes = args.classes,
    dim = args.hidden_dim,
    depth = args.encoder_blocks,
    heads = args.mha_head_cnt,
    mlp_dim = args.hidden_dim,
    dropout = 0.1,
    emb_dropout = 0.1,
    distilled = False,
    # teacher_model=None,
)
elif args.net=="deit":
    # DeiT for cifar10
    # load teacher model
    teacher_model = ResNet50()
    teacher_checkpoint = torch.load("checkpoint/res50-4-ckpt.t7")
    teacher_model.load_state_dict(teacher_checkpoint['model'])
    teacher_model.to(device)
    teacher_model.eval()
    
    # import timm
    # teacher_model = None
    # if args.distillation_type != 'none':
    #     assert args.teacher_path, 'need to specify teacher-path when using distillation'
    #     print(f"Creating teacher model: {args.teacher_model}")
    #     teacher_model = create_model(
    #         args.teacher_model,
    #         pretrained=False,
    #         num_classes=args.classes,
    #         global_pool='avg',
    #     )
    #     if args.teacher_path.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.teacher_path, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.teacher_path, map_location='cpu')
    #     teacher_model.load_state_dict(checkpoint['model'])
    #     teacher_model.to(device)
    #     teacher_model.eval()

    net = ViT(
    image_size = args.size,
    patch_size = args.patch,
    num_classes = args.classes,
    dim = args.hidden_dim,
    depth = args.encoder_blocks,
    heads = args.mha_head_cnt,
    mlp_dim = args.hidden_dim,
    dropout = 0.1,
    emb_dropout = 0.1,
    distilled = True,
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
    

# # fix the seed for reproducibility
# seed = args.seed + utils.get_rank()
# torch.manual_seed(seed)
# np.random.seed(seed)
# # random.seed(seed)


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
net = net.to(device)
# if device == 'cuda':
#     net = nn.DataParallel(net) # make parallel
#     cudnn.benchmark = True
print('resume: ', args.resume)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

mixup_fn = None
mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
if mixup_active:
    mixup_fn = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.classes)

# Loss is CE
# if args.net!="deit":
#     criterion = nn.CrossEntropyLoss()
# else:
#     if mixup_active:
#         # smoothing is handled with mixup label transform
#         criterion = SoftTargetCrossEntropy()
#     elif args.smoothing:
#         criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
#     else:
#         criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
if args.net=="deit":
    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )


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
        # pdb.set_trace()
        if args.net=="deit" and mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            if args.net=="deit" or args.net=="vit":
                outputs = net(inputs, training = True)  # outputs: cls, dist in deit model
            else:
                outputs = net(inputs)

            if args.net=="deit":
                loss = criterion(inputs, outputs, targets)
            else:
                loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        if args.net=="deit":
            _, predicted = outputs[0].max(1)
            total += targets.size(0)
            _, max_target = targets.max(1)
            correct += predicted.eq(max_target).sum().item()
        else:
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
    criterion = nn.CrossEntropyLoss()
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
        print(f'save: {args.net}, acc: {acc}')
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