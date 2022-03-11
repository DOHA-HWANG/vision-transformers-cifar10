import torch
from torchvision import datasets, transforms
from randomaug import RandAugment

def my_Cifar10(imageSize=224, aug=False):
    transform = transforms.Compose([transforms.Resize(imageSize),
                                    transforms.RandomCrop(imageSize, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)
                                    ])

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(imageSize), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        # [transforms.ToTensor(), transforms.Resize(224), transforms.Normalize(mean, std)])

    # Add RandAugment with N, M(hyperparameter)
    if aug:  
        N = 2; M = 14;
        transform.transforms.insert(0, RandAugment(N, M))

    train_dataset = datasets.CIFAR10(
        root='./.data',
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.CIFAR10(
        root='./.data',
        train=False,
        transform=transform_test,
        download=True
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 16, True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 16, True)

    return train_dataset, test_dataset, train_dataloader, test_dataloader