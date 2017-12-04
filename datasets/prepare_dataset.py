import torchvision
from torchvision import datasets
from datasets.fashion import FashionMNIST
import pdb
import numpy as np

def prepare_dataset(args, train_transform, test_transform):

    if args.dataset == 'mnist':
        num_classes = 10
        num_channels = 1
        train_dataset = datasets.MNIST(root='./data',
                                train=True,
                                transform=train_transform,
                                download=True)

        test_dataset = datasets.MNIST(root='./data',
                                train=False,
                                transform=test_transform,
                                download=True)

    elif args.dataset == 'fashionmnist':
        num_classes = 10
        num_channels = 1
        train_dataset = FashionMNIST(root='./data/fashionmnist',
                                train=True,
                                transform=train_transform,
                                download=True)

        test_dataset = FashionMNIST(root='./data/fashionmnist',
                                train=False,
                                transform=test_transform,
                                download=True)
    elif args.dataset == 'cifar10':
        num_classes = 10
        num_channels = 3
        train_dataset = datasets.CIFAR10(root='./data',
                                         train=True,
                                         transform=train_transform,
                                         download=True)

        test_dataset = datasets.CIFAR10(root='./data',
                                        train=False,
                                        transform=test_transform,
                                        download=True)

    elif args.dataset == 'cifar100':
        num_classes = 100
        num_channels = 3
        train_dataset = datasets.CIFAR100(root='./data',
                                          train=True,
                                          transform=train_transform,
                                          download=True)

        test_dataset = datasets.CIFAR100(root='./data',
                                         train=False,
                                         transform=test_transform,
                                         download=True)

    elif args.dataset == 'svhn':
        num_classes = 10
        num_channels = 3
        train_dataset = datasets.SVHN(root='./data',
                                      split='train',
                                      transform=train_transform,
                                      download=True)

        extra_dataset = datasets.SVHN(root='./data',
                                      split='extra',
                                      transform=train_transform,
                                      download=True)

        # Combine both training splits, as is common practice for SVHN
        data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
        train_dataset.data = data
        train_dataset.labels = labels

        test_dataset = datasets.SVHN(root='./data',
                                     split='test',
                                     transform=test_transform,
                                     download=True)

    elif args.dataset == 'tinyimagenet':
        num_classes = 200
        num_channels = 3

        train_dataset = datasets.ImageFolder(root='/scratch/vthangar/tiny-imagenet-200/train',
                                            transform=train_transform)
        test_dataset = datasets.ImageFolder(root='/scratch/vthangar/tiny-imagenet-200/val',
                                                        transform=test_transform)

    return num_classes, num_channels, train_dataset, test_dataset
