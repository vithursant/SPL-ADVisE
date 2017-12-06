import matplotlib
matplotlib.use('Agg')
import os
import os.path
# import argparse
from tqdm import tqdm, trange
import numpy as np
import pdb
import csv

import torch
import torchvision
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, MultiStepLR

from optimizers.spld import *
from optimizers.random import *
from optimizers.leap import *

from dml.magnet import *

from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from models.wide_resnet import Wide_ResNet
from models.lenet import LeNet
from models.magnet_lenet import MagnetLeNet
from models.fashion_model import FashionSimpleNet
from models.vgg_cifar import VGG
from models.inception import Inception3
from models.resnext import resnext
from models.densenet import densenet
#from models.vgg import *

from datasets.transform import *
from datasets.prepare_dataset import *

from utils.sampler import SubsetSequentialSampler
from utils.augment import *
from utils.parse_hyperparam import *
from utils.csv_logger import *

from magnet_loss.magnet_tools import *
from magnet_loss.magnet_loss import MagnetLoss
from magnet_loss.utils import *

args = parse_settings()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

if args.dataset =='svhn':
    # parameters from https://arxiv.org/pdf/1605.07146.pdf
    args.learning_rate1 = 0.01
    args.data_augmentation = False
    args.epochs = 160
    args.dropout_rate = 0.4

if args.model != 'wideresnet':
    args.dropout_rate = 0.0

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print args

train_transform, test_transform = preform_transform(args)
num_classes, num_channels, train_dataset, test_dataset = prepare_dataset(args, train_transform, test_transform)

if args.model == 'resnet18':
    cnn = PreActResNet18(channels=num_channels, num_classes=num_classes)
elif args.model == 'resnet34':
    cnn = PreActResNet34(channels=num_channels, num_classes=num_classes)
elif args.model == 'resnet50':
    cnn = PreActResNet50(channels=num_channels, num_classes=num_classes)
elif args.model == 'resnet101':
    cnn = PreActResNet101(channels=num_channels, num_classes=num_classes)
elif args.model == 'resnet152':
    cnn = PreActResNet152(channels=num_channels, num_classes=num_classes)
elif args.model == 'inceptionv3':
    cnn = Inception3(num_classes=num_classes)
elif args.model == 'vgg16':
    cnn = VGG(depth=16, num_classes=num_classes, channels=num_channels)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = Wide_ResNet(depth=16, num_classes=num_classes, widen_factor=8, dropout_rate=args.dropout_rate)
    else:
        cnn = Wide_ResNet(depth=28, num_classes=num_classes, widen_factor=10, dropout_rate=args.dropout_rate)
elif args.model == 'lenet':
    cnn = LeNet()
elif args.model == 'magnetlenet':
    cnn = MagnetLeNet(num_classes)
elif args.model == 'magnetfashion':
    cnn = FashionSimpleNet(num_classes)
elif args.model == 'resnext':
    cnn = resnext(cardinality=args.cardinality, num_classes=num_classes, depth=args.depth, widen_factor=args.widen_factor, dropRate=args.dropout_rate)
elif args.model == 'densenet':
    cnn = densenet(num_classes=num_classes, depth=args.depth, growthRate=args.growthRate, compressionRate=args.compressionRate, dropRate=args.dropout_rate)

if args.leap or args.magnet:
    if args.embedding_model == 'resnet18':
        num_classes = 2
        embedding_cnn = PreActResNet18(channels=num_channels, num_classes=num_classes)
    elif args.embedding_model == 'resnet34':
        num_classes = 2
        embedding_cnn = PreActResNet34(channels=num_channels, num_classes=num_classes)
    elif args.embedding_model == 'resnet50':
        num_classes = 2
        embedding_cnn = PreActResNet50(channels=num_channels, num_classes=num_classes)
    elif args.embedding_model == 'resnet101':
        num_classes = 2
        embedding_cnn = PreActResNet101(channels=num_channels, num_classes=num_classes)
    elif args.embedding_model == 'resnet152':
        num_classes = 2
        embedding_cnn = PreActResNet152(channels=num_channels, num_classes=num_classes)
    elif args.embedding_model == 'vgg16':
        num_classes = 2
        embedding_cnn = VGG(depth=16, num_classes=num_classes, channels=num_channels)
    elif args.embedding_model == 'vgg11':
        num_classes = 2
        embedding_cnn = vgg11(num_classes)
    elif args.embedding_model == 'wideresnet':
        if args.dataset == 'svhn':
            num_classes = 2
            embedding_cnn = Wide_ResNet(depth=16, num_classes=num_classes, widen_factor=8, dropout_rate=args.dropout_rate)
        else:
            num_classes = 2
            embedding_cnn = Wide_ResNet(depth=28, num_classes=num_classes, widen_factor=10, dropout_rate=args.dropout_rate)
    elif args.embedding_model == 'magnetlenet':
        num_classes = 2
        embedding_cnn = MagnetLeNet(num_classes)
    elif args.embedding_model == 'magnetfashion':
        num_classes = 2
        embedding_cnn = FashionSimpleNet(num_classes)

if not args.leap:
    cnn = torch.nn.DataParallel(cnn).cuda()
    criterion = nn.CrossEntropyLoss()

cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate1,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[23400, 46800, 62400], gamma=0.1)

def main():
    # TODO: Test on SVHN, Skin Lesion Classification, Tiny Imagenet
    # TODO: Try LEAP for CapsNet
    if args.spld:
        self_paced_learning_with_diversity(args, train_dataset, test_dataset, cnn_optimizer, cnn, scheduler)
    elif args.random:
        random_sampling(args, train_dataset, test_dataset, cnn_optimizer, cnn, scheduler, criterion)
    elif args.leap:
        leap(args, train_dataset, test_dataset, cnn_optimizer, embedding_cnn, cnn, scheduler)
    elif args.magnet:
        magnet(args, train_dataset, test_dataset, embedding_cnn, scheduler)

if __name__ == '__main__':
    main()
