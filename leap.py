import matplotlib
matplotlib.use('Agg')
import os
import os.path
import argparse
from tqdm import tqdm, trange
import numpy as np
import pdb
import csv

import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, MultiStepLR

from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from models.wide_resnet import Wide_ResNet
from models.lenet import LeNet
from models.magnet_lenet import MagnetLeNet
from models.fashion_model import FashionSimpleNet
from models.vgg_cifar import VGG

from datasets.fashion import FashionMNIST
from utils.sampler import SubsetSequentialSampler
from utils.augment import *
from magnet_loss.magnet_tools import *
from magnet_loss.magnet_loss import MagnetLoss
from magnet_loss.utils import *
