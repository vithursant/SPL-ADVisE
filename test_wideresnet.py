#sqsub -q gpu -f mpi -n 1 --mpp=4G -r 240m --gpp 1 -o /dev/null --nompirun screen -D -m

import os
import os.path
import argparse
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
import numpy as np
import pdb

#import visdom
import csv
from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from models.wide_resnet import Wide_ResNet

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, MultiStepLR
#from utils.validation_set_split import validation_split
#from utils.transforms import HolePunch, BernoulliNoise

#vis = visdom.Visdom()
#vis.env = 'cnn_feat_dropout'

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wideresnet']
dataset_names = ['cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--test_id', type=int, default=0, metavar='N',
                    help='test id number to be used for filenames')
parser.add_argument('--model', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names)
parser.add_argument('--dataset', '-d', metavar='D', default='cifar10',
                    choices=dataset_names)
parser.add_argument('--folder', '-f', default='baseline',
                    choices=['baseline', 'final_tests'])
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--learning_rate', type=float, default=0.1, metavar='N',
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--holepunch', action='store_true', default=False,
                    help='adds random blobs to the original image')
parser.add_argument('--n_holes', type=int, default=1, metavar='S',
                    help='number of holes in image, if using swiss cheese or holepunch')
parser.add_argument('--radius', type=int, default=8, metavar='S',
                    help='radius ofthe holes')
parser.add_argument('--dropout_rate', type=float, default=0.3, metavar='N',
                    help='dropout rate')
parser.add_argument('--square', action='store_true', default=False,
                    help='use square instead of circle for hole')
parser.add_argument('--bernoulli', action='store_true', default=False,
                    help='add bernoulli noise')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

if args.dataset =='svhn':
    # parameters from https://arxiv.org/pdf/1605.07146.pdf
    args.learning_rate = 0.01
    args.data_augmentation = False
    args.epochs = 160
    args.dropout_rate = 0.4

if args.model != 'wideresnet':
    args.dropout_rate = 0.0

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/baseline'):
    os.makedirs('results/baseline')

i = args.test_id
while os.path.isfile('results/' + args.folder + '/log_baseline' + str(i) + '.csv'):
    i += 1
args.test_id = i
test_id = str(args.test_id)

print args

# Image Preprocessing
if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
else:
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.holepunch:
    train_transform.transforms.append(HolePunch(n_holes=args.n_holes, radius=args.radius, square=args.square))
if args.bernoulli:
    train_transform.transforms.append(BernoulliNoise(p=0.5))


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if args.dataset == 'cifar10':
    num_classes = 10
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

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)


def test(cnn, loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        if args.dataset == 'svhn':
            labels = labels.type_as(torch.LongTensor()).view(-1) - 1

        images = Variable(images, requires_grad=False).cuda()
        labels = Variable(labels, requires_grad=False).cuda()
        pred, _ = cnn(images)
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum()
    val_acc = correct / total
    cnn.train()
    return val_acc


if args.model == 'resnet18':
    cnn = PreActResNet18(num_classes=num_classes)
elif args.model == 'resnet34':
    cnn = PreActResNet34(num_classes=num_classes)
elif args.model == 'resnet50':
    cnn = PreActResNet50(num_classes=num_classes)
elif args.model == 'resnet101':
    cnn = PreActResNet101(num_classes=num_classes)
elif args.model == 'resnet152':
    cnn = PreActResNet152(num_classes=num_classes)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = Wide_ResNet(depth=16, num_classes=num_classes, widen_factor=8, dropout_rate=args.dropout_rate)
    else:
        cnn = Wide_ResNet(depth=28, num_classes=num_classes, widen_factor=10, dropout_rate=args.dropout_rate)

cnn = torch.nn.DataParallel(cnn).cuda()
criterion = nn.CrossEntropyLoss().cuda()

cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)


class CSVLogger():
    def __init__(self, filename='log.csv', fieldnames=['epoch']):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


csv_logger = CSVLogger(filename='results/' + args.folder + '/log_baseline' + test_id + '.csv', fieldnames=['epoch', 'train_acc', 'test_acc'])

real_samples = None
feature_samples1 = None
feature_samples2 = None

for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    xentropy_loss_extrap_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        if args.dataset == 'svhn':
            labels = labels.type_as(torch.LongTensor()).view(-1) - 1

        images = Variable(images).cuda(async=True)
        labels = Variable(labels).cuda(async=True)

        cnn.zero_grad()
        pred, _ = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.data[0]

        # Calculate running average of accuracy
        _, pred = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (pred == labels.data).sum()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    torchvision.utils.save_image(images.data, 'results/baseline_' + test_id + '.jpg', normalize=True)

    test_acc = test(cnn, test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step(epoch)

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

#torch.save(cnn.state_dict(), 'checkpoints/baseline_' + test_id + '.pt')
csv_logger.close()