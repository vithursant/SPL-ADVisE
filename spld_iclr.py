#sqsub -q gpu -f mpi -n 1 --mpp=4G -r 240m --gpp 1 -o /dev/null --nompirun screen -D -m

import matplotlib
matplotlib.use('Agg')
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
import torch.nn.functional as F
from tqdm import tqdm, trange
import numpy as np
import pdb

#import visdom
import csv
from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from models.wide_resnet import Wide_ResNet
from models.lenet import LeNet
from models.magnet_lenet import MagnetLeNet
from models.fashion_model import FashionSimpleNet
from models.vgg_cifar import VGG
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, MultiStepLR

from datasets.fashion import FashionMNIST
from utils.sampler import SubsetSequentialSampler
from utils.augment import *
from magnet_loss.magnet_tools import *
from magnet_loss.magnet_loss import MagnetLoss
from magnet_loss.utils import *

#vis = visdom.Visdom()
#vis.env = 'cnn_feat_dropout'

model_names = ['vgg','resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wideresnet', 'lenet', 'magnetlenet', 'magnetfashion']
magnet_names = ['vgg', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wideresnet', 'lenet', 'magnetlenet', 'magnetfashion']
dataset_names = ['cifar10', 'cifar100', 'svhn', 'mnist', 'fashionmnist', 'tinyimagenet']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--test_id', type=int, default=0, metavar='N',
                    help='test id number to be used for filenames')
parser.add_argument('--model', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names)
parser.add_argument('--shallow-model', '-s', metavar='ARCH', default='magnetlenet',
                    choices=magnet_names)
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
parser.add_argument('--spl', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--random', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='Number of classes')
parser.add_argument('--max-iter', type=int, default=20, metavar='N',
                    help='Number of clustering iterations')
parser.add_argument('--learning_rate1', type=float, default=0.1, metavar='N',
                    help='learning rate 1')
parser.add_argument('--learning_rate2', type=float, default=1e-4, metavar='N',
                    help='learning rate 2')
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
parser.add_argument('--en-scheduler', action='store_true', default=False,
                    help='Enable LR scheduler')
parser.add_argument('--magnet', action='store_true', default=False,
                    help='Enable Magnet Loss')
parser.add_argument('--spldml', action='store_true', default=False,
                    help='Enable SPLDML')
parser.add_argument('--plot', action='store_true', default=False,
                    help='Plot clustering')

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

#CUDA_VISIBLE_DEVICES = 0
args = parser.parse_args()
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

if args.spl:
    if not os.path.exists('spl_results'):
        os.makedirs('spl_results')
    if not os.path.exists('spl_results/baseline'):
        os.makedirs('spl_results/baseline')

    i = 0
    while os.path.isfile('spl_results/' + args.folder + '/' + args.dataset + '_spl_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)

    spl_logger = CSVLogger(filename='spl_results/' + args.folder + '/' + args.dataset + '_spl_log_' + test_id + '.csv', fieldnames=['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])

if args.spldml:
    if not os.path.exists('spld_dml_results'):
        os.makedirs('spld_dml_results')
    if not os.path.exists('spld_dml_results/baseline'):
        os.makedirs('spld_dml_results/baseline')
    if not os.path.exists('spld_dml_results/baseline/magnet'):
        os.makedirs('spld_dml_results/baseline/magnet')

    i = 0
    while os.path.isfile('spld_dml_results/' + args.folder + '/' + args.dataset + '_spldml_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)

    i = 0
    while os.path.isfile('spld_dml_results/' + args.folder + '/magnet/' + args.dataset + '_spldml_magnet_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)

    spldml_logger = CSVLogger(filename='spld_dml_results/' + args.folder + '/' + args.dataset + '_spldml_log_' + test_id + '.csv', fieldnames=['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])
    spldml_magnet_logger = CSVLogger(filename='spld_dml_results/' + args.folder + '/magnet/' + args.dataset + '_spldml_magnet_log_' + test_id + '.csv', fieldnames=['epoch', 'batch_loss'])

if args.magnet:
    if not os.path.exists('magnet_results'):
        os.makedirs('magnet_results')
    if not os.path.exists('magnet_results/baseline'):
        os.makedirs('magnet_results/baseline')

    i = 0
    while os.path.isfile('magnet_results/' + args.folder + '/' + args.dataset + '_magnet_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)

    magnet_logger = CSVLogger(filename='magnet_results/' + args.folder + '/' + args.dataset + '_magnet_log_' + test_id + '.csv', fieldnames=['epoch', 'batch_loss'])

if args.random:
    if not os.path.exists('random_results'):
        os.makedirs('random_results')
    if not os.path.exists('random_results/baseline'):
        os.makedirs('random_results/baseline')

    i = 0
    while os.path.isfile('random_results/' + args.folder + '/' + args.dataset + '_random_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)

    random_logger = CSVLogger(filename='random_results/' + args.folder + '/' + args.dataset + '_random_log_' + test_id + '.csv', fieldnames=['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])

print args

# Image Preprocessing
if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
elif args.dataset in ['cifar10', 'cifar100']:
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
elif args.dataset in ['mnist', 'fashionmnist']:
    normalize = transforms.Normalize((0.1307,), (0.3081,))

train_transform = transforms.Compose([])

if args.data_augmentation:
    if args.dataset in ['mnist', 'fashionmnist']:
        train_transform.transforms.append(transforms.RandomCrop(28, padding=4))
    elif args.dataset == 'tinyimagenet':
        train_transform.transforms.append(transforms.RandomCrop(32, padding=8))
    else:
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

    train_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train',
                                        transform=train_trainsform)
    test_dataset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/test',
                                                    transform=test_transform)

def encode_onehot(labels, n_classes):
    '''
        One hot encode the labels, which is used for calculating the
        loss per sample
    '''
    onehot = torch.FloatTensor(labels.size()[0], n_classes)
    labels = labels.data

    if labels.is_cuda:
        onehot = onehot.cuda()

    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)

    return onehot

def test(cnn, loader):
    loss_func = nn.CrossEntropyLoss(size_average=False).cuda()

    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    test_loss = 0.
    for images, labels in loader:
        if args.dataset == 'svhn':
            labels = labels.type_as(torch.LongTensor()).view(-1) - 1

        images = Variable(images, requires_grad=False).cuda()
        labels = Variable(labels, requires_grad=False).cuda()
        pred, _ = cnn(images)
        test_loss += loss_func(pred, labels).data[0]
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum()
    val_acc = correct / total
    val_loss = test_loss / total
    cnn.train()
    return val_acc, val_loss

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
elif args.model == 'vgg':
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

if args.spldml or args.magnet:
    if args.shallow_model == 'resnet18':
        num_classes = 2
        shallow_net = PreActResNet18(channels=num_channels, num_classes=num_classes)
    elif args.shallow_model == 'resnet34':
        num_classes = 2
        shallow_net = PreActResNet34(channels=num_channels, num_classes=num_classes)
    elif args.shallow_model == 'resnet50':
        num_classes = 2
        shallow_net = PreActResNet50(channels=num_channels, num_classes=num_classes)
    elif args.shallow_model == 'resnet101':
        num_classes = 2
        shallow_net = PreActResNet101(channels=num_channels, num_classes=num_classes)
    elif args.shallow_model == 'resnet152':
        num_classes = 2
        shallow_net = PreActResNet152(channels=num_channels, num_classes=num_classes)
    elif args.shallow_model == 'vgg':
        num_classes = 2
        shallow_net = VGG(depth=16, num_classes=num_classes, channels=num_channels)
    elif args.shallow_model == 'wideresnet':
        if args.dataset == 'svhn':
            num_classes = 2
            shallow_net = Wide_ResNet(depth=16, num_classes=num_classes, widen_factor=8, dropout_rate=args.dropout_rate)
        else:
            num_classes = 2
            shallow_net = Wide_ResNet(depth=28, num_classes=num_classes, widen_factor=10, dropout_rate=args.dropout_rate)
    elif args.shallow_model == 'magnetlenet':
        num_classes = 2
        shallow_net = MagnetLeNet(num_classes)
    elif args.shallow_model == 'magnetfashion':
        num_classes = 2
        shallow_net = FashionSimpleNet(num_classes)

if not args.spldml:
    cnn = torch.nn.DataParallel(cnn).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

# if args.dataset in ['cifar10', 'cifar100', 'svhn']:
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate1,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
# else:
#     cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learning_rate)

if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[23400, 46800, 62400], gamma=0.1)

if args.spl:
    train_sampler = SubsetSequentialSampler(range(len(train_dataset)), range(len(train_dataset)))
    train_loader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             sampler=train_sampler)

    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4)

    batch_train_inds = np.random.choice(range(len(train_dataset)), len(train_dataset), replace=False)
    train_loader.sampler.batch_indices = batch_train_inds
    #pdb.set_trace()
    k = 8
    m = 8
    d = 8

    updates = 0
    num_samples = 0

    epoch_steps = int(ceil(len(train_dataset)) / args.batch_size)
    n_steps = epoch_steps * 15

    if args.dataset in ['cifar10', 'svhn']:
        spld_params = [500, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['mnist', 'fashionmnist']:
        #spld_params = [500, 1e-3, 5e-2, 1e-1]
        spld_params = [500, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['cifar100']:
        spld_params = [10, 5e-1, 1e-1, 1e-1]

    if args.dataset == 'svhn':
        labels = train_dataset.labels
        labels = np.hstack(labels)
        #pdb.set_trace()
    else:
        labels = getattr(train_dataset, 'train_labels')
        #pdb.set_trace()
    if args.dataset == 'svhn':
        initial_reps = compute_reps(cnn, train_dataset, 4680)
    else:
        initial_reps = compute_reps(cnn, train_dataset, 400)

    batch_builder = ClusterBatchBuilder(labels, k, m, d)
    batch_builder.update_clusters(initial_reps, max_iter=args.max_iter)

    for i in range(n_steps):
        loss_vec = np.array([])
        xentropy_loss_avg = 0.
        xentropy_loss_extrap_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for batch_idx, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(updates))

            if args.dataset == 'svhn':
                labels = labels.type_as(torch.LongTensor()).view(-1) - 1

            images = Variable(images).cuda(async=True)
            labels = Variable(labels).cuda(async=True)

            cnn.zero_grad()
            pred, features = cnn(images)

            # Compute the loss for each sample in the minibatch
            onehot_target = Variable(encode_onehot(labels, args.num_classes))
            xentropy_loss_vector = -1 * torch.sum(torch.log(F.softmax(pred))
                                                                * onehot_target,
                                                            dim=1,
                                                            keepdim=True)

            # Sum minibatch loss vector
            loss_vec = np.append(loss_vec,
                                    xentropy_loss_vector.data.cpu().numpy())
            xentropy_loss_vector_sum = xentropy_loss_vector.sum()

            # Average minibatch loss vector
            xentropy_loss_vector_mean = xentropy_loss_vector.mean()
            xentropy_loss_avg += xentropy_loss_vector_mean.data[0]
            #pdb.set_trace()
            # Backward
            xentropy_loss_vector_mean.backward()
            cnn_optimizer.step()

            # Calculate running average of accuracy
            _, pred = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (pred == labels.data).sum()
            accuracy = correct / total

            start = batch_idx * images.size(0)
            stop = start + images.size(0)
            #pdb.set_trace()
            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (batch_idx + 1)),
                acc='%.3f' % accuracy)

            batch_builder.update_losses(batch_train_inds[start:stop],
                                    xentropy_loss_vector.squeeze(), 'spld')

            updates += 1

        torchvision.utils.save_image(images.data, 'spl_results/' + args.folder + '/' + args.dataset + '_spl_log_' + test_id + '.jpg', normalize=True)

        test_acc, test_loss = test(cnn, test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        row = {'epoch': str(updates), 'train_acc': str(accuracy), 'train_loss': str(xentropy_loss_avg / (updates)), 'test_acc': str(test_acc), 'test_loss': str(test_loss)}
        spl_logger.writerow(row)

        if args.en_scheduler:
            scheduler.step(updates)

        batch_train_inds = batch_builder.gen_batch_spl(spld_params[0], spld_params[1], args.batch_size)
        np.random.shuffle(batch_train_inds)
        train_loader.sampler.batch_indices = batch_train_inds.astype(np.int32)

        # Increase the learning pace
        spld_params[0] *= (1+spld_params[2])
        spld_params[0] = int(round(spld_params[0]))
        spld_params[1] *= (1+spld_params[3])

        if args.dataset == 'cifar10':
            if updates >= 200*390:
                break
        elif args.dataset in ['mnist','fashionmnist']:
            if updates >= 60*390:
                break

        # if i > 0 and i % 10 == 0:
        #   print("Refreshing clusters")
        #   reps = compute_reps(cnn, train_dataset, 400)
        #   batch_builder.update_clusters(reps)
    spl_logger.close()

if args.random:
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
    real_samples = None
    feature_samples1 = None
    feature_samples2 = None

    updates = 0

    for epoch in range(args.epochs):

        xentropy_loss_avg = 0.
        xentropy_loss_extrap_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(updates))

            if args.dataset == 'svhn':
                labels = labels.type_as(torch.LongTensor()).view(-1) - 1

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

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

            updates += 1

        torchvision.utils.save_image(images.data, 'random_results/' + args.folder + '/' + args.dataset + '_random_log_' + test_id + '.jpg', normalize=True)

        test_acc, test_loss = test(cnn, test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        row = {'epoch': str(updates), 'train_acc': str(accuracy), 'train_loss': str(xentropy_loss_avg / (updates)), 'test_acc': str(test_acc), 'test_loss': str(test_loss)}
        random_logger.writerow(row)

        if args.en_scheduler:
            scheduler.step(updates)

        # row = {'epoch': str(updates), 'train_acc': str(accuracy), 'train_loss': str(xentropy_loss_avg), 'test_acc': str(test_acc), 'test_loss': str(test_loss)}
        # random_logger.writerow(row)

        if args.dataset in ['cifar10', 'cifar100']:
            if updates >= 200*390:
                break
        elif args.dataset in ['mnist','fashionmnist']:
            if updates >= 60*390:
                break

        #print(str(cnn_optimizer.param_groups[0]['lr']))
    #torch.save(cnn.state_dict(), 'checkpoints/baseline_' + test_id + '.pt')
    random_logger.close()

if args.magnet:
    shallow_net = torch.nn.DataParallel(shallow_net).cuda()
    args.batch_size = 64
    n_train = len(train_dataset)
    train_sampler = SubsetSequentialSampler(range(len(train_dataset)), range(args.batch_size))
    train_loader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=1,
                             sampler=train_sampler)

    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=1)

    k = 8
    m = 8
    d = 8
    alpha = 1.0

    cnn_optimizer = torch.optim.Adam(shallow_net.parameters(), lr=args.learning_rate2)
    minibatch_magnet_loss = MagnetLoss()

    if args.dataset == 'svhn':
        labels = train_dataset.labels
    else:
        labels = getattr(train_dataset, 'train_labels')

    if args.dataset == 'svhn':
        initial_reps = compute_reps(shallow_net, train_dataset, 4680)
    else:
        initial_reps = compute_reps(shallow_net, train_dataset, 400)

    batch_builder = ClusterBatchBuilder(labels, k, m, d)
    batch_builder.update_clusters(initial_reps, max_iter=args.max_iter)

    batch_losses = []

    batch_example_inds, batch_class_inds = batch_builder.gen_batch()
    train_loader.sampler.batch_indices = batch_example_inds

    if args.dataset in ['mnist', 'fashionmnist']:
        n_epochs = 15
    elif args.dataset in ['cifar10', 'cifar100', 'svhn']:
        n_epochs = 30
    elif args.dataset in ['tinyimagenet']:
        n_epochs = 50

    epoch_steps = int(ceil(float(len(train_dataset) / args.batch_size)))
    #epoch_steps = len(train_loader)
    n_steps = epoch_steps * n_epochs
    cluster_refresh_interval = epoch_steps

    _ = cnn.train()
    updates = 0

    progress_bar = tqdm(range(n_steps))
    for i in progress_bar:
        batch_loss_avg = 0.

        for batch_idx, (images, targets) in enumerate(train_loader):
            #progress_bar.set_description('Epoch ' + str(updates))
            images = Variable(images).cuda()
            targets = Variable(targets).cuda()

            shallow_net.zero_grad()
            pred, _ = shallow_net(images)

            batch_loss, batch_example_losses = minibatch_magnet_loss(pred,
                                                                    batch_class_inds,
                                                                    m,
                                                                    d,
                                                                    alpha)
            batch_loss.backward()
            cnn_optimizer.step()

            batch_loss_avg += batch_loss.data[0]
            updates += 1

        progress_bar.set_postfix(
             magnetloss='%.3f' % (batch_loss_avg / (batch_idx + 1)))
        batch_builder.update_losses(batch_example_inds,
                                    batch_example_losses, 'magnet')

        batch_losses.append(batch_loss.data[0])

        # if not i % 100:
        #     print (i, batch_loss)

        if not i % cluster_refresh_interval:
            print("Refreshing clusters")
            if args.dataset == 'svhn':
                reps = compute_reps(shallow_net, train_dataset, 4680)
            else:
                reps = compute_reps(shallow_net, train_dataset, 400)
            batch_builder.update_clusters(reps)

        if args.plot:
            if not i % 2000:
                n_plot = 8000
                #pdb.set_trace()
                if args.dataset == 'svhn':
                    plot_embedding(compute_reps(shallow_net, train_dataset, 4680)[:n_plot],
                                                labels[:n_plot],
                                                name='magnet_results/' + args.folder + '/' + args.dataset + '_magnet_log_' + test_id + '_' + str(i))
                else:
                    plot_embedding(compute_reps(shallow_net, train_dataset, 400)[:n_plot],
                                                labels[:n_plot],
                                                name='magnet_results/' + args.folder + '/' + args.dataset + '_magnet_log_' + test_id + '_' + str(i))
                #plot_embedding(compute_reps(cnn, train_dataset, 400)[:n_plot], labels[:n_plot], name=args.dataset + '_' + test_id + '_' + str(i))

        batch_example_inds, batch_class_inds = batch_builder.gen_batch()
        train_loader.sampler.batch_indices = batch_example_inds

        row = {'epoch': str(updates), 'batch_loss': str(batch_loss.data[0])}
        magnet_logger.writerow(row)

    if args.plot:
        plot_smooth(batch_losses, "batch-losses")

    magnet_logger.close()

if args.spldml:
    shallow_net = torch.nn.DataParallel(shallow_net).cuda()
    args.batch_size = 64
    n_train = len(train_dataset)

    train_sampler = SubsetSequentialSampler(range(len(train_dataset)), range(args.batch_size))
    train_loader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=1,
                             sampler=train_sampler)

    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=1)

    k = 8
    m = 8
    d = 8
    alpha = 1.0

    shallow_optimizer = torch.optim.Adam(shallow_net.parameters(), lr=args.learning_rate2)
    minibatch_magnet_loss = MagnetLoss()

    if args.dataset == 'svhn':
        labels = train_dataset.labels
    else:
        labels = getattr(train_dataset, 'train_labels')

    if args.dataset == 'svhn':
        initial_reps = compute_reps(shallow_net, train_dataset, 4680)
    else:
        initial_reps = compute_reps(shallow_net, train_dataset, 400)

    batch_builder = ClusterBatchBuilder(labels, k, m, d)
    batch_builder.update_clusters(initial_reps, max_iter=args.max_iter)

    batch_losses = []

    batch_example_inds, batch_class_inds = batch_builder.gen_batch()
    train_loader.sampler.batch_indices = batch_example_inds

    if args.dataset in ['mnist', 'fashionmnist']:
        n_epochs = 15
    elif args.dataset in ['cifar10', 'cifar100', 'svhn']:
        n_epochs = 30
    elif args.dataset in ['tinyimagenet']:
        n_epochs = 50

    epoch_steps = int(ceil(float(len(train_dataset) / args.batch_size)))
    #epoch_steps = len(train_loader)
    n_steps = epoch_steps * n_epochs
    cluster_refresh_interval = epoch_steps

    _ = shallow_net.train()
    updates = 0

    #if args.dataset in ['cifar10', 'cifar100']:
    #    n_steps = 200

    progress_bar = tqdm(range(n_steps))
    for i in progress_bar:
        batch_loss_avg = 0.

        for batch_idx, (images, targets) in enumerate(train_loader):
            #progress_bar.set_description('Epoch ' + str(updates))
            images = Variable(images).cuda()
            targets = Variable(targets).cuda()

            shallow_net.zero_grad()
            pred, _ = shallow_net(images)

            batch_loss, batch_example_losses = minibatch_magnet_loss(pred,
                                                                    batch_class_inds,
                                                                    m,
                                                                    d,
                                                                    alpha)
            batch_loss.backward()
            shallow_optimizer.step()

            batch_loss_avg += batch_loss.data[0]
            updates += 1

        progress_bar.set_postfix(
             magnetloss='%.3f' % (batch_loss_avg / (batch_idx + 1)))

        batch_builder.update_losses(batch_example_inds,
                                    batch_example_losses, 'magnet')

        batch_losses.append(batch_loss.data[0])

        # if not i % 100:
        #     print (i, batch_loss)

        if not i % cluster_refresh_interval:
            print("Refreshing clusters")
            if args.dataset == 'svhn':
                reps = compute_reps(shallow_net, train_dataset, 4680)
            else:
                reps = compute_reps(shallow_net, train_dataset, 400)
            batch_builder.update_clusters(reps)

        if args.plot:
            if not i % 1000:
                n_plot = 8000
                if args.dataset == 'svhn':
                    plot_embedding(compute_reps(shallow_net, train_dataset, 4680)[:n_plot], labels[:n_plot], name='spld_dml_results/' + args.folder + '/magnet/' + args.dataset + '_spldml_magnet_log_' + test_id + '_' + str(i))
                else:
                    plot_embedding(compute_reps(shallow_net, train_dataset, 400)[:n_plot], labels[:n_plot], name='spld_dml_results/' + args.folder + '/magnet/' + args.dataset + '_spldml_magnet_log_' + test_id + '_' + str(i))

        batch_example_inds, batch_class_inds = batch_builder.gen_batch()
        train_loader.sampler.batch_indices = batch_example_inds

        row = {'epoch': str(updates), 'batch_loss': str(batch_loss.data[0])}
        spldml_magnet_logger.writerow(row)

    if args.plot:
        plot_smooth(batch_losses, args.dataset + '_' + test_id + '_batch-losses')

    spldml_magnet_logger.close()

    cnn = torch.nn.DataParallel(cnn).cuda()

    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        args.batch_size = 128
    else:
        args.batch_size = 64

    train_sampler = SubsetSequentialSampler(range(len(train_dataset)), range(len(train_dataset)))
    train_loader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             sampler=train_sampler)

    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4)

    batch_train_inds = np.random.choice(range(len(train_dataset)), len(train_dataset), replace=False)
    train_loader.sampler.batch_indices = batch_train_inds
    #pdb.set_trace()

    updates = 0
    num_samples = 0

    epoch_steps = int(ceil(len(train_dataset)) / args.batch_size)
    n_steps = epoch_steps * 15

    if args.dataset in ['cifar10', 'svhn']:
        spld_params = [500, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['mnist', 'fashionmnist']:
        spld_params = [500, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['cifar100']:
        spld_params = [10, 5e-1, 1e-1, 1e-1]
        #spld_params = [500, 1e-3, 5e-2, 1e-1]
        #spld_params = [100, 1e-3, 5e-2, 1e-1]

    if args.dataset == 'svhn':
        labels = train_dataset.labels
    else:
        labels = getattr(train_dataset, 'train_labels')

    for i in range(n_steps):
        loss_vec = np.array([])
        xentropy_loss_avg = 0.
        xentropy_loss_extrap_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for batch_idx, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(updates))

            if args.dataset == 'svhn':
                labels = labels.type_as(torch.LongTensor()).view(-1) - 1

            images = Variable(images).cuda(async=True)
            labels = Variable(labels).cuda(async=True)

            cnn.zero_grad()
            pred, features = cnn(images)

            # Compute the loss for each sample in the minibatch
            onehot_target = Variable(encode_onehot(labels, args.num_classes))
            xentropy_loss_vector = -1 * torch.sum(torch.log(F.softmax(pred))
                                                                * onehot_target,
                                                            dim=1,
                                                            keepdim=True)

            # Sum minibatch loss vector
            loss_vec = np.append(loss_vec,
                                    xentropy_loss_vector.data.cpu().numpy())
            xentropy_loss_vector_sum = xentropy_loss_vector.sum()

            # Average minibatch loss vector
            xentropy_loss_vector_mean = xentropy_loss_vector.mean()
            xentropy_loss_avg += xentropy_loss_vector_mean.data[0]
            #pdb.set_trace()

            # Backward
            xentropy_loss_vector_mean.backward()
            cnn_optimizer.step()

            # Calculate running average of accuracy
            _, pred = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (pred == labels.data).sum()
            accuracy = correct / total

            start = batch_idx * images.size(0)
            stop = start + images.size(0)
            #pdb.set_trace()
            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (batch_idx + 1)),
                acc='%.3f' % accuracy)

            updates += 1

            batch_builder.update_losses(batch_train_inds[start:stop],
                                    xentropy_loss_vector.squeeze(), 'spld')

        torchvision.utils.save_image(images.data, 'spld_dml_results/' + args.folder + '/' + args.dataset + '_spldml_magnet_log_' + test_id + '.jpg', normalize=True)

        test_acc, test_loss = test(cnn, test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        row = {'epoch': str(updates), 'train_acc': str(accuracy), 'train_loss': str(xentropy_loss_avg / (updates)), 'test_acc': str(test_acc), 'test_loss': str(test_loss)}
        spldml_logger.writerow(row)

        if args.en_scheduler:
            scheduler.step(updates)

        # row = {'epoch': str(updates), 'train_acc': str(accuracy), 'train_loss': str(xentropy_loss_avg), 'test_acc': str(test_acc), 'test_loss': str(test_loss)}
        # spldml_logger.writerow(row)

        batch_train_inds = batch_builder.gen_batch_spl(spld_params[0], spld_params[1], args.batch_size)
        np.random.shuffle(batch_train_inds)
        train_loader.sampler.batch_indices = batch_train_inds

        # Increase the learning pace
        spld_params[0] *= (1+spld_params[2])
        spld_params[0] = int(round(spld_params[0]))
        spld_params[1] *= (1+spld_params[3])

        if args.dataset in ['cifar10', 'cifar100']:
            if updates >= 200*390:
                break
        elif args.dataset in ['mnist','fashionmnist']:
            if updates >= 60*390:
                break
    spldml_logger.close()
