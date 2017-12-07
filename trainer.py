import matplotlib
matplotlib.use('Agg')
import os
import os.path
# import argparse
from tqdm import tqdm, trange
import numpy as np
import pdb
import csv
import random
import shutil

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
#from optimizers.random import *
from optimizers.leap import *
from optimizers.random_selector import random_selector
from optimizers.spld_selector import spld_selector
from optimizers.leap_selector import leap_selector

from dml.magnet import *

from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from models.wide_resnet import Wide_ResNet
from models.lenet import LeNet
from models.magnet_lenet import MagnetLeNet
from models.fashion_model import FashionSimpleNet
from models.vgg_cifar import VGG
from models.inception import inception_v3
from models.resnext import resnext
from models.densenet import densenet
from models.preresnet import preresnet
#from models.vgg import *

from datasets.transform import *
from datasets.prepare_dataset import *

from utils.sampler import SubsetSequentialSampler
from utils.augment import *
from utils.parse_hyperparam import *
from utils.csv_logger import *
from utils.misc import *
from utils.txt_logger import *
from utils.visualize import *

# progress bar
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar

from magnet_loss.magnet_tools import *
from magnet_loss.magnet_loss import MagnetLoss
from magnet_loss.utils import *

args = parse_settings()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

if args.dataset =='svhn':
    # parameters from https://arxiv.org/pdf/1605.07146.pdf
    args.learning_rate1 = 0.01
    args.data_augmentation = False
    args.epochs = 160
    args.dropout_rate = 0.4

if args.model != 'wideresnet':
    args.dropout_rate = 0.0

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    train_transform, test_transform = preform_transform(args)
    num_classes, num_channels, train_dataset, test_dataset = prepare_dataset(args,
                                                                             train_transform,
                                                                             test_transform)

    # Model
    print("==> Creating model '{}'".format(args.model))

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
    elif args.model == 'preresnet':
        cnn = preresnet(depth=args.depth, num_classes=num_classes)
    elif args.model == 'inceptionv3':
        #cnn = Inception3(num_classes=num_classes)
        cnn = inception_v3(pretrained=True, num_classes=num_classes)
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

    cnn = torch.nn.DataParallel(cnn).cuda()
    cudnn.benchmark = True

    print('    Total params: %.2fM' % (sum(p.numel() for p in cnn.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, cnn.parameters()),
                          lr=args.learning_rate1,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # Resume
    title = args.dataset + '-' + args.model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        cnn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['No. Samples', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(test_loader, cnn, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    if args.random:
        random_selector(args, state, start_epoch, train_dataset, test_dataset, cnn, criterion, optimizer, use_cuda, logger)
    elif args.spld:
        spld_selector(args, state, train_dataset, test_dataset, cnn, criterion, optimizer, use_cuda, logger)
    elif args.leap:
        leap_selector(  args,
                        state,
                        train_dataset,
                        test_dataset,
                        cnn,
                        embedding_cnn,
                        criterion,
                        optimizer,
                        use_cuda,
                        logger)

    # for epoch in range(start_epoch, args.epochs):
    #     adjust_learning_rate(optimizer, epoch)
    #     print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['learning_rate1']))
    #
    #     train_loss, train_acc = train(train_loader, cnn, criterion, optimizer, epoch, use_cuda)
    #     test_loss, test_acc = test(test_loader, cnn, criterion, epoch, use_cuda)
    #
    #     # append logger file
    #     logger.append([state['learning_rate1'], train_loss, test_loss, train_acc, test_acc])
    #
    #     # save model
    #     is_best = test_acc > best_acc
    #     best_acc = max(test_acc, best_acc)
    #     save_checkpoint({
    #             'epoch': epoch + 1,
    #             'state_dict': cnn.state_dict(),
    #             'acc': test_acc,
    #             'best_acc': best_acc,
    #             'optimizer' : optimizer.state_dict(),
    #         }, is_best, checkpoint=args.checkpoint)
    #
    # logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))
    #
    # print('Best acc:')
    # print(best_acc)

# def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
#     # switch to train mode
#     model.train()
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     end = time.time()
#
#     bar = Bar('Processing', max=len(trainloader))
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda(async=True)
#         inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
#
#         # compute output
#         outputs, _ = model(inputs)
#         loss = criterion(outputs, targets)
#
#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
#         losses.update(loss.data[0], inputs.size(0))
#         top1.update(prec1[0], inputs.size(0))
#         top5.update(prec5[0], inputs.size(0))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         # plot progress
#         bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
#                     batch=batch_idx + 1,
#                     size=len(trainloader),
#                     data=data_time.avg,
#                     bt=batch_time.avg,
#                     total=bar.elapsed_td,
#                     eta=bar.eta_td,
#                     loss=losses.avg,
#                     top1=top1.avg,
#                     top5=top5.avg,
#                     )
#         bar.next()
#     bar.finish()
#     return (losses.avg, top1.avg)
#
# def test(testloader, model, criterion, epoch, use_cuda):
#     global best_acc
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     end = time.time()
#     bar = Bar('Processing', max=len(testloader))
#     for batch_idx, (inputs, targets) in enumerate(testloader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()
#         inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
#
#         # compute output
#         outputs, _ = model(inputs)
#         loss = criterion(outputs, targets)
#
#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
#         losses.update(loss.data[0], inputs.size(0))
#         top1.update(prec1[0], inputs.size(0))
#         top5.update(prec5[0], inputs.size(0))
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         # plot progress
#         bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
#                     batch=batch_idx + 1,
#                     size=len(testloader),
#                     data=data_time.avg,
#                     bt=batch_time.avg,
#                     total=bar.elapsed_td,
#                     eta=bar.eta_td,
#                     loss=losses.avg,
#                     top1=top1.avg,
#                     top5=top5.avg,
#                     )
#         bar.next()
#     bar.finish()
#     return (losses.avg, top1.avg)

if __name__ == '__main__':
    main()
