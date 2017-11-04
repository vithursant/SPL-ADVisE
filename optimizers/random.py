import os
import numpy as np
import pdb
from tqdm import tqdm, trange

import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torch.nn as nn

from utils.csv_logger import *

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

def test(args, model, loader):
    loss_func = nn.CrossEntropyLoss(size_average=False).cuda()

    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    test_loss = 0.
    for images, labels in loader:
        if args.dataset == 'svhn':
            labels = labels.type_as(torch.LongTensor()).view(-1) - 1

        images = Variable(images, requires_grad=False).cuda()
        labels = Variable(labels, requires_grad=False).cuda()
        pred, _ = model(images)
        test_loss += loss_func(pred, labels).data[0]
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum()
    val_acc = correct / total
    val_loss = test_loss / total
    model.train()

    return val_acc, val_loss

def random_sampling(args, train_dataset, test_dataset, optimizer, model, scheduler, criterion):
    if not os.path.exists('random_results'):
        os.makedirs('random_results')
    if not os.path.exists('random_results/baseline'):
        os.makedirs('random_results/baseline')

    i = 0
    while os.path.isfile('random_results/' + args.folder + '/' + args.dataset + '_random_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)

    random_logger = CSVLogger(args=args, filename='random_results/' + args.folder + '/' + args.dataset + '_random_log_' + test_id + '.csv', fieldnames=['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])

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

            images = Variable(images).cuda(async=True)
            labels = Variable(labels).cuda(async=True)

            model.zero_grad()
            pred, _ = model(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            optimizer.step()
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

        test_acc, test_loss = test(args, model, test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        row = {'epoch': str(updates), 'train_acc': str(accuracy), 'train_loss': str(xentropy_loss_avg / (updates)), 'test_acc': str(test_acc), 'test_loss': str(test_loss)}
        random_logger.writerow(row)

        if args.en_scheduler:
            scheduler.step(updates)

        if args.dataset == 'cifar10':
            if updates >= 200*390:
                break
        elif args.dataset in ['mnist','fashionmnist']:
            if updates >= 60*390:
                break

        #print(str(cnn_optimizer.param_groups[0]['lr']))
    #torch.save(cnn.state_dict(), 'checkpoints/baseline_' + test_id + '.pt')
    random_logger.close()
