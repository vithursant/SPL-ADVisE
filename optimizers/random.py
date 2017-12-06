import os
import numpy as np
import pdb
from tqdm import tqdm, trange
from progress.bar import Bar as Bar
import time

import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torch.nn as nn

from utils.average_meter import AverageMeter, accuracy
from utils.csv_logger import *

def encode_onehot(labels, n_classes):
    '''
    One hot encode the labels, which is used for calculating the
    loss per sample
    '''
    #pdb.set_trace()
    onehot = torch.FloatTensor(labels.size()[0], n_classes)
    labels = labels.data

    if labels.is_cuda:
        onehot = onehot.cuda()

    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)

    return onehot

# def test(args, model, loader):
#     loss_func = nn.CrossEntropyLoss(size_average=False).cuda()
#
#     model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
#     correct = 0.
#     total = 0.
#     test_loss = 0.
#     for images, labels in loader:
#         if args.dataset == 'svhn':
#             labels = labels.type_as(torch.LongTensor()).view(-1) - 1
#
#         images = Variable(images, requires_grad=False).cuda()
#         labels = Variable(labels, requires_grad=False).cuda()
#         pred, _ = model(images)
#         test_loss += loss_func(pred, labels).data[0]
#         pred = torch.max(pred.data, 1)[1]
#         total += labels.size(0)
#         correct += (pred == labels.data).sum()
#     val_acc = correct / total
#     val_loss = test_loss / total
#     model.train()
#
#     return val_acc, val_loss

def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['learning_rate1'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['learning_rate1']

def test(args, model, criterion, test_loader):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(test_loader))

    for batch_idx, (images, labels) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = Variable(images, volatile=True).cuda()
        labels = Variable(labels).cuda()

        # compute output
        outputs, _ = model(images)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        losses.update(loss.data[0], images.size(0))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, top1.avg)

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
    #pdb.set_trace()
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)

    #batch_idx, (img, targ) = iter(train_loader)
    #pdb.set_trace()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=4)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    state = {k: v for k, v in args._get_kwargs()}

    updates = 0

    #labels = getattr(train_dataset, 'class_to_idx')
    #pdb.set_trace()
    model.train()

    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['learning_rate1']))

        xentropy_loss_avg = 0.
        xentropy_loss_extrap_avg = 0.
        correct = 0.
        total = 0.

        #progress_bar = tqdm(train_loader)
        bar = Bar('Processing', max=len(train_loader))
        for batch_idx, (images, labels) in enumerate(train_loader):
            #pdb.set_trace()
            #pdb.set_trace()
            #progress_bar.set_description('Epoch ' + str(updates))
            data_time.update(time.time() - end)

            if args.dataset == 'svhn':
                labels = labels.type_as(torch.LongTensor()).view(-1) - 1

            images = Variable(images).cuda(async=True)
            labels = Variable(labels).cuda(async=True)

            outputs, _ = model(images)
            xentropy_loss = criterion(outputs, labels)
            #pdb.set_trace()
            # # Compute the loss for each sample in the minibatch
            # onehot_target = Variable(encode_onehot(labels, args.num_classes))
            # pdb.set_trace()
            # xentropy_loss_vector = -1 * torch.sum(torch.log(F.softmax(pred))
            #                                                     * onehot_target,
            #                                                 dim=1,
            #                                                 keepdim=True)
            #pdb.set_trace()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            losses.update(xentropy_loss.data[0], images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            xentropy_loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #xentropy_loss_avg += xentropy_loss.data[0]

            # Calculate running average of accuracy
            # _, pred = torch.max(pred.data, 1)
            # total += labels.size(0)
            # correct += (pred == labels.data).sum()
            # accuracy = correct / total

            updates += 1

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(train_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
            # progress_bar.set_postfix(
            #     xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            #     acc='%.3f' % accuracy)


        torchvision.utils.save_image(images.data, 'random_results/' + args.folder + '/' + args.dataset + '_random_log_' + test_id + '.jpg', normalize=True)

        test_acc, test_loss = test(args, model, criterion, test_loader)
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

        elif args.dataset in ['svhn']:
            if updates >= 15000:
                break
        #print(str(cnn_optimizer.param_groups[0]['lr']))
    #torch.save(cnn.state_dict(), 'checkpoints/baseline_' + test_id + '.pt')
    random_logger.close()
