import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from utils.misc import adjust_learning_rate, learning_rate_cifar
from utils.eval import *
from utils.misc import *

import os
import numpy as np
import pdb
from tqdm import tqdm, trange
import time
import shutil

# progress bar
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar

from utils.sampler import SubsetSequentialSampler
from utils.csv_logger import *

from magnet_loss.magnet_tools import *
from magnet_loss.magnet_loss import MagnetLoss
from magnet_loss.utils import *

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

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def train_magnet(   args,
                    embedding_model,
                    embedding_optimizer,
                    train_dataset,
                    train_loader,
                    minibatch_magnet_loss,
                    batch_class_inds,
                    batch_example_inds,
                    leap_magnet_logger,
                    batch_builder,
                    n_steps,
                    cluster_refresh_interval,
                    labels,
                    test_id):

    # magnet loss parameters
    k = 1
    m = 30
    d = 30
    alpha = 3.52

    updates = 0
    batch_losses = []
    progress_bar = tqdm(range(1000))
    for i in progress_bar:
        batch_loss_avg = 0.
        for batch_idx, (images, targets) in enumerate(train_loader):
            #progress_bar.set_description('Epoch ' + str(updates))
            images = Variable(images).cuda()
            targets = Variable(targets).cuda()

            embedding_model.zero_grad()
            pred, _ = embedding_model(images)

            batch_loss, batch_example_losses = minibatch_magnet_loss(pred,
                                                                    batch_class_inds,
                                                                    m,
                                                                    d,
                                                                    alpha)
            batch_loss.backward()
            embedding_optimizer.step()

            batch_loss_avg += batch_loss.data[0]
            updates += 1

        progress_bar.set_postfix(magnetloss='%.3f' % (batch_loss_avg / (batch_idx + 1)))

        batch_builder.update_losses(batch_example_inds,
                                    batch_example_losses, 'magnet')

        batch_losses.append(batch_loss.data[0])

        if not i % cluster_refresh_interval:
            print("Refreshing clusters")
            if args.dataset == 'svhn':
                reps = compute_reps(embedding_model, train_dataset, 4680)
            else:
                reps = compute_reps(embedding_model, train_dataset, 400)
            batch_builder.update_clusters(args.dataset, reps)

        if args.plot:
            if not i % 1000:
                n_plot = 8000
                if args.dataset == 'svhn':
                    plot_embedding(compute_reps(embedding_model, train_dataset, 4680)[:n_plot], labels[:n_plot], name='leap_results/magnet/' + args.dataset + '_leap_magnet_log_' + test_id + '_' + str(i))
                else:
                    plot_embedding(compute_reps(embedding_model, train_dataset, 1000)[:n_plot], labels[:n_plot], name='leap_results/magnet/' + args.dataset + '_leap_magnet_log_' + test_id + '_' + str(i))

        batch_example_inds, batch_class_inds = batch_builder.gen_batch()
        train_loader.sampler.batch_indices = batch_example_inds

        row = {'epoch': str(updates), 'batch_loss': str(batch_loss.data[0])}
        leap_magnet_logger.writerow(row)

    if args.plot:
        plot_smooth(batch_losses, args.dataset + '_' + test_id + '_batch-losses')

def leap_selector(  args,
                    state,
                    train_dataset,
                    test_dataset,
                    student_model,
                    embedding_model,
                    criterion,
                    optimizer,
                    use_cuda,
                    logger):

    global spld_params
    #args.train_batch = 64
    # Magnet Loss
    if not os.path.exists('leap_results'):
        os.makedirs('leap_results')
    if not os.path.exists('leap_results/magnet'):
        os.makedirs('leap_results/magnet')

    i = 0
    while os.path.isfile('leap_results/magnet/' + args.dataset + '_leap_magnet_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)
    leap_magnet_logger = CSVLogger(args=args, filename='leap_results/magnet/' + args.dataset + '_leap_magnet_log_' + test_id + '.csv', fieldnames=['epoch', 'batch_loss'])

    embedding_model = torch.nn.DataParallel(embedding_model).cuda()
    args.train_batch = 64
    n_train = len(train_dataset)

    train_sampler = SubsetSequentialSampler(range(len(train_dataset)), range(args.train_batch))
    train_loader = DataLoader(train_dataset,
                             batch_size=args.train_batch,
                             shuffle=False,
                             num_workers=4,
                             sampler=train_sampler)

    test_loader = DataLoader(test_dataset,
                            batch_size=args.train_batch,
                            shuffle=True,
                            num_workers=4)

    # magnet loss parameters
    k = 8
    m = 8
    d = 8
    alpha = 1.0

    embedding_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=args.learning_rate2)
    minibatch_magnet_loss = MagnetLoss()

    if args.dataset == 'svhn':
        labels = train_dataset.labels.flatten()
    elif args.dataset == 'cub2002010':
        labels = [int(i[1]) for i in train_dataset.imgs]
    else:
        labels = getattr(train_dataset, 'train_labels')

    if args.dataset == 'svhn':
        initial_reps = compute_reps(embedding_model, train_dataset, 4680)
    else:
        initial_reps = compute_reps(embedding_model, train_dataset, 1000)

    batch_builder = ClusterBatchBuilder(labels, k, m, d)
    batch_builder.update_clusters(args.dataset, initial_reps, max_iter=args.max_iter)

    batch_example_inds, batch_class_inds = batch_builder.gen_batch()
    train_loader.sampler.batch_indices = batch_example_inds.astype(np.int32)

    if args.dataset in ['mnist', 'fashionmnist']:
        n_epochs = 15
    elif args.dataset in ['cifar10', 'cifar100', 'svhn']:
        n_epochs = 50
    elif args.dataset in ['tinyimagenet']:
        n_epochs = 50
    elif args.dataset in ['cub2002010', 'cub2002011']:
        n_epochs = 50

    epoch_steps = int(ceil(float(len(train_dataset) / args.train_batch)))
    #epoch_steps = len(train_loader)
    n_steps = epoch_steps * n_epochs
    cluster_refresh_interval = epoch_steps

    if args.dataset in ['svhn']:
        n_steps = 8000
    if args.dataset in ['cifar100']:
        n_steps = 1000

    _ = embedding_model.train()

    updates = 0
    batch_losses = []
    progress_bar = tqdm(range(n_steps))
    for i in progress_bar:
        batch_loss_avg = 0.
        for batch_idx, (images, targets) in enumerate(train_loader):
            #progress_bar.set_description('Epoch ' + str(updates))
            images = Variable(images).cuda()
            targets = Variable(targets).cuda()

            embedding_model.zero_grad()
            pred, _ = embedding_model(images)

            batch_loss, batch_example_losses = minibatch_magnet_loss(pred,
                                                                    batch_class_inds,
                                                                    m,
                                                                    d,
                                                                    alpha)
            batch_loss.backward()
            embedding_optimizer.step()

            batch_loss_avg += batch_loss.data[0]
            updates += 1

        progress_bar.set_postfix(magnetloss='%.3f' % (batch_loss_avg / (batch_idx + 1)))

        batch_builder.update_losses(batch_example_inds,
                                    batch_example_losses, 'magnet')

        batch_losses.append(batch_loss.data[0])

        if not i % cluster_refresh_interval:
            print("Refreshing clusters")
            if args.dataset == 'svhn':
                reps = compute_reps(embedding_model, train_dataset, 4680)
            else:
                reps = compute_reps(embedding_model, train_dataset, 400)
            batch_builder.update_clusters(args.dataset, reps, max_iter=args.max_iter)

        if args.plot:
            if not i % args.plot_interval:
                n_plot = args.plot_num_samples
                if args.dataset == 'svhn':
                    plot_embedding( compute_reps(embedding_model, train_dataset, 4680)[:n_plot],
                                    labels[:n_plot],
                                    name='leap_results/magnet/' + args.dataset + '_leap_magnet_log_' + test_id + '_' + str(i),
                                    num_classes=args.num_classes)
                else:
                    plot_embedding(compute_reps(embedding_model, train_dataset, 1000)[:n_plot],
                                                labels[:n_plot],
                                                name='leap_results/magnet/' + args.dataset + '_leap_magnet_log_' + test_id + '_' + str(i),
                                                num_classes=args.num_classes)

        batch_example_inds, batch_class_inds = batch_builder.gen_batch()
        train_loader.sampler.batch_indices = batch_example_inds

        row = {'epoch': str(updates), 'batch_loss': str(batch_loss.data[0])}
        leap_magnet_logger.writerow(row)

    if args.plot:
        plot_smooth(batch_losses, args.dataset + '_' + test_id + '_batch-losses')

    leap_magnet_logger.close()

    # Student CNN
    student_model = torch.nn.DataParallel(student_model).cuda()

    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        args.train_batch = 128
    else:
        args.train_batch = 64

    train_sampler = SubsetSequentialSampler(range(len(train_dataset)), range(len(train_dataset)))
    train_loader = DataLoader(train_dataset,
                             batch_size=args.train_batch,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             sampler=train_sampler)

    test_loader = DataLoader(test_dataset,
                            batch_size=args.train_batch,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4)

    batch_train_inds = np.random.choice(range(len(train_dataset)), len(train_dataset), replace=False)
    train_loader.sampler.batch_indices = batch_train_inds.astype(np.int32)
    #pdb.set_trace()

    updates = 0
    num_samples = 0

    epoch_steps = int(ceil(len(train_dataset)) / args.train_batch)
    n_steps = epoch_steps * 15

    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        spld_params = [500, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['mnist', 'fashionmnist']:
        spld_params = [500, 5e-1, 1e-1, 1e-1]
        #spld_params = [500, 1e-3, 5e-2, 1e-1]
        #spld_params = [100, 1e-3, 5e-2, 1e-1]

    if args.dataset == 'svhn':
        labels = train_dataset.labels
        labels = np.hstack(labels)
    elif args.dataset == 'cub2002010':
        labels = [int(i[1]) for i in train_dataset.imgs]
    else:
        labels = getattr(train_dataset, 'train_labels')

    args.schedule = [int(ceil(len(train_dataset)) / args.train_batch) * args.schedule[0],
                    int(ceil(len(train_dataset)) / args.train_batch) * args.schedule[1]]

    best_acc = 0  # best test accuracy
    updates = 0
    for i in range(n_steps):
        if args.dataset in ['cifar10', 'cifar100']:
            optimizer = optim.SGD(student_model.parameters(), lr=learning_rate_cifar(args.learning_rate1, i), momentum=0.9, weight_decay=5e-4)
            print('LEAP LR: %f' % (learning_rate_cifar(args.learning_rate1, i)))
        else:
            state = adjust_learning_rate(args, state, optimizer, i)
            print(args.dataset + ' LEAP ' + 'Iteration: [%d | %d] LR: %f' % (i + 1, n_steps, state['learning_rate1']))

        train_loss, train_acc, updates = train_student(args, student_model, train_loader, optimizer, use_cuda, updates, batch_builder, batch_train_inds)
        test_loss, test_acc = test_student(test_loader, student_model, criterion, use_cuda)

        batch_train_inds = batch_builder.gen_batch_spl(spld_params[0], spld_params[1], args.train_batch)
        train_loader.sampler.batch_indices = batch_train_inds.astype(np.int32)

        # Increase the learning pace
        spld_params[0] *= (1+spld_params[2])
        spld_params[0] = int(round(spld_params[0]))
        spld_params[1] *= (1+spld_params[3])

        # append logger file
        logger.append([updates, state['learning_rate1'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'iteration': i + 1,
                'updates': updates,
                'state_dict': student_model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

        if args.dataset == 'cifar10':
            if updates >= 200*390:
                break

        elif args.dataset in ['mnist','fashionmnist']:
            if updates >= 60*390:
                break

        elif args.dataset in ['svhn']:
            if updates >= 15000:
                break

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train_student(args, model, trainloader, optimizer, use_cuda, updates, batch_builder, batch_train_inds):

    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.dataset == 'svhn':
            targets = targets.type_as(torch.LongTensor()).view(-1) - 1

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, features = model(inputs)

        # Compute the loss for each sample in the minibatch
        onehot_target = Variable(encode_onehot(targets, args.num_classes))
        loss_vector = -1 * torch.sum(torch.log(F.softmax(outputs))
                                                            * onehot_target,
                                                        dim=1,
                                                        keepdim=True)
        loss = loss_vector.mean()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # number of samples
        updates += targets.data.shape[0]

        # update losses in cluster
        start = batch_idx * inputs.size(0)
        stop = start + inputs.size(0)
        batch_builder.update_losses(batch_train_inds[start:stop],
                                    loss_vector.squeeze(), 'spld')


        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
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

    return (losses.avg, top1.avg, updates)

def test_student(testloader, model, criterion, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
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
