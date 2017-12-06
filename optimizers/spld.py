import os
import numpy as np
import pdb

import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torch.nn as nn

from utils.csv_logger import *
from utils.sampler import SubsetSequentialSampler

from magnet_loss.magnet_tools import *

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

def self_paced_learning_with_diversity(args, train_dataset, test_dataset, optimizer, model, scheduler):
    if not os.path.exists('spld_results'):
        os.makedirs('spld_results')
    if not os.path.exists('spld_results/baseline'):
        os.makedirs('spld_results/baseline')

    i = 0
    while os.path.isfile('spld_results/' + args.folder + '/' + args.dataset + '_spld_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)

    spld_logger = CSVLogger(args=args, filename='spld_results/' + args.folder + '/' + args.dataset + '_spld_log_' + test_id + '.csv', fieldnames=['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])

    train_sampler = SubsetSequentialSampler(range(len(train_dataset)), range(len(train_dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4)

    batch_train_inds = np.random.choice(range(len(train_dataset)), len(train_dataset), replace=False)
    train_loader.sampler.batch_indices = batch_train_inds.astype(np.int32)

    k = 8
    m = 8
    d = 8

    updates = 0
    num_samples = 0

    epoch_steps = int(ceil(len(train_dataset)) / args.batch_size)
    n_steps = epoch_steps * 15

    if args.dataset in ['cifar10', 'svhn']:
        spld_params = [500, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['cifar100']:
        spld_params = [10, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['mnist', 'fashionmnist']:
        #spld_params = [500, 1e-3, 5e-2, 1e-1]
        spld_params = [500, 5e-1, 1e-1, 1e-1]

    if args.dataset == 'svhn':
        labels = train_dataset.labels
        labels = np.hstack(labels)
    else:
        labels = getattr(train_dataset, 'train_labels')

    if args.dataset == 'svhn':
        initial_reps = compute_reps(model, train_dataset, 4680)
    else:
        initial_reps = compute_reps(model, train_dataset, 400)

    batch_builder = ClusterBatchBuilder(labels, k, m, d)
    batch_builder.update_clusters(args.dataset, initial_reps, max_iter=args.max_iter)

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

            model.zero_grad()
            pred, features = model(images)

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

            # Backward
            xentropy_loss_vector_mean.backward()
            optimizer.step()

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

        torchvision.utils.save_image(images.data, 'spld_results/' + args.folder + '/' + args.dataset + '_spld_log_' + test_id + '.jpg', normalize=True)

        test_acc, test_loss = test(args, model, test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        row = {'epoch': str(updates), 'train_acc': str(accuracy), 'train_loss': str(xentropy_loss_avg / (updates)), 'test_acc': str(test_acc), 'test_loss': str(test_loss)}
        spld_logger.writerow(row)

        if args.en_scheduler:
            scheduler.step(updates)

        batch_train_inds = batch_builder.gen_batch_spl(spld_params[0], spld_params[1], args.batch_size)
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
        elif args.dataset in ['svhn']:
            if updates >= 15000:
                break

    spld_logger.close()
