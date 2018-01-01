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
from utils.sampler import SubsetSequentialSampler

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

def leap(args, train_dataset, test_dataset, optimizer, embedding_model, student_model, logger):
    if not os.path.exists('leap_results'):
        os.makedirs('leap_results')
    if not os.path.exists('leap_results/baseline'):
        os.makedirs('leap_results/baseline')
    if not os.path.exists('leap_results/baseline/magnet'):
        os.makedirs('leap_results/baseline/magnet')

    i = 0
    while os.path.isfile('leap_results/' + args.folder + '/' + args.dataset + '_leap_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)

    i = 0
    while os.path.isfile('leap_results/' + args.folder + '/magnet/' + args.dataset + '_leap_magnet_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)

    leap_logger = CSVLogger(args=args, filename='leap_results/' + args.folder + '/' + args.dataset + '_leap_log_' + test_id + '.csv', fieldnames=['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])
    leap_magnet_logger = CSVLogger(args=args, filename='leap_results/' + args.folder + '/magnet/' + args.dataset + '_leap_magnet_log_' + test_id + '.csv', fieldnames=['epoch', 'batch_loss'])

    embedding_model = torch.nn.DataParallel(embedding_model).cuda()
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

    if args.dataset in ['cifar100']:
        k = 8
        m = 8
        d = 8
        alpha = 1.0
    else:
        k = 8
        m = 8
        d = 8
        alpha = 1.0

    #embedding_optimizer = torch.optim.Adam(embedding_model.parameters(), lr=args.learning_rate2)
    embedding_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=args.learning_rate2)
    minibatch_magnet_loss = MagnetLoss()

    if args.dataset == 'svhn':
        labels = train_dataset.labels.flatten()
    else:
        labels = getattr(train_dataset, 'train_labels')

    if args.dataset == 'svhn':
        initial_reps = compute_reps(embedding_model, train_dataset, 4680)
    else:
        initial_reps = compute_reps(embedding_model, train_dataset, 400)

    batch_builder = ClusterBatchBuilder(labels, k, m, d)
    batch_builder.update_clusters(args.dataset, initial_reps, max_iter=args.max_iter)

    batch_losses = []

    batch_example_inds, batch_class_inds = batch_builder.gen_batch()
    train_loader.sampler.batch_indices = batch_example_inds.astype(np.int32)

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

    if args.dataset in ['svhn']:
        n_steps = 8000

    _ = embedding_model.train()
    updates = 0

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
                reps = compute_reps(embedding_model, train_dataset, 4680)
            else:
                reps = compute_reps(embedding_model, train_dataset, 400)
            batch_builder.update_clusters(args.dataset, reps)

        if args.plot:
            if not i % 1000:
                n_plot = 8000
                if args.dataset == 'svhn':
                    plot_embedding(compute_reps(embedding_model, train_dataset, 4680)[:n_plot], labels[:n_plot], name='leap_results/' + args.folder + '/magnet/' + args.dataset + '_leap_magnet_log_' + test_id + '_' + str(i))
                else:
                    plot_embedding(compute_reps(embedding_model, train_dataset, 400)[:n_plot], labels[:n_plot], name='leap_results/' + args.folder + '/magnet/' + args.dataset + '_leap_magnet_log_' + test_id + '_' + str(i))

        batch_example_inds, batch_class_inds = batch_builder.gen_batch()
        train_loader.sampler.batch_indices = batch_example_inds

        row = {'epoch': str(updates), 'batch_loss': str(batch_loss.data[0])}
        leap_magnet_logger.writerow(row)

    if args.plot:
        plot_smooth(batch_losses, args.dataset + '_' + test_id + '_batch-losses')

    leap_magnet_logger.close()

    student_model = torch.nn.DataParallel(student_model).cuda()

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
    train_loader.sampler.batch_indices = batch_train_inds.astype(np.int32)
    #pdb.set_trace()

    updates = 0
    num_samples = 0

    epoch_steps = int(ceil(len(train_dataset)) / args.batch_size)
    n_steps = epoch_steps * 15

    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        spld_params = [500, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['mnist', 'fashionmnist']:
        spld_params = [500, 5e-1, 1e-1, 1e-1]
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

            student_model.zero_grad()
            pred, features = student_model(images)

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

            updates += 1

            batch_builder.update_losses(batch_train_inds[start:stop],
                                    xentropy_loss_vector.squeeze(), 'spld')

        torchvision.utils.save_image(images.data, 'leap_results/' + args.folder + '/' + args.dataset + '_leap_log_' + test_id + '.jpg', normalize=True)

        test_acc, test_loss = test(args, student_model, test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        row = {'epoch': str(updates), 'train_acc': str(accuracy), 'train_loss': str(xentropy_loss_avg / (updates)), 'test_acc': str(test_acc), 'test_loss': str(test_loss)}
        leap_logger.writerow(row)

        if args.en_scheduler:
            scheduler.step(updates)

        # row = {'epoch': str(updates), 'train_acc': str(accuracy), 'train_loss': str(xentropy_loss_avg), 'test_acc': str(test_acc), 'test_loss': str(test_loss)}
        # spldml_logger.writerow(row)

        batch_train_inds = batch_builder.gen_batch_spl(spld_params[0], spld_params[1], args.batch_size)
        train_loader.sampler.batch_indices = batch_train_inds.astype(np.int32)

        # Increase the learning pace
        spld_params[0] *= (1+spld_params[2])
        spld_params[0] = int(round(spld_params[0]))
        spld_params[1] *= (1+spld_params[3])

        # append logger file
        logger.append([updates, state['learning_rate1'], str(xentropy_loss_avg / (updates)), str(test_loss), str(accuracy), str(test_acc)])

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
    leap_logger.close()
