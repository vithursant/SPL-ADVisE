import torch
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from utils.misc import adjust_learning_rate
from utils.eval import *
from utils.misc import *
import time
import shutil

# progress bar
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar

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

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def train(args, trainloader, model, criterion, optimizer, use_cuda, updates, batch_builder, batch_train_inds):
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

        # if args.dataset == 'svhn':
        #     targets = targets.type_as(torch.LongTensor()).view(-1) - 1

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, _ = model(inputs)
        #loss = criterion(outputs, targets)

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

        torchvision.utils.save_image(inputs.data, args.checkpoint + '.jpg', normalize=True)
    bar.finish()

    return (losses.avg, top1.avg, updates)

def test(testloader, model, criterion, use_cuda):
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

def spld_selector(args, state, train_dataset, test_dataset, cnn, criterion, optimizer, use_cuda, logger):
    global spld_params

    best_acc = 0  # best test accuracy
    updates = 0

    # magnet loss parameters
    k = 8
    m = 8
    d = 8

    train_sampler = SubsetSequentialSampler(range(len(train_dataset)), range(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_batch,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.train_batch,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4)

    batch_train_inds = np.random.choice(range(len(train_dataset)), len(train_dataset), replace=False)
    batch_train_inds = batch_train_inds.astype(np.int32)
    train_loader.sampler.batch_indices = batch_train_inds

    if args.dataset in ['cifar10', 'svhn']:
        spld_params = [500, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['cifar100']:
        spld_params = [50, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['cub2002010', 'cub2002011']:
        k = 8
        m = 8
        d = 30
        spld_params = [5, 5e-1, 1e-1, 1e-1]
    elif args.dataset in ['mnist', 'fashionmnist']:
        #spld_params = [500, 1e-3, 5e-2, 1e-1]
        spld_params = [500, 5e-1, 1e-1, 1e-1]

    if args.dataset == 'svhn':
        labels = train_dataset.labels
        labels = np.hstack(labels)
    elif args.dataset in ['cub2002010', 'cub2002011']:
        labels = [int(i[1]) for i in train_dataset.imgs]
    else:
        labels = getattr(train_dataset, 'train_labels')
    #pdb.set_trace()
    if args.dataset == 'svhn':
        initial_reps = compute_reps(cnn, train_dataset, 4680)
    else:
        initial_reps = compute_reps(cnn, train_dataset, 1000)

    batch_builder = ClusterBatchBuilder(labels, k, m, d)
    batch_builder.update_clusters(args.dataset, initial_reps, max_iter=args.max_iter)

    epoch_steps = int(ceil(len(train_dataset)) / args.train_batch)
    n_steps = epoch_steps * args.epochs
    if args.dataset in ['cifar100']:
        n_steps = 250
    args.schedule = [int(ceil(len(train_dataset)) / args.train_batch) * args.schedule[0],
                    int(ceil(len(train_dataset)) / args.train_batch) * args.schedule[1]]

    for i in range(n_steps):
        if args.dataset in ['cifar10', 'cifar100']:
            optimizer = optim.SGD(cnn.parameters(), lr=learning_rate_cifar(args.learning_rate1, epoch), momentum=0.9, weight_decay=5e-4)
            print('SPLD LR: %f' % (learning_rate_cifar(args.learning_rate1, i)))
        else:
            state = adjust_learning_rate(args, state, optimizer, i)
            print(args.dataset + ' SPLD ' + 'Iteration: [%d | %d] LR: %f' % (i + 1, n_steps, state['learning_rate1']))

        train_loss, train_acc, updates = train(args, train_loader, cnn, criterion, optimizer, use_cuda, updates, batch_builder, batch_train_inds)
        test_loss, test_acc = test(test_loader, cnn, criterion, use_cuda)

        batch_train_inds = batch_builder.gen_batch_spl(spld_params[0], spld_params[1], args.train_batch)
        np.random.shuffle(batch_train_inds)
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
                'state_dict': cnn.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)
