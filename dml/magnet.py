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

def magnet(args, train_dataset, test_dataset, embedding_cnn, scheduler):
    if not os.path.exists('magnet_results'):
        os.makedirs('magnet_results')
    if not os.path.exists('magnet_results/baseline'):
        os.makedirs('magnet_results/baseline')

    i = 0
    while os.path.isfile('magnet_results/' + args.folder + '/' + args.dataset + '_magnet_log_' + str(i) + '.csv'):
        i += 1
    args.test_id = i
    test_id = str(args.test_id)

    magnet_logger = CSVLogger(args=args, filename='magnet_results/' + args.folder + '/' + args.dataset + '_magnet_log_' + test_id + '.csv', fieldnames=['epoch', 'batch_loss'])

    embedding_cnn = torch.nn.DataParallel(embedding_cnn).cuda()
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

    optimizer = torch.optim.Adam(embedding_cnn.parameters(), lr=args.learning_rate2)
    minibatch_magnet_loss = MagnetLoss()

    if args.dataset == 'svhn':
        labels = train_dataset.labels.flatten()
    elif args.dataset == 'tinyimagenet':
        # class_map_file = "/scratch/vthangar/tiny-imagenet-200/words.txt"
        # with open(class_map_file) as f:
        #     class_map = f.readlines()
        # class_map = [x.strip().split('\t') for x in class_map]
        #
        # wnids_file = "/scratch/vthangar/tiny-imagenet-200/wnids.txt"
        #
        # with open(wnids_file, 'r') as f:
        #     wnids = f.readlines()
        # wnids = [x.strip() for x in wnids]
        # wnids.sort()
        # ID_to_label = {}
        # for label, ID in enumerate(wnids):
        #     ID_to_label[ID] = label
        # pdb.set_trace()
        #from os import listdir
        #from os.path import isfile, join

        labels = "/scratch/vthangar/tiny-imagenet-200/wnids.txt"
        # First load wnids
        with open(labels, 'r') as f:
            wnids = [x.strip() for x in f]

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}
        pdb.set_trace()
        # #nsamples = int(round(sample*500))
        # label_dict=dict()
        # labels=open(labels,'r').read().split('\n')
        #
        # while '' in labels:
        #     labels.remove('')
        #
        # for i in range(len(labels)):
        #     label_dict[labels[i]] = i
        #
        # print "built dictionary of labels (id --> number) "
        #
        # file_names=[]
        # labels=[]
        #
        # for label in label_dict:
        #     #print str(label)
        #     cur_dir = "/scratch/vthangar/tiny-imagenet-200/train/" + label + "/images"
        #     onlyfiles = [f for f in listdir(cur_dir) if isfile(join(cur_dir, f))]
        #     onlyfiles = [cur_dir + '/' + f for f in onlyfiles]
        #     file_names = file_names+onlyfiles
        #     print str(len(file_names))
        #     pdb.set_trace()
        #     #cur_labels=nsamples*[label_dict[label]]
        #     #labels = labels + cur_labels
        #     print str(len(labels))
        # print str(file_names)
        # pdb.set_trace()
        # our_index = file_names.index(cur_dir + '/' + sys.argv[1])
        # #print str(our_index)
        # outf=open('train_data_names.txt','w')
        # outf.write('\n'.join(file_names))
    else:
        labels = getattr(train_dataset, 'train_labels')
    pdb.set_trace()
    if args.dataset == 'svhn':
        initial_reps = compute_reps(embedding_cnn, train_dataset, 4680)
    else:
        initial_reps = compute_reps(embedding_cnn, train_dataset, 400)

    batch_builder = ClusterBatchBuilder(labels, k, m, d)
    batch_builder.update_clusters(args.dataset, initial_reps, max_iter=args.max_iter)

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

    if args.dataset in ['svhn']:
        n_steps = 8000

    _ = embedding_cnn.train()
    updates = 0

    progress_bar = tqdm(range(n_steps))
    for i in progress_bar:
        batch_loss_avg = 0.

        for batch_idx, (images, targets) in enumerate(train_loader):
            #progress_bar.set_description('Epoch ' + str(updates))
            images = Variable(images).cuda()
            targets = Variable(targets).cuda()

            embedding_cnn.zero_grad()
            pred, _ = embedding_cnn(images)

            batch_loss, batch_example_losses = minibatch_magnet_loss(pred,
                                                                    batch_class_inds,
                                                                    m,
                                                                    d,
                                                                    alpha)
            batch_loss.backward()
            optimizer.step()

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
                reps = compute_reps(embedding_cnn, train_dataset, 4680)
            else:
                reps = compute_reps(embedding_cnn, train_dataset, 400)
            batch_builder.update_clusters(args.dataset, reps)

        if args.plot:
            if not i % 2000:
                n_plot = 8000
                #pdb.set_trace()
                if args.dataset == 'svhn':
                    plot_embedding(compute_reps(embedding_cnn, train_dataset, 4680)[:n_plot],
                                                labels[:n_plot],
                                                name='magnet_results/' + args.folder + '/' + args.dataset + '_magnet_log_' + test_id + '_' + str(i))
                else:
                    plot_embedding(compute_reps(embedding_cnn, train_dataset, 400)[:n_plot],
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
