# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

__docformat__ = 'restructedtext en'

import six.moves.cPickle as cPickle
import gzip
import os
import os.path
import sys
import timeit
import time
import csv
import shutil
import numpy as np

from math import ceil
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import visdom

from utils.train_settings import parse_settings
from utils.cluster_dataset import group_data, compute_clusters, get_cluster
from datasets.load_dataset import load_dataset

from models.lenet import LeNet, FashionLeNet, LeNetCifar10
from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from models.resnet import ResNetCifar10
from models.googlenet import GoogLeNet
from models.wide_resnet import Wide_ResNet
from models.resnet_all import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from utils.logger import CSVLogger
from utils.average_meter import AverageMeter
from visualizer.visualize import VisdomLinePlotter

from magnet_loss.magnet_tools import *

import pdb
import random

args = parse_settings()

args.cuda = not args.no_cuda and torch.cuda.is_available()

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

csv_logger = CSVLogger(filename='results/' + args.folder + '/log_baseline' + test_id + '.csv', fieldnames=['updates', 'train_acc', 'test_acc', 'learning_rate'])

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Saves checkpoint to disk
    """
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

def encode_onehot(labels, n_classes):
	"""
	One hot encode the labels, which is used for calculating the
	loss per sample
	"""
	onehot = torch.FloatTensor(labels.size()[0], n_classes)
	labels = labels.data

	if labels.is_cuda:
		onehot = onehot.cuda()

	onehot.zero_()
	onehot.scatter_(1, labels.view(-1, 1), 1)

	return onehot

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(model, testloader, optimizer, updates, lr_scheduler):
	#Testing
	loss_vec = np.array([])
	batch_time = AverageMeter()
	losses = AverageMeter()
	accs = AverageMeter()
	end = time.time()

	model.eval()

	for batch_idx, (img, target) in enumerate(testloader):
		img = Variable(img, volatile=True).cuda()
		target = Variable(target, volatile=True).cuda(async=True)

		output, _ = model(img)

		# Compute the loss for each sample in the minibatch
		onehot_target = Variable(encode_onehot(target, args.num_classes))
		xentropy_loss_vector = -1 * torch.sum(torch.log(F.softmax(output))
                                                * onehot_target,
                                              dim=1,
                                              keepdim=True)

		# Sum minibatch loss vector
		loss_vec = np.append(loss_vec,
								xentropy_loss_vector.data.cpu().numpy())

		xentropy_loss_vector_sum = xentropy_loss_vector.sum()
		xentropy_loss_vector_mean = xentropy_loss_vector.mean()

        # measure accuracy and record loss
		prec1 = accuracy(output.data, target.data, topk=(1,))[0]
		losses.update(xentropy_loss_vector_sum.data[0], img.size(0))
		accs.update(prec1[0], img.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if batch_idx % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {accs.val:.3f} ({accs.avg:.3f})'.format(
					batch_idx, len(testloader), batch_time=batch_time, loss=losses,
					accs=accs))

	lr_scheduler.step(accs.avg)

	# log avg values to somewhere
	if args.visdom:
		plotter.plot('acc', 'test', updates, accs.avg)
		plotter.plot('loss', 'test', updates, losses.avg)

	return accs.avg

def train(model, batch_builder, trainloader, trainset, n_train, optimizer, iteration, n_steps, updates, batch_train_inds):
	#adjust_learning_rate(optimizer, i+1)
	loss_vec = np.array([])
	batch_time = AverageMeter()
	losses = AverageMeter()
	accs = AverageMeter()

    # switch to train mode
	model.train()

	#pdb.set_trace()
	end = time.time()
	
	for batch_idx, (img, target) in tqdm(enumerate(trainloader)):
		img = Variable(img).cuda()
		target = Variable(target).cuda(async=True)

		optimizer.zero_grad()
		output, features = model(img)

		# Compute the loss for each sample in the minibatch
		onehot_target = Variable(encode_onehot(target, args.num_classes))
		xentropy_loss_vector = -1 * torch.sum(torch.log(F.softmax(output))
                                                            * onehot_target,
                                                        dim=1,
                                                        keepdim=True)

		# Sum minibatch loss vector
		loss_vec = np.append(loss_vec,
								xentropy_loss_vector.data.cpu().numpy())
		xentropy_loss_vector_sum = xentropy_loss_vector.sum()
		xentropy_loss_vector_mean = xentropy_loss_vector.mean()

		# Backward
		xentropy_loss_vector_sum.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# measure accuracy and record loss
		prec1 = accuracy(output.data, target.data, topk=(1,))[0]
		losses.update(xentropy_loss_vector_sum.data[0], img.size(0))
		accs.update(prec1[0], img.size(0))

		if batch_idx % args.print_freq == 0:
			print('Iteration: [{0}][{1}/{2}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {accs.val:.3f} ({accs.avg:.3f})'.format(
					batch_idx, n_steps, len(trainloader), batch_time=batch_time,
					loss=losses, accs=accs))

		start = batch_idx * img.size(0)
		stop = start + img.size(0)
		updates += 1
		batch_builder.update_losses(batch_train_inds[start:stop],
								xentropy_loss_vector.squeeze())

	batch_train_inds = batch_builder.gen_batch_spl(spld_params[0], spld_params[1], spld_params[2])

	batch_train_inds = batch_train_inds[0]

	trainloader.sampler.batch_indices = batch_train_inds

	# Increase the learning pace
	spld_params[0] *= (1+spld_params[2])
	spld_params[0] = int(round(spld_params[0]))
	spld_params[1] *= (1+spld_params[3])

	if iteration > 0:
		if args.visdom:
			plotter.plot('acc', 'train', updates, accs.avg)
			plotter.plot('loss', 'train', updates, losses.avg)

	return updates, batch_train_inds, accs.avg

def main(trainloader, trainset, testloader, n_train):

	best_prec1 = 0
	updates = 0

	m = 8
	k = 8
	d = 8

	batch_losses = []

	epoch_steps = int(ceil(len(trainset)) / args.batch_size)
	n_steps = epoch_steps * 15
	#pdb.set_trace()

	global plotter
	plotter = VisdomLinePlotter(env_name=args.name)

	if args.spl:
		print("Start Experiment for " + str(args.model) + " on " + args.dataset)
		#spld_params = [50, 1e-3, 5e-2, 1e-1]
		global spld_params 
		spld_params = [50, 4e-1, 1e-1, 1e-1]

		prev_mean_all_loss = float('inf')

		if args.dataset == 'mnist':
			model = LeNet()
			model.cuda()
			print(model)

		if args.dataset == 'cifar10':
			num_classes = 10

			if args.model == 'resnet18':
				model = PreActResNet18(num_classes=num_classes)
				model = ResNet18(num_classes=num_classes)
			elif args.model == 'resnet34':
				model = PreActResNet34(num_classes=num_classes)
				model = ResNet34(num_classes=num_classes)
			elif args.model == 'resnet50':
				model = PreActResNet50(num_classes=num_classes)
				model = ResNet150(num_classes=num_classes)
			elif args.model == 'resnet101':
				model = PreActResNet101(num_classes=num_classes)
				model = ResNet101(num_classes=num_classes)
			elif args.model == 'resnet152':
				model = PreActResNet152(num_classes=num_classes)
				model = ResNet152(num_classes=num_classes)
			elif args.model == 'wideresnet':
				model = Wide_ResNet(28, 10, 0.3, num_classes)

			model.cuda()
			#model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
			#cudnn.benchmark = True

			print(model)

		if args.dataset == 'cifar100':
			num_classes = 100

			if args.model == 'resnet18':
				model = PreActResNet18(num_classes=num_classes)
				model = ResNet18(num_classes=num_classes)
			elif args.model == 'resnet34':
				model = PreActResNet34(num_classes=num_classes)
				model = ResNet34(num_classes=num_classes)
			elif args.model == 'resnet50':
				model = PreActResNet50(num_classes=num_classes)
				model = ResNet150(num_classes=num_classes)
			elif args.model == 'resnet101':
				model = PreActResNet101(num_classes=num_classes)
				model = ResNet101(num_classes=num_classes)
			elif args.model == 'resnet152':
				model = PreActResNet152(num_classes=num_classes)
				model = ResNet152(num_classes=num_classes)
			elif args.model == 'wideresnet':
				model = Wide_ResNet(28, 10, 0.3, num_classes)

		if args.dataset == 'fashionmnist':
			model = FashionLeNet()
			mode.cuda()
			print(model)

		criterion = nn.CrossEntropyLoss(size_average=False)

		if args.dataset == 'cifar10':
			optimizer = optim.SGD(model.parameters(),
	                              lr=args.lr,
	                              momentum=args.momentum,
	                              nesterov=True,
	                              weight_decay=args.weight_decay)

		if args.dataset == 'cifar100':
			optimizer = optim.SGD(model.parameters(),
	                              lr=args.lr,
	                              momentum=args.momentum,
	                              nesterov=True,
	                              weight_decay=args.weight_decay)

		if args.dataset == 'mnist':
			optimizer = optim.SGD(model.parameters(),
	                              lr=args.lr,
	                              momentum=args.momentum)

		lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_ratio,
		                                 patience=args.lr_patience, verbose=True,
		                                 threshold=args.lr_threshold, threshold_mode='abs',
		                                 cooldown=args.lr_delay)

		print('----lr scheduler: lr:{} DecayRatio:{} Patience:{} Threshold:{} Before_epoch:{}'.format(args.lr, args.lr_decay_ratio, args.lr_patience, args.lr_threshold, args.lr_delay))	

		# Create batcher
		labels = getattr(trainset, 'train_labels')
		#pdb.set_trace()

		# Extract training features
		initial_reps = compute_reps(model, trainset, 100)

		# Build Cluster
		batch_builder = ClusterBatchBuilder(labels, k, m, d)
		# Update cluster based on training features
		batch_builder.update_clusters(initial_reps)

		#pdb.set_trace()
		losses = AverageMeter()
		accs = AverageMeter()

		_ = model.train()

		for i in tqdm(range(n_steps)):
			# Forward and backward on the entire dataset for the first iteration
			if i == 0:
				batch_train_inds = np.arange(len(trainset))
				trainloader.sampler.batch_indices = batch_train_inds

			updates, batch_train_inds, accs_avg = train(model,
														batch_builder,
														trainloader, 
														trainset, 
														n_train, 
														optimizer,
														i,
														n_steps,
														updates,
														batch_train_inds)

			if i > 0:
				# evaluate on validation set
				prec1 = validate(model, testloader, optimizer, updates, lr_scheduler)
				row = {'updates': str(updates), 'train_acc': str(accs_avg), 'test_acc': str(prec1), 'learning_rate': str(optimizer.param_groups[0]['lr'])}
				csv_logger.writerow(row)

				# remember best prec@1 and save checkpoint
				is_best = prec1 > best_prec1
				best_prec1 = max(prec1, best_prec1)

				save_checkpoint({
					'epoch': updates + 1,
					'state_dict': model.state_dict(),
					'best_prec1': best_prec1,
				}, is_best)

			# Refresh the cluster every frequency
			if i % 10 == 0:
				print("Refreshing clusters")
				reps = compute_reps(model, trainset, 400)
				batch_builder.update_clusters(reps)

		print ('Best accuracy: ', best_prec1)
		csv_logger.close()

if __name__ == '__main__':

	trainloader, testloader, trainset, testset, n_train = load_dataset(args)
	n_train_batches = len(trainset) // args.batch_size
	n_test_batches =  len(testset) // args.batch_size

	print("==>>> Total Training Batch Number: {}".format(len(trainset)))
	print("==>>> Total Number of Train Batches: {}".format(n_train_batches))
	print("==>>> Total Number of Test Batches: {}".format(n_test_batches))

	print("Starting Training...")
	main(trainloader, trainset, testloader, n_train)