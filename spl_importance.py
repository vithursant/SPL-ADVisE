# -*- coding: utf-8 -*-

from __future__ import print_function

__docformat__ = 'restructedtext en'

import pdb

import six.moves.cPickle as cPickle
import gzip
import os
import os.path
import sys
import timeit
import time
import csv
import shutil

from math import ceil

import numpy
numpy.random.seed(5)

from tqdm import tqdm

import matplotlib.pyplot as pyplot

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

import torchvision
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
import torch.backends.cudnn as cudnn

from torch.optim.lr_scheduler import ReduceLROnPlateau

import visdom

from utils.train_settings import parse_settings
from utils.cluster_dataset import group_data, compute_clusters, get_cluster
from datasets.load_dataset import load_dataset

from models.lenet import LeNet, FashionLeNet, LeNetCifar10
from models.preact_resnet import PreActResNet18
from models.resnet import ResNetCifar10
from models.googlenet import GoogLeNet
from models.wide_resnet import Wide_ResNet
from models.resnet_all import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from utils.sampler import SubsetSequentialSampler
from utils.min_max_sort import min_max_sort

import matplotlib.pyplot as plt

from utils.average_meter import AverageMeter
from visualizer.visualize import VisdomLinePlotter

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

from magnet_loss.magnet_tools import *

from utils.sampler import SubsetSequentialSamplerSPLDML

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

csv_logger = CSVLogger(filename='results/' + args.folder + '/log_baseline' + test_id + '.csv', fieldnames=['updates', 'train_acc', 'train_loss', 'test_acc', 'test_loss', 'learning_rate'])

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

from utils.reweighting import BiasedReweightingPolicy 

def BiasedReweightingPolicy(k=1.0, idxs, scores):
	N = len(scores)
	s = scores[idxs]
	w = scores.sum() / N / s
	w_hat = w**self.k
	w_hat *= w.dot(s) / w_hat.dot(s)

	return w_hat[:, np.newaxis]	

def AdditiveSmoothingSampler(idxs, scores, xy, smooth=1.0):
	return idxs, scores + smooth, xy

def AdaptiveAdditiveSmoothingSampler(idx, scores, xy, percentage=0.5, forget=0.9):
	mu = 1.0
	mu = forget * mu + (1 - forget) * scores.mean()
	return idxs, scores + percentage * mu, xy

def ModelSampler(model, batch_size, trainset, reweighting,
				large_batch=1024, forward_batch_size=128):
	assert batch_size < large_batch:

	idxs = np.random.choice(N, large_batch)
	x, y = trainset[idxs]

	return (idxs, scores, (x,y))

def AdjustedBiasedReweightingPolicy(idxs, scores, xy, k=1.0):
	N = len(scores)
	S1 = scores[np.setdiff1d(np.arange(N), idxs)].sum()

	return np.tile([float(N), float(S1)], (len(idxs), 1))
	
def AdjustedBiasedReweighting(input_shape, k=1.0):
	s, s_hat = x

	# Compute the variables defined in the class comment
	S2 = torch.sum(s)
	S1 = s_hat[0, 1]
	N = s_hat[0, 0]

	# Compute the unbiased weights
	a2 = (S1 + S2) / N / s

	# Compute the biased weights and the scaling factor t
	a1 = torch.pow(a2, self.k)
	sT = torch.transpose(s)
	t = torch.dot(sT, a2) / torch.dot(sT, a1)

	return [a1 * t].detach()[0]	

def train(model, trainset):
	loss_vec = numpy.array([])
	batch_time = AverageMeter()
	losses = AverageMeter()
	accs = AverageMeter()

	# switch to train mode
	model.train()

	#pdb.set_trace()
	end = time.time()

	k=0.5
	smooth=0.0
	adaptive_smoothing=False
	presample=256
	large_batch=1024
	forward_batch_size=128
	N = len(trainset)

	reweighting = BiasedReweightingPolicy(k)
	train_sampler = ModelSampler(model, batch_size, trainset, reweighting)
	adaptive_smoothing_factory = AdditiveSmoothingSampler()

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
		loss_vec = numpy.append(loss_vec,
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
		#prec1 = accuracy(output.data, target.data, topk=(1,))[0]
		#losses.update(xentropy_loss_vector_sum.data[0], img.size(0))
		#accs.update(prec1[0], img.size(0))

		# if batch_idx % args.print_freq == 0:
		# 	print('Iteration: [{0}][{1}/{2}]\t'
		# 			'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
		# 			'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
		# 			'Prec@1 {accs.val:.3f} ({accs.avg:.3f})'.format(
		# 			batch_idx, n_steps, len(trainloader), batch_time=batch_time,
		# 			loss=losses, accs=accs))

		# Compute training accuracy
		_, predict = torch.max(output, 1)
		train_accuracy = (target == predict.squeeze()).float().mean()

		losses.update(xentropy_loss_vector_sum.data[0], img.size(0))
		accs.update(train_accuracy.data[0], img.size(0))

		updates += 1

	if iteration > 0:
		if args.visdom:
			plotter.plot(model_name + ' acc', 'train', updates, accs.avg)
			plotter.plot(model_name + ' loss', 'train', updates, losses.avg)

	return updates, accs.avg, losses.avg

def main():
	if args.dataset =='svhn':
		# parameters from https://arxiv.org/pdf/1605.07146.pdf
		args.lr = 0.01
		args.data_augmentation = False
		args.epochs = 160
		args.dropout = 0.4
		num_classes = 10

	if args.dataset == 'cifar10':
		num_classes = 10
	if args.dataset == 'cifar100':
		num_classes = 100

	if args.model == 'resnet18':
		deep_model = PreActResNet18(num_classes=num_classes)
	elif args.model == 'resnet34':
		deep_model = PreActResNet34(num_classes=num_classes)
	elif args.model == 'resnet50':
		deep_model = PreActResNet50(num_classes=num_classes)
	elif args.model == 'resnet101':
		deep_model = PreActResNet101(num_classes=num_classes)
	elif args.model == 'resnet152':
		deep_model = PreActResNet152(num_classes=num_classes)
	elif args.model == 'google':
		deep_model = GoogLeNet()
	elif args.model == 'wideresnet':
		if args.dataset == 'cifar10':
			deep_model = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=num_classes)
		if args.dataset == 'svhn':
			deep_model = Wide_ResNet(depth=16, widen_factor=8, dropout_rate=args.dropout, num_classes=num_classes)	
	elif args.model == 'lenet':
		deep_model = LeNet()
	elif args.model == 'fashionlenet':
		deep_model = FashionLeNet()

	deep_model.cuda()

	if args.dataset in ['cifar10', 'svhn', 'cifar100']:
		deep_model = torch.nn.DataParallel(deep_model, device_ids=range(torch.cuda.device_count()))
		cudnn.benchmark = True

	if args.dataset in ['cifar10', 'svhn', 'cifar100']:
		optimizer = optim.SGD(deep_model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              nesterov=True,
                              weight_decay=args.weight_decay)

	if args.dataset in ['mnist', 'fashionmnist']:
		optimizer = optim.SGD(deep_model.parameters(),
                              lr=learning_rate,
                              momentum=momentum)

	epoch_steps = int(ceil(len(trainset)) / args.batch_size)
	n_steps = epoch_steps * 15
	best_prec1 = 0
	updates = 0

	for i in tqdm(range(n_steps)):
		train(deep_model, trainset)