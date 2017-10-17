# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

__docformat__ = 'restructedtext en'

import pdb

import six.moves.cPickle as cPickle
import gzip
import os
import sys
import timeit
import datetime

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
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms

import visdom

from utils.cluster_dataset import group_data, compute_clusters, get_cluster
from utils.train_settings import parse_settings
from datasets.load_dataset import load_dataset

from models.lenet import LeNet, FashionLeNet
from models.preact_resnet import PreActResNet18
from models.wide_resnet import Wide_ResNet
from models.googlenet import GoogLeNet

from utils.sampler import SubsetSequentialSampler
from utils.min_max_sort import min_max_sort

import matplotlib.pyplot as plt

from utils.average_meter import AverageMeter
from visualizer.visualize import VisdomLinePlotter

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

import config as cf

import pdb
import random
#viz = visdom.Visdom()

args = parse_settings()
#print(args)
#args.cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
# if args.cuda:
# 	torch.cuda.manual_seed(args.seed)

if use_cuda:
	torch.cuda.manual_seed(args.seed)

if args.mnist:
	model = LeNet()
	model.cuda()
	print(model)

if args.cifar10:
	#start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	#model = PreActResNet18()
	#model = ResNetCifar10()
	#model = Wide_ResNet(args.depth, args.widen_factor, args.dropout, int(len(classes)))
	model = GoogLeNet()

	if use_cuda:
		model.cuda()
		model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
		cudnn.benchmark = True

	print(model)

if args.fashionmnist:
	model = FashionLeNet()
	model.cuda()
	print(model)

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

def extract_features(model, n_train, trainloader):
	start_index = 0
	#train_features = ()
	train_features_seq = ()
	for batch_idx, (img, target) in tqdm(enumerate(trainloader)):
		#print(img)
		#xit()
		end_index = min([start_index + args.batch_size, n_train])
		#print("End index: {}".format(end_index))
		batch_index = range(start_index, end_index)
		#print("Batch index: {}".format(batch_index))
		start_index = end_index
		#print("Start index: {}".format(start_index))
		img = Variable(img).cuda()
		_, train_features = model(img)
		train_features = train_features.data
		#print(train_features.cpu().numpy())
		#print(type(train_features.cpu().numpy()))
		train_features_seq = train_features_seq + (train_features.cpu().numpy(), )

	#print(" ")
	#print("End index: {}".format(end_index))
	#print("Batch index: {}".format(batch_index))
	#print("Start index: {}".format(start_index))

	# Generate labels, cluster centers, nn_center
	#train_features_seq = numpy.vstack(train_features_seq)

	return numpy.vstack(train_features_seq)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR divided by 5 after 60, 120 and 160 epochs"""
    # lr = args.lr * ((0.5 ** int(epoch > 60)) * (0.5 ** int(epoch > 120))* (0.5 ** int(epoch > 160)))
    lr = lr * (0.92 ** int(epoch % 3 == 0))
    print(lr)
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_spl(trainloader, trainset, testloader, n_train):
	learning_rate = args.lr
	loss_weight = args.loss_weight
	curriculum_rate = args.curriculum_rate
	momentum = numpy.float64(args.momentum)
	decay_after_epochs = args.decay_after_epochs
	tsne_dim = 10

	epoch_steps = int(ceil(len(trainset)) / args.batch_size)
	n_steps = epoch_steps * 15
	#pdb.set_trace()

	global plotter
	plotter = VisdomLinePlotter(env_name=args.name)

	if args.spl:
		print("Start SPLD for LeNet")
		spld_params = [5000, 1e-3, 5e-2, 1e-1]

		prev_mean_all_loss = float('inf')

		criterion = nn.CrossEntropyLoss(size_average=False)

		if args.cifar10:
			optimizer = optim.SGD(model.parameters(),
	                              lr=learning_rate,
	                              momentum=momentum,
	                              weight_decay=args.weight_decay)
			spld_params = [50, 4e-1, 1e-1, 1e-1]

		if args.mnist:
			optimizer = optim.SGD(model.parameters(),
	                              lr=learning_rate,
	                              momentum=momentum)		


		# train_features_seq = extract_features(model,
		# 									  n_train,
		# 									  trainloader)

		# labels_, cluster_centers, nn_center = group_data(train_features_seq,
		# 												 tsne_dim,
		# 												 args.num_cluster,
		# 												 'mnist',
		# 												 save_file=False)

		loss_vec = numpy.array([])
		losses = AverageMeter()
		accs = AverageMeter()
		batch_losses = []

		losses_test = AverageMeter()
		accs_test = AverageMeter()


		#batch_example_inds, batch_class_inds = batch_builder.gen_batch()
		#trainloader.sampler.batch_indices = batch_example_inds
		#pdb.set_trace()

		_ = model.train()
		print("hello")
		for curriculum_epoch in tqdm(range(args.curriculum_epochs)):
			running_loss = 0.
			running_acc = 0
			start_idx = 0

			losses = AverageMeter()
			accs = AverageMeter()

			if curriculum_epoch == 0:
				print('Generating cluster classification on training data ...')
				#train_results = compute_clusters(trainloader, model, len(range(args.num_classes)), args.feature_size)

				#print(train_idx)
				print("First curriculum epoch")

				train_features_seq = extract_features(model,
                                                      n_train,
                                                      trainloader)

				labels_, cluster_centers, nn_center = group_data(train_features_seq,
                                                                 10,
                                                                 args.num_cluster,
                                                                 'mnist',
                                                                 save_file=False)

				#labels_, labels_weight, cluster_centers, nn_center = get_cluster(train_features_seq,
				#																 100,
				#																 args.num_cluster,
				#																 'mnist',
				#																 save_file=False)
				#labels_ = labels_.astype('int32')
				#nn_center = nn_center.astype('int32')
				#print(labels_weight)
				#exit()
				#plot_kmeans_hist(labels_, bins=args.num_cluster)
				#print(trainloader)
				#exit()

			loss_vec = numpy.array([])
			losses = AverageMeter()
			accs = AverageMeter()

			for batch_idx, (img, target) in enumerate(trainloader):
				#print("Batch idx: {}".format(batch_idx))
				img = Variable(img).cuda()
				target = Variable(target).cuda()

				optimizer.zero_grad()
				output, _ = model(img)

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

				losses.update(xentropy_loss_vector_sum.data[0], img.size(0))

				# Backward
				xentropy_loss_vector_sum.backward()
				optimizer.step()
				#print(xentropy_loss_vector_sum)

				# Compute training accuracy
				_, predict = torch.max(output, 1)

				train_accuracy = (target == predict.squeeze()).float().mean()
				accs.update(train_accuracy.data[0], img.size(0))

				if batch_idx % args.log_interval == 0:
					print('Train Epoch: {} [{}/{}]\t'
							'Loss: {:.4f} ({:.4f}) \t'
							'Acc: {:.2f}% ({:.2f}%) \t'.format(curriculum_epoch+1,
                                                                batch_idx * len(img),
                                                                len(trainset),
                                                                losses.val, losses.avg,
                                                                100. * accs.val,
                                                                100. * accs.avg))


			# log avg values to somewhere
			if args.visdom:
				plotter.plot('acc', 'train', curriculum_epoch+1, accs.avg)
				plotter.plot('loss', 'train', curriculum_epoch+1, losses.avg)

			# Compute the mean loss
			#print(len(loss_vec))
			curr_mean_all_loss = numpy.mean(loss_vec)
			#print('mean training loss = ', curr_mean_all_loss)


			# Self-paced learning with diversity sampling
			# Selecting samples using the self-paced learning with diversity
			# update clustering and submodular function F
			#if epoch % args.fea_freq == 0:

			train_idx = numpy.array([])

			print("Start self-paced learning with diversity")
			#select sample: self-paced learning with diversity
			#print(loss_vec)
			#print(len(loss_vec))
			train_idx = numpy.array([])
			#print(len(loss_vec))
			#for i in loss_vec:
			#	print(i)
			#exit()

			for i in range(args.num_cluster):
				i_cluster = numpy.where(labels_ == i)[0]
				#pdb.set_trace()
				#print(i_cluster)
				#exit()
				iloss_vec = loss_vec[i_cluster]
				#pdb.set_trace()
				#print(iloss_vec)
				sortIndex = numpy.argsort(iloss_vec)
				#print(sortIndex)
				#print(iloss_vec[sortIndex])
				#exit()
				#print(iloss_vec)
				#iloss_vec = min_max_sort(iloss_vec, len(iloss_vec))
				#print(iloss_vec)
				#exit()

				# iloss_vec -= spld_params[1] * \
				# 						numpy.divide(1,
				# 									numpy.sqrt(range(1, 1+len(i_cluster)))
                #                         + numpy.sqrt(range(len(i_cluster))))

				iloss_vec[sortIndex] -= spld_params[1] * \
										numpy.divide(1,
													numpy.sqrt(range(1, 1+len(i_cluster)))
                                            + numpy.sqrt(range(len(i_cluster))))


				#exit()
				K = min([spld_params[0], len(iloss_vec)-1])
				train_idx = numpy.append(train_idx,
                                         i_cluster[numpy.argpartition(iloss_vec, K)[:K]])

			train_idx = train_idx.astype('int32')

			# Increase the learning pace
			spld_params[0] *= (1 + spld_params[2])
			spld_params[0] = int(round(spld_params[0]))
			spld_params[1] *= (1 + spld_params[3])
			#print(train_idx)
			#print(len(train_idx))

			#decay_after_epochs = 3
			#if (curriculum_epoch + 1) % decay_after_epochs == 0:
			#print(prev_mean_all_loss)

			'''
				TODO: update learning rate and learning pace
				TODO: Update clustering
			'''
			'''
			if curr_mean_all_loss > 1.005 * prev_mean_all_loss:
				learning_rate *= 0.95
				optimizer.param_groups[0]['lr'] = learning_rate
				momentum_update = (1.0 - (1.0 - momentum) * 0.95).clip(max=0.9999)
				optimizer.param_groups[0]['momentum'] = momentum_update
			'''
        	# update learning rate and learning pace
			# if curriculum_epoch % args.lr_freq == 0:
			# 	adjust_learning_rate(optimizer, curriculum_epoch, learning_rate)
            	#loss_weight *= curriculum_rate + 1
            	#print ('loss_weight, num_learner_per_cluster', args.loss_weight, args.num_learner_per_cluster)

			# update clustering
			#if curriculum_epoch % args.fea_freq == 0:
				 # clustering
			#	train_features_seq = extract_features(model,
            #                                          n_train,
            #                                          trainloader)
			#	labels_, cluster_centers, nn_center = group_data(train_features_seq,
            #                                                     10,
            #                                                     args.num_cluster,
            #                                                     'mnist',
            #                                                     save_file=False)

			print("Finished SPLD")

			#print(prev_mean_all_loss)
			#if curr_mean_all_loss > 1.005 * prev_mean_all_loss:
			#	learning_rate *= 0.96
			#	print(prev_mean_all_loss)
			#	print(curr_mean_all_loss)
			#	print(learning_rate)
			#	optimizer.param_groups[0]['lr'] = learning_rate

			prev_mean_all_loss = curr_mean_all_loss

			#print(len(trainloader.sampler.indices))
			#trainloader.sampler.indices = list(range(0, 60000))
			trainloader.sampler.batch_indices = train_idx

			# Testing
			model.eval()
			losses_test = AverageMeter()
			accs_test = AverageMeter()

			for batch_idx, (img, target) in enumerate(testloader):
				img = Variable(img).cuda()
				target = Variable(target).cuda()

				score, _ = model(img)

				# Compute the loss for each sample in the minibatch
				onehot_target = Variable(encode_onehot(target, args.num_classes))
				xentropy_loss_vector = -1 * torch.sum(torch.log(F.softmax(score))
	                                                    * onehot_target,
	                                                  dim=1,
	                                                  keepdim=True)

				# Sum minibatch loss vector
				loss_vec = numpy.append(loss_vec,
										xentropy_loss_vector.data.cpu().numpy())

				xentropy_loss_vector_sum = xentropy_loss_vector.sum()
				xentropy_loss_vector_mean = xentropy_loss_vector.mean()

				_, predict = torch.max(score, 1)
				test_accuracy = (target == predict.squeeze()).float().mean()

				accs_test.update(test_accuracy.data[0], img.size(0))
				losses_test.update(xentropy_loss_vector_sum.data[0], img.size(0))

			# log avg values to somewhere
			print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(losses_test.avg, 100. * accs_test.avg))
			if args.visdom:
				plotter.plot('acc', 'test', curriculum_epoch+1, accs_test.avg)
				plotter.plot('loss', 'test', curriculum_epoch+1, losses_test.avg)

	torch.save(model.state_dict(), model.name())

def train_random(trainloader, trainset, testloader, n_train):
	learning_rate = args.lr
	loss_weight = args.loss_weight
	curriculum_rate = args.curriculum_rate
	momentum = numpy.float64(args.momentum)
	decay_after_epochs = args.decay_after_epochs
	tsne_dim = 10

	epoch_steps = int(ceil(len(trainset)) / args.batch_size)
	n_steps = epoch_steps * 15
	#pdb.set_trace()

	global plotter
	plotter = VisdomLinePlotter(env_name=args.name)

	if args.mnist:
		model = LeNet()
		print(model)

	if args.cifar10:
		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		#model = PreActResNet18()
		model = ResNetCifar10()
		print(model)

	if args.fashionmnist:
		model = FashionLeNet()
		print(model)


	model.cuda()

	criterion = nn.CrossEntropyLoss(size_average=False)
	optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum)

	losses = AverageMeter()
	accs = AverageMeter()
	batch_losses = []

	losses_test = AverageMeter()
	accs_test = AverageMeter()

	for epoch in range(args.epochs):
		#running_loss = 0.
		#running_acc = 0.
		#losses = AverageMeter()
		#accs = AverageMeter()
		
		model.train()
		for batch_idx, (img, target) in tqdm(enumerate(trainloader)):
			img = Variable(img).cuda()
			target = Variable(target).cuda()

			optimizer.zero_grad()
			output, _ = model(img)

			loss = criterion(output, target)

			#loss_mean = loss.mean()

			losses.update(loss.data[0], img.size(0))

			loss.backward()
			optimizer.step()

			_, predict = torch.max(output, 1)
			train_accuracy = (target == predict.squeeze()).float().mean()

			accs.update(train_accuracy.data[0], img.size(0))

			if batch_idx % args.log_interval == 0:
				print('Train Epoch: {} [{}/{}]\t'
						'Loss: {:.4f} ({:.4f}) \t'
						'Acc: {:.2f}% ({:.2f}%) \t'.format(epoch+1,
                                                           batch_idx * len(img),
                                                           len(trainset),
                                                           losses.val, losses.avg,
                                                           100. * accs.val,
                                                           100. * accs.avg))

		# log avg values to somewhere
		if args.visdom:
			plotter.plot('acc', 'train', epoch+1, accs.avg)
			plotter.plot('loss', 'train', epoch+1, losses.avg)

		# Testing
		model.eval()

		for batch_idx, (img, target) in enumerate(testloader):
			# volatile=True is used in inference mode. it makes stopping histoy recording for backward().
			img = Variable(img, volatile=True).cuda()
			target = Variable(target).cuda()

			score, _ = model(img)

			loss = criterion(score, target)
			#loss_mean = loss.mean()

			_, predict = torch.max(score, 1)

			test_accuracy = (target == predict.squeeze()).float().mean()

			accs_test.update(test_accuracy.data[0], img.size(0))
			losses_test.update(loss.data[0], img.size(0))

		# log avg values to somewhere
		print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(losses_test.avg, 100. * accs_test.avg))
		if args.visdom:
			plotter.plot('acc', 'test', epoch+1, accs_test.avg)
			plotter.plot('loss', 'test', epoch+1, losses_test.avg)

	torch.save(model.state_dict(), model.name())

if __name__ == '__main__':
	"""
	TODO:
		Train on a larger dataset, MNIST Fashion, tiny imagenet
		Fix KMeans
		Try different CNN model
		Ensemble curriculum learning
		Integrate Magnet Loss
	"""
	trainloader, testloader, trainset, testset, n_train = load_dataset(args)
	n_train_batches = len(trainset) // args.batch_size
	n_test_batches =  len(testset) // args.batch_size

	print("==>>> Total Training Batch Number: {}".format(len(trainset)))
	print("==>>> Total Number of Train Batches: {}".format(n_train_batches))
	print("==>>> Total Number of Test Batches: {}".format(n_test_batches))

	print("Starting Training...")
	if args.spl:
		train_spl(trainloader, trainset, testloader, n_train)
	else:
		train_random(trainloader, trainset, testloader, n_train)