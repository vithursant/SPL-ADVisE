import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

import torchvision
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torchvision import transforms

from datasets.fashion import FASHION

import numpy as np
from utils.sampler import StratifiedSampler, SubsetSequentialSampler, ClassificationBasedSampler, SubsetSequentialSamplerSPLDML, SubsetSequentialSamplerSPLD

def load_dataset(args):
	'''
		Loads the dataset specified
	'''

	# MNIST dataset
	if args.dataset == 'mnist':
		trans_img = transforms.Compose([
				transforms.ToTensor()
			])

		print("Downloading MNIST data...")
		trainset = MNIST('./data', train=True, transform=trans_img, download=True)
		testset = MNIST('./data', train=False, transform=trans_img, download=True)

	# CIFAR-10 dataset
	if args.dataset == 'cifar10':
		# Data
		print('==> Preparing data..')
		transform_train = transforms.Compose([
		    transforms.RandomCrop(32, padding=4),
		    transforms.RandomHorizontalFlip(),
		    transforms.ToTensor(),
		    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		trainset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
		testset = CIFAR10(root='./data', train=False, transform=transform_test, download=True)

	if args.dataset == 'cifar100':
		# Data
		print('==> Preparing data..')
		transform_train = transforms.Compose([
		    transforms.RandomCrop(32, padding=4),
		    transforms.RandomHorizontalFlip(),
		    transforms.ToTensor(),
		    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		trainset = CIFAR100(root='./data', train=True, transform=transform_train, download=True)
		testset = CIFAR100(root='./data', train=False, transform=transform_test, download=True)

	if args.dataset == 'fashionmnist':

		normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     	 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

		transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.1307,), (0.3081,))])

		trainset = FASHION(root='./data', train=True, transform=transform, download=True)
		testset = FASHION(root='./data', train=False, transform=transform, download=True)

	if args.dataset == 'svhn':
		train_transform = transforms.Compose([])

		normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
										std=[x / 255.0 for x in [50.1, 50.6, 50.8]])

		train_transform.transforms.append(transforms.ToTensor())
		train_transform.transforms.append(normalize)

		trainset = SVHN(root='./data',
								split='train',
								transform=train_transform,
								download=True)



		extra_dataset = SVHN(root='./data',
										split='extra',
										transform=train_transform,
										download=True)



	    # Combine both training splits, as is common practice for SVHN

		data = np.concatenate([trainset.data, extra_dataset.data], axis=0)
		labels = np.concatenate([trainset.labels, extra_dataset.labels], axis=0)

		trainset.data = data
		trainset.labels = labels

		test_transform = transforms.Compose([transforms.ToTensor(), normalize])
		testset = SVHN(root='./data',
								split='test',
								transform=test_transform,
								download=True)

	# Self-Paced Learning Enabled
	if args.spld:
		train_idx = np.arange(len(trainset))
		#numpy.random.shuffle(train_idx)
		n_train = len(train_idx) 
		train_sampler = SubsetSequentialSamplerSPLDML(range(len(trainset)), range(len(trainset)))
		trainloader = DataLoader(trainset,
								 batch_size=args.batch_size,
								 shuffle=False,
								 num_workers=4,
								 sampler=train_sampler)

		testloader = DataLoader(testset,
								batch_size=args.batch_size,
								shuffle=True,
								num_workers=4)
	elif args.spldml:
		n_train = len(trainset)
		train_sampler = SubsetSequentialSamplerSPLDML(range(len(trainset)), range(args.batch_size))
		trainloader = DataLoader(trainset,
								 batch_size=args.batch_size,
								 shuffle=False,
								 num_workers=1,
								 sampler=train_sampler)

		testloader = DataLoader(testset,
								batch_size=args.batch_size,
								shuffle=True,
								num_workers=1)
	# Deep Metric Learning
	elif args.dml:
		n_train = len(trainset)
		train_sampler = SubsetSequentialSampler(range(len(trainset)), range(args.batch_size))
		trainloader = DataLoader(trainset,
								 batch_size=args.batch_size,
								 shuffle=False,
								 num_workers=1,
								 sampler=train_sampler)

		testloader = DataLoader(testset,
								batch_size=args.batch_size,
								shuffle=True,
								num_workers=1)
	elif args.stratified:
		n_train = len(trainset)
		labels = getattr(trainset, 'train_labels')

		if isinstance(labels, list):
			labels = torch.FloatTensor(np.array(labels))

		train_sampler = StratifiedSampler(labels, args.batch_size)
		trainloader = DataLoader(trainset,
								 batch_size=args.batch_size,
								 shuffle=False,
								 num_workers=4,
								 sampler=train_sampler)

		testloader = DataLoader(testset,
								batch_size=args.batch_size,
								shuffle=False,
								num_workers=4)
	# Random sampling
	else:
		n_train = len(trainset)
		trainloader = DataLoader(trainset,
								 batch_size=args.batch_size,
								 shuffle=True,
								 num_workers=4)

		testloader = DataLoader(testset,
								batch_size=args.batch_size,
								shuffle=True,
								num_workers=4)

	return trainloader, testloader, trainset, testset, n_train
