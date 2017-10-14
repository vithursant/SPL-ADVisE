import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from tqdm import tqdm
import pdb

from string import ascii_lowercase
from random import choice

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.init as init

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

def spl(losses, lamda=0.15):
	selected_idx = [i for i, loss in enumerate(losses) if loss < lamda] #all(loss < lambda for x in losses)
	#pdb.set_trace()
	sortedselected_idx = np.sort(np.take(losses, selected_idx))
	#pdb.set_trace()
	return sortedselected_idx

def spld(losses, groupmembership, lamda=0.01, gamma=0.3):
	groupidx = np.unique(groupmembership)
	n_clusters = len(groupidx)

	selected_idx = np.zeros(len(losses))
	selected_scores = np.zeros(len(losses))

	for j in range(n_clusters):
		idx_ingroup = np.where(groupmembership == groupidx[j])[0]
		loss_ingroup = np.take(loss, idx_ingroup)
		rank_ingroup = np.sort(np.argsort(loss_ingroup)) + 1
		pdb.set_trace()
		for i in range(len(idx_ingroup)):
			#pdb.set_trace()
			if(loss_ingroup[i] < lamda + gamma / (np.sqrt(rank_ingroup[i]) + np.sqrt(rank_ingroup[i]-1))):
				selected_idx[idx_ingroup[i]] = 1
			else:
				selected_idx[idx_ingroup][i] = 0
			#pdb.set_trace()
			selected_scores[idx_ingroup[i]] = loss_ingroup[i] - lamda - gamma/(np.sqrt(rank_ingroup[i])+np.sqrt(rank_ingroup[i]-1))
			#pdb.set_trace()
	#pdb.set_trace()
	selected_idx = np.where(selected_idx == 1)
	#pdb.set_trace()
	sortedselected_idx = np.take(selected_scores, selected_idx)
	#pdb.set_trace()

	return selected_idx, sortedselected_idx

if __name__ == '__main__':
	letters = [choice(ascii_lowercase) for _ in range(14)]
	groupmembership = [1,1,1,1,1,1,2,2,2,3,3,3,3,4]
	n_clusters = 4

	groups = init_list_of_objects(n_clusters)
	for i in range(n_clusters):
		for j, letter in enumerate(letters):
			if groupmembership[j] == i+1:
				groups[i].append(letter)

	loss = [0.05, 0.12, 0.12, 0.12, 0.15, 0.40, 0.17, 0.18, 0.35, 0.15, 0.16, 0.20, 0.50, 0.28]
	
	print("When lambda=0.15, SPL selects:")
	print(spl(loss, lamda=0.15))

	print("When lambda=0.03 and gamma=0.2, SPLD selects:")
	print(spld(loss, groupmembership, lamda=0.15, gamma = 0.15))

	print("When lambda=0.03 and gamma=0.2, SPLD selects:")
	print(spld(loss, groupmembership, lamda=0.10, gamma = 0.185))

	print("When lambda=0.03 and gamma=0.2, SPLD selects:")
	print(spld(loss, groupmembership, lamda=0.05, gamma = 0.2))

	print("When lambda=0.0 and gamma=0.2, SPLD selects:")
	print(spld(loss, groupmembership, lamda=0.00, gamma = 0.285))
	pdb.set_trace()
	'''
	loss <- c(0.05,0.12,0.12,0.12,0.15,0.40,0.17,0.18,0.35,0.15,0.16,0.20,0.50,0.28)

	print(paste("When lambda=0.15, SPL selects:"))
	print(vid[spl(loss, lambda=0.15)])

	print(paste("When lambda=0.03 and gamma=0.2, SPLD selects:"))
	print(vid[spld(loss, groupmembership, lambda=0.05, gamma = 0.2)])

	print(paste("When lambda=0.0 and gamma=0.2, SPLD selects:"))
	vid[spld(loss, groupmembership, lambda=0.00, gamma = 0.285))
	'''

	# TODO: IDEA OF MINIBATCHES DISMISSED, instead feed samples in terms of loss and diversity
	# COPY this over to the gen_batch_spld function and it can create batches from there on
	# create batches based on the spld algorithm, where its based on the easiness and the loss
	# then once we have that we can take samples sequentially and put into minibatches
	# make sure the samples that are selected is divisible by the minibatch number