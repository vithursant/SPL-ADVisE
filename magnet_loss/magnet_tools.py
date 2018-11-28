# Copyright 2018 Vithursan Thangarasa.
#
# This file is part of SPL-ADVisE.
#
# SPL-ADVisE is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# SPL-ADVisE is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# SPL-ADVisE. If not, see <http://www.gnu.org/licenses/>.

from math import ceil
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb

def compute_reps(model, X, chunk_size, device):
    """Compute representations for input in chunks."""
    chunks = int(ceil(float(len(X)) / chunk_size))
    #print(chunks)
    reps = []
    labels = []

    trainloader = DataLoader(X,
                         batch_size=chunks,
                         shuffle=False,
                         num_workers=4)

    for batch_idx, (img, target) in tqdm(enumerate(trainloader)):
        img = Variable(img).to(device)
        #pdb.set_trace()
        output, train_features = model(img)
        #train_features = model(img)
        embeddings = output.data

        reps.append(embeddings.cpu().numpy())
        #labels.append(target.cpu().numpy())
    return np.vstack(reps)

class ClusterBatchBuilder(object):
    """Sample minibatches for magnet loss."""
    def __init__(self, labels, k, m, d):
        #pdb.set_trace()
        if isinstance(labels, np.ndarray):
            self.num_classes = np.unique(labels).shape[0]
            self.labels = labels
        elif isinstance(labels, list):
            self.num_classes = np.unique(labels).shape[0]
            self.labels = np.array(labels)
        else:
            #pdb.set_trace()
            self.num_classes = np.unique(labels.numpy()).shape[0]
            self.labels = labels.numpy()

        self.k = k # Class labels for each example
        self.m = m # The number of clusters in the batch.
        self.d = d # The number of examples in each cluster

        self.centroids = None

        if isinstance(labels, np.ndarray):
            self.assignments = np.zeros_like(labels, int)
        elif isinstance(labels, list):
            self.assignments = np.zeros_like(np.array(labels), int)
        else:
            self.assignments = np.zeros_like(labels.numpy(), int)

        self.cluster_assignments = {}
        self.cluster_classes = np.repeat(range(self.num_classes), k)
        self.magnet_example_losses = None
        self.spld_example_losses = None
        self.cluster_losses = None
        self.has_loss = None
        #pdb.set_trace()


    def update_clusters(self, rep_data, max_iter=20):
        """
        Given an array of representations for the entire training set,
        recompute clusters and store example cluster assignments in a
        quickly sampleable form.
        """

        # Lazily allocate array of zeroes as placeholders for centroids
        if self.centroids is None:
            #pdb.set_trace()
            self.centroids = np.zeros([self.num_classes * self.k, rep_data.shape[1]])

        #pdb.set_trace()

        for class_idx in range(self.num_classes):

            class_mask = self.labels == class_idx # Boolean mask for selecting examples
            #pdb.set_trace()
            class_examples = rep_data[class_mask] # Mask features based on the class mask

            kmeans = KMeans(n_clusters=self.k, init='k-means++', n_init=10, max_iter=max_iter)
            kmeans.fit(class_examples)
            #pdb.set_trace()

            # Save cluster centroids for finding impostor clusters
            start = self.get_cluster_ind(class_idx, 0)
            stop = self.get_cluster_ind(class_idx, self.k)
            self.centroids[start:stop] = kmeans.cluster_centers_ # Should create self.k * self.num_classes number of centroids
            #pdb.set_trace()

            # Update assignments with new global cluster indexes
            self.assignments[class_mask] = self.get_cluster_ind(class_idx, kmeans.predict(class_examples))
            #pdb.set_trace()

        # Construct a map from cluster to example indexes for fast batch creation
        for cluster in range(self.k * self.num_classes):
            cluster_mask = self.assignments == cluster
            # np.flatnonzero return indices that are non-zero in the flattened version of cluster_mask
            self.cluster_assignments[cluster] = np.flatnonzero(cluster_mask)    # Example indexes mapped to the cluster
            #pdb.set_trace()
        #pdb.set_trace()

    def update_losses(self, indexes, losses, loss_type):
        """
        Given a list of examples indexes and corresponding losses
        store the new losses and update corresponding cluster losses.
        """
        # Lazily allocate structures for losses
        if self.magnet_example_losses is None or self.spld_example_losses is None:
            self.magnet_example_losses = np.zeros_like(self.labels, float)
            self.spld_example_losses = np.zeros_like(self.labels, float)
            self.cluster_losses = np.zeros([self.k * self.num_classes], float)

            self.has_loss = np.zeros_like(self.labels, bool)
            self.spld_cluster_ex_losses = [None] * (self.k * self.num_classes)

        # Update example losses
        if loss_type == "magnet":
            magnet_losses = losses.data.cpu().numpy()
            self.magnet_example_losses[indexes] = magnet_losses
            self.has_loss[indexes] = magnet_losses

        if loss_type == "spld":
            spld_losses = losses.data.cpu().numpy()
            self.spld_example_losses[indexes] = spld_losses

        # Find affected clusters and update the corresponding cluster losses
        # Gets the clusters in which the lossses belong to
        clusters = np.unique(self.assignments[indexes])
        #pdb.set_trace()
        for cluster in clusters:
            cluster_inds_mask = self.assignments == cluster

            if loss_type == "magnet":
                cluster_example_losses = self.magnet_example_losses[cluster_inds_mask]

            if loss_type == "spld":
                cluster_example_losses = self.spld_example_losses[cluster_inds_mask]
                self.spld_cluster_ex_losses[cluster] = cluster_example_losses

            if loss_type == "magnet":
                # Take the average closs in the cluster of examples for which we have measured a loss
                # Array of clusters with their average example loss
                self.cluster_losses[cluster] = np.mean(cluster_example_losses[self.has_loss[cluster_inds_mask]]) # gets the losses within the cluster and calculates the mean

    def gen_batch(self):
        """
        Sample a batch by first sampling a seed cluster proportionally to
        the mean loss of the clusters, then finding nearest neighbor
        "impostor" clusters, then sampling d examples uniformly from each cluster.

        The generated batch will consist of m clusters each with d consecutive
        examples.
        """

        # Sample seed cluster proportionally to cluster losses if available
        if self.cluster_losses is not None:
            p = self.cluster_losses / np.sum(self.cluster_losses)
            seed_cluster = np.random.choice(self.num_classes * self.k, p=p)
        else:
            seed_cluster = np.random.choice(self.num_classes * self.k)

        #pdb.set_trace()

        # Get imposter clusters by ranking centroids by distance
        sq_dists = ((self.centroids[seed_cluster] - self.centroids) ** 2).sum(axis=1)

        #pdb.set_trace()

        # Assure only clusters of different class from seed are chosen
        sq_dists[self.get_class_ind(seed_cluster) == self.cluster_classes] = np.inf

        #pdb.set_trace()

        # Get top impostor clusters and add seed

        # Returns an array of indices of the same shape as a that index data along the given axis in partitioned order.
        clusters = np.argpartition(sq_dists, self.m-1)[:self.m-1]
        clusters = np.concatenate([[seed_cluster], clusters])

        #pdb.set_trace()

        # Sample examples uniformly from cluster
        batch_indexes = np.empty([self.m * self.d], int)
        #pdb.set_trace()
        for i, c in enumerate(clusters):
            #pdb.set_trace()
            # Go through each cluster and select random samples that are size self.d
            x = np.random.choice(self.cluster_assignments[c], self.d, replace=True)
            start = i * self.d
            stop = start + self.d
            batch_indexes[start:stop] = x
            #pdb.set_trace()
        #pdb.set_trace()

        # Translate class indexes to index for classes within the batch
        class_inds = self.get_class_ind(clusters)
        batch_class_inds = []
        inds_map = {}
        class_count = 0
        for c in class_inds:
            if c not in inds_map:
                inds_map[c] = class_count
                class_count += 1
            batch_class_inds.append(inds_map[c])
            #pdb.set_trace()
        #pdb.set_trace()
        return batch_indexes, np.repeat(batch_class_inds, self.d)

    def gen_batch_spl(self, max_pos_len, diversity_ratio, batch_size):
        """
        Generate batch from DML for the SPL framework
        """

        # Sample examples using SPLD
        selected_samples_idx = np.zeros(len(self.labels), int)
        selected_scores = np.zeros(len(self.labels), float)

        for cluster in range(self.num_classes*self.k):
            idx_incluster = self.cluster_assignments[cluster]
            loss_incluster = np.sort(self.spld_cluster_ex_losses[cluster])
            rank_incluster = np.sort(np.argsort(loss_incluster))+1

            sort_idx = np.argsort(loss_incluster)
            #total_count += 1
            loss_incluster[sort_idx] -= diversity_ratio * 1 / (np.sqrt(rank_incluster) + np.sqrt(rank_incluster-1))

            K = min([max_pos_len, len(loss_incluster)])
            #selected_samples_idx[idx_incluster[np.argpartition(loss_incluster, K)[:K]]] = 1
            selected_samples_idx[idx_incluster[:K]] = 1
            #selected_samples_idx[idx_incluster[:len(loss_incluster)-1]] = 1
            #m = torch.distributions.Normal(torch.Tensor(diversity_ratio * 1 / (np.sqrt(rank_incluster) + np.sqrt(rank_incluster-1))), torch.Tensor([1.0]))
            #pdb.set_trace()
        selected_samples_idx = np.where(selected_samples_idx == 1)

        #left_over = len(selected_samples_idx[0]) % batch_size

        # if left_over > 0:
        #     selected_samples_idx = np.append(selected_samples_idx[0], np.random.choice(range(len(self.labels)), batch_size - left_over, replace=False))
        # else:
        #     selected_samples_idx = selected_samples_idx[0]
            #grouploss_threshold = np.vstack((loss_incluster[sort_idx], grouped_threshold))
            #pdb.set_trace()

            #loss_incluster[sort_idx] -= gamma_param / ((np.sqrt(range(1, 1+len(idx_incluser))) + np.sqrt(range(len(idx_incluser)))))
            #pdb.set_trace()
            #for i in range(1, len(loss_incluster)):
            #    loss_incluster[i] - lambda_param - gamma_param/(np.sqrt(rank_incluster[i])+np.sqrt(rank_incluster[i]-1))
            # for i in range(1, len(loss_incluster)):
            #     #print(len(idx_incluster))
            #     #print(len(loss_incluster))
            #     if(loss_incluster[i] < min(self.losses) + lambda_param / ((np.sqrt(rank_incluster[i]) + np.sqrt(rank_incluster[i]-1)))):
            #         pos_count += 1
            #         selected_samples_idx[idx_incluster[i]] = 1
            #     else:
            #         neg_count += 1
            #         selected_samples_idx[idx_incluster[i]] = 0
            #selected_scores[idx_incluster[i]] = loss_incluster[i] - lambda_param - gamma_param /(np.sqrt(rank_incluster[i])+np.sqrt(rank_incluster[i]-1))
            #pdb.set_trace()
        #selected_samples_idx = np.where(selected_samples_idx == 1)
        #sortedselected_idx = np.take(selected_scores, selected_samples_idx)

        #metric = grouploss_threshold[0] - 0.01 * grouploss_threshold[1]
        #pdb.set_trace()

        # selected_samples_idx = np.zeros(len(self.labels), int)
        # selected_scores = np.zeros(len(self.labels), float)

        # pos_count = 0
        # neg_count = 0
        # total_count = 0
        # for j, cluster in enumerate(clusters):
        #     # TODO make sure idx_incluster and loss_incluster lengths are the same
        #     idx_incluster = self.cluster_assignments[cluster]
        #     #loss_incluster = (self.cluster_ex_losses[cluster] - np.max(self.cluster_ex_losses[cluster]))/-np.ptp(self.cluster_ex_losses[cluster])
        #     loss_incluster = np.sort(self.cluster_ex_losses[cluster])
        #     loss_incluster = (loss_incluster - np.max(loss_incluster))/-np.ptp(loss_incluster)
        #     rank_incluster = np.sort(np.argsort(loss_incluster)) + 1
        #     #pdb.set_trace()

        #     for i in range(len(loss_incluster)):
        #         #pdb.set_trace()
        #         total_count += 1
        #         if(loss_incluster[i] < lambda_param + gamma_param / ((np.sqrt(rank_incluster[i]) + np.sqrt(rank_incluster[i]-1)))):
        #             pos_count += 1
        #             selected_samples_idx[idx_incluster[i]] = 1
        #         else:
        #             neg_count += 1
        #             selected_samples_idx[idx_incluster[i]] = 0
        #     selected_scores[idx_incluster[i]] = loss_incluster[i] - lambda_param - gamma_param /(np.sqrt(rank_incluster[i])+np.sqrt(rank_incluster[i]-1))
        #     #pdb.set_trace()
        # selected_samples_idx = np.where(selected_samples_idx == 1)
        # sortedselected_idx = np.take(selected_scores, selected_samples_idx)
        # #pdb.set_trace()
        return selected_samples_idx[0]

    def get_cluster_ind(self, c, i):
        """
        Given a class index and a cluster index within the class
        return the global cluster index
        """
        return c * self.k + i

    def get_class_ind(self, c):
        """
        Given a cluster index return the class index.
        """
        return c / self.k
