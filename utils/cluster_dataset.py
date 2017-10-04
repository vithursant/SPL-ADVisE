from __future__ import print_function
from __future__ import division

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances

from scipy.sparse import issparse

from tqdm import tqdm

def compute_clusters(test_loader, network, num_clusters, feature_size):
    network.eval()
    embeddings = np.zeros(shape=(len(test_loader.dataset), feature_size),
                          dtype=float)
    labels_true = np.zeros(shape=(len(test_loader.dataset)), dtype=int)
    for batch_idx, (data, classes, ids) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data)

        # compute embeddings
        f = network(data)
        embeddings[ids.numpy(),:] = f.cpu().data.numpy()
        labels_true[ids.numpy()] = classes.cpu().numpy()
    print('Generated embeddings, now running k-means for %d clusters...' % num_clusters)

    # initialize centroid  for each cluster
    unique_classes = np.unique(labels_true)
    num_classes = len(unique_classes)
    initial_centers = np.zeros(shape=(num_clusters, feature_size), dtype=float)
    for i in range(num_classes):
        c_ids = np.where(labels_true == unique_classes[i])
        use_im = np.random.choice(c_ids[0])
        initial_centers[i,:] = embeddings[use_im,:]

    kmeans_model = KMeans(n_clusters=num_clusters, random_state=1,
                          max_iter=1000, tol=1e-3,
                          init=initial_centers, n_init=1)
    labels_predicted = kmeans_model.fit_predict(embeddings)

    # map predicted clusters to actual class ids
    cluster_to_class = np.zeros(shape=(num_classes,), dtype=int)
    for i in range(num_classes):
        # figure out which class this cluster must be
        # set of points that belong to this cluster
        cluster_points = np.where(labels_predicted == i)
        # true class labels for these points
        actual_labels = labels_true[cluster_points]
        # map cluster to most frequently occuring class
        unique, indices = np.unique(actual_labels, return_inverse=True)
        mode = unique[np.argmax(
            np.apply_along_axis(
                np.bincount, 0, indices.reshape(actual_labels.shape),
                None, np.max(indices) + 1), axis=0)]
        cluster_to_class[i] = mode

    # map cluster id to class ids
    labels_copy = np.copy(labels_predicted)
    for i in range(num_classes):
        cluster_points = np.where(labels_copy == i)
        labels_predicted[cluster_points] = cluster_to_class[i]
    
    #print('Labels true')
    #print(labels_true)
    #print('Labels predicted')
    #print(labels_predicted)

    acc = metrics.accuracy_score(labels_true, labels_predicted)
    nmi_s = metrics.cluster.normalized_mutual_info_score(
              labels_true, labels_predicted)
    mi = metrics.cluster.mutual_info_score(
            labels_true, labels_predicted
            )
    h1 = metrics.cluster.entropy(labels_true)
    h2 = metrics.cluster.entropy(labels_predicted)
    nmi = 2*mi/(h1+h2)
    print(mi, h1, h2)
    precision = metrics.precision_score(labels_true, labels_predicted,
                                       average='micro')
    recall = metrics.recall_score(labels_true, labels_predicted,
                                  average='micro')
    f1_score = 2*precision*recall/(precision+recall)

    print('Accuracy : %f' % acc)
    print('NMI : %f vs old %f' % (nmi, nmi_s))
    print('Precision : %f' % precision)
    print('Recall : %f' % recall)
    print('F1 score : %f' % f1_score)
    print("")
    results = {
                'true' : labels_true,
                'predicted' : labels_predicted,
                'accuracy' : acc,
                'precision' : precision,
                'recall' : recall,
                'f1' : f1_score,
                'nmi' : nmi
              }
    return  results

def get_cluster(X, pca_dim, num_cluster, dataset_name, save_file = True, topk = 1):

    n = X.shape[0]
    center_nn = np.array([])
    centers_ = ()

    # dimension reduction
    if issparse(X):
        print ('TruncatedSVD of sparse X', (n, X.shape[1]))
        svd = TruncatedSVD(n_components=pca_dim, algorithm='randomized', n_iter=15)
        X_pca = svd.fit_transform(X)
        print ('TruncatedSVD finished')
    elif n > 10000:
        print ('PCA of data size', n)
        pca = PCA(n_components = pca_dim, svd_solver='randomized')
        X_pca = pca.fit_transform(X)
        print ('PCA finished')
    else:
        X_pca = X
        print ('PCA not applied')

    # clustering
    print ('k-means to', num_cluster, 'clusters')
    kmeans = MiniBatchKMeans(n_clusters = num_cluster, max_iter = 100, init_size = 3*num_cluster).fit(X_pca.astype('float64'))    
    labels_ = kmeans.labels_.astype('int32')
    labels_ = np.array([np.where(labels_ == i)[0].astype('int32') for i in range(num_cluster)])
    labels_weight = np.asarray(list(map(len, labels_)))
    #print(labels_weight)
    labels_weight = np.divide(labels_weight,float(np.max(labels_weight)))
    nnz_ind = np.where(labels_weight != 0)[0]
    labels_ = labels_[nnz_ind]
    labels_weight = labels_weight[nnz_ind]
    
    for j in range(len(nnz_ind)):
        centers_ = centers_ + (np.mean(X[labels_[j], :], axis = 0),)
        center_nn = np.append(center_nn, labels_[j][np.argmin(euclidean_distances([kmeans.cluster_centers_[nnz_ind[j]]], X_pca[labels_[j]]))])
    centers_ = np.vstack(centers_)

    if save_file:
        np.savetxt(dataset_name + '_kmeans_labels.txt', cluster_label)
        np.savetxt(dataset_name + '_kmeans_centers.txt', cluster_centers)
        np.savetxt(dataset_name + '_center_nn.txt', center_nn)
        labels_, labels_weight, centers_, center_nn = [],[],[],[]
    else:
        return labels_, labels_weight, centers_, center_nn.astype('int32')

def group_data(X, tsne_dim, num_cluster, dataset_name, save_file = True):

    print('clustering...')
    n = X.shape[0]
    center_nn = np.array([])
    cluster_centers = ()
    if tsne_dim <= 0 and not issparse(X) and n <= 10000:
        X_tsne = X
    elif    n > 10000:
        if tsne_dim == 0:
            tsne_dim = 48
        print('TruncatedSVD of data size', (n, X.shape[1]))
        svd = TruncatedSVD(n_components=tsne_dim, algorithm='randomized', n_iter=10, random_state=42)
        X_tsne = svd.fit_transform(X)
        #print(len(X_tsne))
        print('finish TruncatedSVD.')
    else:
        print('PCA of data size', n)
        pca = PCA(n_components = tsne_dim)
        X_tsne = pca.fit_transform(X)
        print('finish PCA.')
    print('k-means to', num_cluster, 'clusters')
    #reduced_data = X_tsne.astype('float64')
    kmeans = KMeans(n_clusters = num_cluster, max_iter = 50, n_jobs=4).fit(X_tsne.astype('float64'))
    cluster_label = kmeans.labels_
    for j in range(num_cluster):
        jIndex = np.where(cluster_label==j)[0]
        centerj = kmeans.cluster_centers_[j] #np.mean(X[jIndex, :], axis = 0)
        cluster_centers = cluster_centers + (centerj,)
        center_nn = np.append(center_nn, jIndex[np.argmin(euclidean_distances([kmeans.cluster_centers_[j]], X_tsne[jIndex]))])

    cluster_centers = np.vstack(cluster_centers)

    if save_file:
        np.savetxt(dataset_name + '_kmeans_labels.txt', cluster_label)
        np.savetxt(dataset_name + '_kmeans_centers.txt', cluster_centers)
        np.savetxt(dataset_name + '_center_nn.txt', center_nn)
        cluster_label, cluster_centers, center_nn = [],[],[]
    else:
        #print("Visualize KMeans")
        #visualize_kmeans(X, cluster_label)
        #reduced_data = PCA(n_components=2).fit_transform(X[:100])
        #kmeans = KMeans(n_clusters = num_cluster, max_iter = 50, n_jobs=4).fit(reduced_data)
        #plot_kmeans_2d(kmeans, reduced_data, cluster_label)
        #print("Done visualizing kmeans")
        #exit()
        return cluster_label, cluster_centers, center_nn

def plot_kmeans_hist(labels_, bins):
    #print(np.histogram(labels_, bins=bins))
    hist, bin_edges = np.histogram(labels_, bins=bins)
    plt.hist(hist, bins=bins)
    plt.title("Histogram with 100 bins")
    plt.show()

def plot_kmeans_2d(kmeans, reduced_data, cluster_label):
    from time import time
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import metrics
    from sklearn.cluster import KMeans
    #from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale

    np.random.seed(42)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

