import torch
import sklearn

class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices, batch_indices):
        self.indices = indices
        self.batch_indices = batch_indices

    def __iter__(self):
        return (self.indices[i] for i in self.batch_indices)

    def __len__(self):
        return len(self.indices)

class SubsetSequentialSamplerSPLD(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in self.indices)

    def __len__(self):
        return len(self.indices)

class SubsetSequentialSamplerSPLDML(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices, batch_indices):
        self.indices = indices
        self.batch_indices = batch_indices

    def __iter__(self):
        #self.batch_idx = []
        #for i in self.batch_indices:
        #    self.batch_idx.append(self.indices[i])

        return (self.indices[i] for i in self.batch_indices)

    def __len__(self):
        return len(self.indices)

"""
Classification based sampler.
"""
class ClassificationBasedSampler(object):
    def __init__(self, num_classes, num_samples):
        self.num_classes = int(num_classes)
        self.num_samples = int(num_samples)
        self.negatives = []

    def Reset(self):
        self.negatives = []

    def SampleNegatives(self, labels_true, labels_pred):
        neg_indices = np.where(labels_true != labels_pred)[0]
        true_classes = labels_true[neg_indices]
        pred_classes = labels_pred[neg_indices]  # what cluster does this point falsely belong to?

        cor_indices = np.where(labels_true == labels_pred)[0]  # some points correctly clustered
        cor_classes = labels_true[cor_indices]
        anchor_candidates = dict()

        for c in np.unique(cor_classes):
            subset_indices = np.where(cor_classes == c)[0]
            anchor_candidates[c] = cor_indices[subset_indices]

        # now get an anchor point (a correct point correctly there in that cluster)
        for i in np.random.permutation(len(neg_indices)):
            pred_class = pred_classes[i]  # predicted class for incorrectly classified point
            if pred_class in anchor_candidates.keys():  # there is an anchor point, a point correctly sent to that class
                self.negatives.append((np.random.choice(anchor_candidates[pred_class]), neg_indices[i]))
            if len(self.negatives) == self.num_samples:
                break

    def ChooseNegatives(self, num):
        sel_indices = np.random.choice(range(len(self.negatives)), num)
        return ([self.negatives[i] for i in sel_indices])

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)