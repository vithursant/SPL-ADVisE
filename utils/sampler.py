import torch
from torch._six import int_classes as _int_classes
from torch.utils.data import Sampler

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