import numpy as np

class ReweightingPolicy(object):
    """ReweightingPolicy defines how we weigh the samples given their
    importance.
    
    Each policy should provide
        1. A layer implementation for use with Keras models
        2. A python implementation for use with the samplers
    """
    def weight_layer(self):
        """Return a layer that accepts the scores and the return value of the
        sample_weights() method and produces new sample weights."""
        raise NotImplementedError()

    def sample_weights(self, idxs, scores):
        """Given the scores and the chosen indices return whatever is needed by
        the weight_layer to produce the sample weights."""
        raise NotImplementedError()

    @property
    def weight_size(self):
        """Return how many numbers per sample make up the sample weights"""
        raise NotImplementedError()

class BiasedReweightingPolicy(ReweightingPolicy):
    """BiasedReweightingPolicy computes the sample weights before  the
    forward-backward pass based on the sampling probabilities. It can introduce
    a bias that focuses on the hard examples when combined with the loss as an
    importance metric."""
    def __init__(self, k=1.0):
        self.k = k

    def weight_layer(self):
        return ExternalReweighting()

    def sample_weights(self, idxs, scores):
        N = len(scores)
        s = scores[idxs]
        w = scores.sum() / N / s
        w_hat = w**self.k
        w_hat *= w.dot(s) / w_hat.dot(s)

        return w_hat[:, np.newaxis]

    @property
    def weight_size(self):
        return 1