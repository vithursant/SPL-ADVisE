import torch
import torch.nn.functional as F
from torch.autograd import Variable

EULER = 0.57721566490153286060651209008240243104215933593992

def one_hot(size, index):
    """ Creates a matrix of one hot vectors.
        ```
        import torch
        import torch_extras
        setattr(torch, 'one_hot', torch_extras.one_hot)
        size = (3, 3)
        index = torch.LongTensor([2, 0, 1]).view(-1, 1)
        torch.one_hot(size, index)
        # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        ```
    """
    mask = torch.LongTensor(*size).fill_(0)
    ones = 1
    if isinstance(index, Variable):
        ones = Variable(torch.LongTensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    ret = mask.scatter_(1, index, ones)
    return ret

def sample_uniform_noise(input):
    return Variable(torch.randn(input.size()).add_(1e-20))

def accurate_uniform_noise(input, N=10000):
    return Variable(torch.randn([input.size()[-1]*N]).add_(1e-20))

def sample_gumbel(input):
    noise = torch.rand(input.size())
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)

def accurate_gumbel(input, N=10000):
    noise = torch.rand([input.size()[-1]*N])
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    #'''
    noise.add_(-EULER)
    #'''
    return Variable(noise)

def gumbel_softmax_sample(input, dummy_hot, hard=True, hi_def=True, N=10000, is_training=True):
    temperature = 1.0

    # hi_def is more accurate version of gumbel estimator as described in https://arxiv.org/abs/1706.04161
    # averages N gumbel distributions and subtracts out Euler's constant
    if is_training:
        if hi_def:
            noise = accurate_gumbel(input, N)
            noise = noise.view(N, input.size(-1))
            noise = torch.mean(noise, 0)
        else:
            noise = sample_gumbel(input)
        
        input = (input + noise) / temperature

    x = F.softmax(input)
    
    if hard:
        x_hard = x.squeeze(0).max(0)[1]
        dummy_hot.zero_()
        dummy_hot.scatter_(0, x_hard.data, 1)
        x = (Variable(dummy_hot) - x).detach() + x
    return x