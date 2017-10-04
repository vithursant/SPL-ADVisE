import nose
from nose.tools import *
from magnet_ops import *

def test_magnet_loss():
	"""Test magnet loss ops."""    
	rand = np.random.RandomState(42)

	# Hyperparams
	m = 6
	d = 4
	K = 5

	# Sample test data
	rdata = rand.random_sample([m*d, 6])
	clusters = np.repeat(range(m), d)

	cluster_classes1 = range(m)
	classes1 = np.repeat(cluster_classes1, d)

	cluster_classes2 = [0, 1, 1, 3, 4, 5]
	classes2 = np.repeat(cluster_classes2, d)

	total_loss2, example_losses2 = magnet_loss(rdata, 
											   classes1, 
											   clusters,
	                                           cluster_classes1, 
	                                           m)

	return total_loss2, example_losses2
	
if __name__ == '__main__':
    total_loss, losses = test_magnet_loss()
    print(total_loss)
    print(losses)