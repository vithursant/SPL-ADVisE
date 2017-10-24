import argparse

def parse_settings():
	model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wideresnet', 'googlenet', 'lenet', 'fashionlenet']
	dataset_names = ['cifar10', 'cifar100', 'svhn', 'fashionmnist', 'mnist', 'fashionlenet']
	
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST SPLD')
	#print(parser)
	parser.add_argument('--batch-size', type=int, default=32, metavar='N',
	                    help='input batch size for training (default: 32)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
	                    help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=50, metavar='N',
	                    help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
	                    help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
	                    help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
	                    help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=20, metavar='N',
	                    help='how many batches to wait before logging training status')
	parser.add_argument('--spl', action='store_true', default=False,
						help='enables self-paced learning framework')
	parser.add_argument('--stratified', action='store_true', default=False,
						help='enables stratified sampling')
	parser.add_argument('--spld', action='store_true', default=False,
						help='enables self-paced learning with diversity')
	parser.add_argument('--spldml', action='store_true', default=False,
						help='Use the SPL DML sampler')
	parser.add_argument('--dml', action='store_true', default=False,
						help='enables deep metric learning')
	parser.add_argument('--curriculum-epochs', type=int, default=40, metavar='N',
						help='Number of curriculum epochs')
	parser.add_argument('--num-cluster', type=int, default=100, metavar='N',
						help='Number of clusters for clustering')
	parser.add_argument('--epoch-iters', type=int, default=10, metavar='N',
						help='Number of iterations per epoch')
	parser.add_argument('--num-classes', type=int, default=10, metavar='N',
						help='Number of classes in the dataset')
	parser.add_argument('--loss-weight', type=float, default=1.2e+6, metavar='LW',
						help='The loss weight')
	parser.add_argument('--curriculum-rate', type=float, default=0.03, metavar='CR',
						help='The curriculum learning rate')
	parser.add_argument('--decay-after-epochs', type=int, default=10, metavar='N',
						help='Decay after epochs')
	parser.add_argument('--magnet-loss', action='store_true', default=False,
						help='Enables the magnet loss for representation learning')
	parser.add_argument('--mnist', action='store_true', default=False,
						help='Use the mnist dataset')
	parser.add_argument('--svhn', action='store_true', default=False,
						help='Use the SVHN dataset')
	parser.add_argument('--fashionmnist', action='store_true', default=False,
						help='Use the Fashion MNIST dataset')
	parser.add_argument('--cifar10', action='store_true', default=False,
						help='Use the CIFAR-10 dataset')
	parser.add_argument('--cifar100', action='store_true', default=False,
						help='Use the CIFAR-100 dataset')
	parser.add_argument('--name', default='SPLD', type=str,
						help='name of experiment')
	parser.add_argument('--visdom', dest='visdom', action='store_true', default=False,
						help='Use visdom to track and plot')
	parser.add_argument('--mining', type=str, default='Hardest',
						help='Method to use for mining hard examples')
	parser.add_argument('--feature-size', type=int, default=64,
						help='size for embeddings/features to learn')
	parser.add_argument('--lr-freq', default=5, type=int,
						help='learning rate changing frequency (default: 5)')
	parser.add_argument('--tensorboard',
						help='Log progress to TensorBoard', action='store_true')
	parser.add_argument('--fea-freq', default=5, type=int,
						help='Refresh clusters (default: 5)')
	parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
						help='weight decay (default: 5e-4)')
	parser.add_argument('--no-augment', dest='augment', action='store_false',
						help='whether to use standard augmentation (default: True)')
	parser.add_argument('--start-epoch', default=0, type=int,
						help='manual epoch number (useful on restarts)')
	parser.add_argument('--print-freq', '-p', default=10, type=int,
						help='print frequency (default: 10)')
	parser.add_argument('--net-type', default='wide-resnet', type=str, help='model')
	parser.add_argument('--depth', default=28, type=int, help='depth of model')
	parser.add_argument('--widen-factor', default=10, type=int, help='width of model')
	parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
	parser.add_argument('--shallow', action='store_true', default=False,
						help='Enable shallow network magnet loss')
	parser.add_argument('-lr_decay_ratio', default=0.1)
	parser.add_argument('-lr_patience', default=10)
	parser.add_argument('-lr_threshold', default=0.02)
	parser.add_argument('-lr_delay', default=5)
	parser.add_argument('--folder', '-f', default='baseline', choices=['baseline', 'final_tests'])

	parser.add_argument('--test_id', type=int, default=0, metavar='N', help='test id number to be used for filenames')

	parser.add_argument('--model', '-a', metavar='ARCH', default='resnet18', choices=model_names)
	parser.add_argument('--dataset', '-d', metavar='D', default='cifar10', choices=dataset_names)
	return parser.parse_args()