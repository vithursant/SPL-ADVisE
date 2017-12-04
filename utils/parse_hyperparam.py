import argparse

model_names = ['vgg11', 'vgg16','resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wideresnet', 'lenet', 'magnetlenet', 'magnetfashion']
magnet_names = ['vgg11', 'vgg16', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wideresnet', 'lenet', 'magnetlenet', 'magnetfashion']
dataset_names = ['cifar10', 'cifar100', 'svhn', 'mnist', 'fashionmnist', 'tinyimagenet']

def parse_settings():
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--test_id', type=int, default=0, metavar='N',
                        help='test id number to be used for filenames')
    parser.add_argument('--model', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names)
    parser.add_argument('--embedding-model', '-s', metavar='ARCH', default='magnetlenet',
                        choices=magnet_names)
    parser.add_argument('--dataset', '-d', metavar='D', default='cifar10',
                        choices=dataset_names)
    parser.add_argument('--folder', '-f', default='baseline',
                        choices=['baseline', 'final_tests'])
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--spld', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--random', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='Number of classes')
    parser.add_argument('--max-iter', type=int, default=20, metavar='N',
                        help='Number of clustering iterations')
    parser.add_argument('--learning_rate1', type=float, default=0.1, metavar='N',
                        help='learning rate 1')
    parser.add_argument('--learning_rate2', type=float, default=1e-4, metavar='N',
                        help='learning rate 2')
    parser.add_argument('--data_augmentation', action='store_true', default=False,
                        help='augment data by flipping and cropping')
    parser.add_argument('--dropout_rate', type=float, default=0.3, metavar='N',
                        help='dropout rate')
    parser.add_argument('--square', action='store_true', default=False,
                        help='use square instead of circle for hole')
    parser.add_argument('--bernoulli', action='store_true', default=False,
                        help='add bernoulli noise')
    parser.add_argument('--en-scheduler', action='store_true', default=False,
                        help='Enable LR scheduler')
    parser.add_argument('--magnet', action='store_true', default=False,
                        help='Enable Magnet Loss')
    parser.add_argument('--leap', action='store_true', default=False,
                        help='Enable LEAP')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot clustering')

    return parser.parse_args()
