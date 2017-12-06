import argparse

model_names = [ 'vgg11',
                'vgg16',
                'resnet18',
                'resnet34',
                'resnet50',
                'resnet101',
                'resnet152',
                'wideresnet',
                'lenet',
                'magnetlenet',
                'magnetfashion',
                'inceptionv3',
                'resnext',
                'densenet',
                'preresnet']

magnet_names = ['vgg11',
                'vgg16',
                'resnet18',
                'resnet34',
                'resnet50',
                'resnet101',
                'resnet152',
                'wideresnet',
                'lenet',
                'magnetlenet',
                'magnetfashion']

dataset_names = [   'cifar10',
                    'cifar100',
                    'svhn',
                    'mnist',
                    'fashionmnist',
                    'tinyimagenet',
                    'cub2002010',
                    'cub2002011']

def parse_settings():
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--test_id', type=int, default=0, metavar='N',
                        help='test id number to be used for filenames')
    parser.add_argument('--dataset', '-d', metavar='D', default='cifar10',
                        choices=dataset_names)
    parser.add_argument('--folder', '-f', default='baseline',
                        choices=['baseline', 'final_tests'])

    # Batch Selector
    parser.add_argument('--magnet', action='store_true', default=False,
                        help='Enable Magnet Loss')
    parser.add_argument('--leap', action='store_true', default=False,
                        help='Enable LEAP')
    parser.add_argument('--random', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--spld', action='store_true', default=False,
                        help='enables CUDA training')

    # Optimization
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
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
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--en-scheduler', action='store_true', default=False,
                        help='Enable LR scheduler')

    # Checkpoints
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot clustering')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # Architecture
    parser.add_argument('--model', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names)
    parser.add_argument('--embedding-model', '-s', metavar='ARCH', default='magnetlenet',
                        choices=magnet_names)
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--depth', type=int, default=29,
                        help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8,
                        help='Model cardinality (group).')
    parser.add_argument('--widen-factor', type=int, default=4,
                        help='Widen factor. 4 -> 64, 8 -> 128, ...')
    parser.add_argument('--growthRate', type=int, default=12,
                        help='Growth rate for DenseNet.')
    parser.add_argument('--compressionRate', type=int, default=2,
                        help='Compression Rate (theta) for DenseNet.')

    # Misc
    parser.add_argument('--manualSeed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    #Device options
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    return parser.parse_args()
