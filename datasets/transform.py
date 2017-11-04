import torchvision
from torchvision import transforms

def preform_transform(args):
    # Image Preprocessing
    if args.dataset == 'svhn':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                         std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    elif args.dataset in ['cifar10', 'cifar100']:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    elif args.dataset in ['mnist', 'fashionmnist']:
        normalize = transforms.Normalize((0.1307,), (0.3081,))

    train_transform = transforms.Compose([])

    if args.data_augmentation:
        if args.dataset in ['mnist', 'fashionmnist']:
            train_transform.transforms.append(transforms.RandomCrop(28, padding=4))
        elif args.dataset == 'tinyimagenet':
            train_transform.transforms.append(transforms.RandomCrop(32, padding=8))
        else:
            train_transform.transforms.append(transforms.RandomCrop(32, padding=4))

        train_transform.transforms.append(transforms.RandomHorizontalFlip())

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    return train_transform, test_transform
