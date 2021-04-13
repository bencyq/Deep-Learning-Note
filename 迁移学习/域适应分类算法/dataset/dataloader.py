from .dataClassification import OfficeHome, Office31, VisDA2017
from .dataDigits import MNIST, SVHN, USPS
from torchvision import transforms as T
from torch.utils.data import DataLoader



class ResizeImage(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class ForeverDataIterator(object):

    r"""A data iterator that will never stop producing data"""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def get_dataloader(args, phase='train', domain='source'):

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if phase == 'train':
        if args.center_crop:
            data_transform = T.Compose([
                ResizeImage(256),
                T.CenterCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:

            data_transform = T.Compose([
                ResizeImage(256),
                T.RandomResizedCrop(224), # T.CenterCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
    else:
        data_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

    if domain == 'source': ### OfficeHome, Office31, VisDA2017
        dataset = Office31(args.root, args.source, transform=data_transform)
    else:
        dataset = Office31(args.root, args.target, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    dataiter = ForeverDataIterator(dataloader)
    return dataloader, dataiter


def get_digits_dataloader(args,  split='train', phase='train', domain='source'):

    if args.num_channels == 3:
        mode = 'RGB'
        mean = std = [0.5, 0.5, 0.5]
    else:
        mode = 'L'
        mean = std = [0.5, ]
    normalize = T.Normalize(mean=mean, std=std)
    if phase == 'train':
        data_transform = T.Compose([
            ResizeImage(args.image_size),
            T.ToTensor(),
            normalize
        ])
    else:
        data_transform = T.Compose([
            ResizeImage(args.image_size),
            T.ToTensor(),
            normalize
        ])
    if domain == 'source': ### MNIST, SVHN, USPS
        if args.source == 'MNIST':
            dataset = MNIST(args.root, mode=mode, split='train', download=True, transform=data_transform)
        elif args.source == 'SVHN':
            dataset = SVHN(args.root, mode=mode, download=True, transform=data_transform)
        else:
            dataset = USPS(args.root, mode=mode, split='train', download=True, transform=data_transform)
    else:
        if args.target == 'MNIST':
            dataset = MNIST(args.root, mode=mode, split='test', download=True, transform=data_transform)
        elif args.target == 'SVHN':
            dataset = SVHN(args.root, mode=mode, download=True, transform=data_transform)
        else:
            dataset = USPS(args.root, mode=mode, split='test', download=True, transform=data_transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    dataiter = ForeverDataIterator(dataloader)
    return dataloader, dataiter


if __name__ == '__main__':

    from dataDigits import MNIST, SVHN, USPS
    from torchvision import transforms as T
    from torch.utils.data import DataLoader
    import argparse
    '/media/fei/H/Transfer/examples/domain_adaptation/digits'
    data_dir = '../../examples/domain_adaptation/digits/data/'
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    parser.add_argument('--root', metavar='DIR', default=data_dir, help='root path of dataset')
    parser.add_argument('--data', metavar='DATA', default='SVHN')
    parser.add_argument('--source', default='s', help='source domain(s)')
    parser.add_argument('--target', default='t', help='target domain(s)')
    parser.add_argument('--batch-size', default=32, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('--workers', default=2, type=int, help='loading workers (default: 4)')
    parser.add_argument('--num-channels', default=3, choices=[1, 3], type=int, help='the number of image channels')
    parser.add_argument('--image-size', type=int, default=32,  help='the size of input image')
    args = parser.parse_args()
    _, dataloader = get_digits_dataloader(args, phase='train', domain='source')
    ###显示
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import numpy as np
    img, target = next(dataloader)
    imgs = make_grid(img, nrow=8, normalize=True, pad_value=2)
    imgs = np.transpose(imgs.numpy(), (1, 2, 0))
    plt.imshow(imgs)
    plt.show()