from .officehome import OfficeHome
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
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224), # T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
    if domain == 'source':
        dataset = OfficeHome(args.root, args.source, transform=transform)
    else:
        dataset = OfficeHome(args.root, args.target, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    dataiter = ForeverDataIterator(dataloader)
    return dataloader, dataiter



if __name__ == '__main__':
    import argparse
    data_dir = '../../examples/domain_adaptation/classification/data/officehome'
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    parser.add_argument('--root', metavar='DIR', default=data_dir, help='root path of dataset')
    parser.add_argument('--data', metavar='DATA', default='OfficeHome')
    parser.add_argument('--source', default='Ar', help='source domain(s)')
    parser.add_argument('--target', default='Cl', help='target domain(s)')
    parser.add_argument('--batch-size', default=32, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('--workers', default=2, type=int, help='loading workers (default: 4)')
    args = parser.parse_args()
    _, dataloader = get_dataloader(args, phase='train', domain='source')
    ###显示
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import numpy as np
    img, target = next(dataloader)
    imgs = make_grid(img, nrow=8, normalize=True, pad_value=2)
    imgs = np.transpose(imgs.numpy(), (1, 2, 0))
    plt.imshow(imgs)
    plt.show()