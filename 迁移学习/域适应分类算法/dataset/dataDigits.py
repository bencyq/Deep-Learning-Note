import torchvision.datasets as D
from PIL import  Image
import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader


class MNIST(D.MNIST):

    def __init__(self, root, mode='L', split='train', **kwargs):
        assert mode in ["L", "RGB"]
        assert split in ['train', 'test']
        super(MNIST, self).__init__(root, train=split=='train', **kwargs)
        self.mode = mode

    def __getitem__(self, index):

        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L').convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class USPS(D.USPS):

    def __init__(self, root, mode='L', split='train', **kwargs):
        assert mode in ["L", "RGB"]
        assert split in ['train', 'test']
        super(USPS, self).__init__(root, train=split=='train', **kwargs)
        self.mode = mode

    def __getitem__(self, index):

        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img, mode='L').convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class SVHN(D.SVHN):

    def __init__(self, root, mode='RGB', **kwargs):
        assert mode in ["L", "RGB"]
        super(SVHN, self).__init__(root, **kwargs)
        self.mode = mode

    def __getitem__(self, index):

        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert(self.mode)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



