import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
import os



class ImageList(datasets.VisionDataset):

    def __init__(self, root, classes, data_list_file, transform=None, target_transform=None):

        super(ImageList, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root_dir = root
        self.samples = self.parse_data_file(data_list_file)
        self.classes = classes
        self.loader = default_loader
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data_list_file = data_list_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target =  self.target_transform(target)
        return img, target

    def parse_data_file(self, file_name):

        with open(file_name, 'r') as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root_dir, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self):
        return len(self.classes)

    @classmethod
    def domains(self):
        raise NotImplemented