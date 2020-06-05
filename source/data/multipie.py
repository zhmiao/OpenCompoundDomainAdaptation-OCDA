import os.path
from PIL import Image
from torch.utils.data import Dataset

# Within package imports
from .data_loader import register_dataset_obj, register_data_params
from .data_loader import DatasetParams

class MPParams(DatasetParams):
    
    num_channels = 3
    image_size   = 224
    mean         = 0.5
    std          = 0.5
    num_cls      = 5

class MP_dataset(Dataset):

    """MultiPIE datasets
    """

    def __init__(self, root, train=True, transform=None, *args):

        self.root = root
        self.train = train
        self.src = self.root.rsplit('/', 1)[1]
        self.dataroot = self.root.rsplit('/', 1)[0]

        if self.train:
            self.txt = os.path.join(root, '%s_train.txt'%self.src)
        else:
            self.txt = os.path.join(root, '%s_test.txt'%self.src)

        self.img_path = []
        self.labels = []
        self.transform = transform

        with open(self.txt) as f:
            for line in f:
                self.img_path.append(os.path.join(self.dataroot, line.split()[0]))
                self.labels.append(int(line.split()[1]))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        target = self.labels[index]
        
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        return image, target


@register_data_params('mp05')
class MP05Params(MPParams):
    pass

@register_dataset_obj('mp05')
class MP05(MP_dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
download=False):
        super(MP05, self).__init__(root=root, train=train, transform=transform)

@register_data_params('mp08')
class MP08Params(MPParams):
    pass

@register_dataset_obj('mp08')
class MP08(MP_dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
download=False):
        super(MP08, self).__init__(root=root, train=train, transform=transform)

@register_data_params('mp09')
class MP09Params(MPParams):
    pass

@register_dataset_obj('mp09')
class MP09(MP_dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
download=False):
        super(MP09, self).__init__(root=root, train=train, transform=transform)

@register_data_params('mp13')
class MP13Params(MPParams):
    pass

@register_dataset_obj('mp13')
class MP13(MP_dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
download=False):
        super(MP13, self).__init__(root=root, train=train, transform=transform)

@register_data_params('mp14')
class MP14Params(MPParams):
    pass

@register_dataset_obj('mp14')
class MP14(MP_dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
download=False):
        super(MP14, self).__init__(root=root, train=train, transform=transform)

@register_data_params('mp19')
class MP19Params(MPParams):
    pass

@register_dataset_obj('mp19')
class MP19(MP_dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
download=False):
        super(MP19, self).__init__(root=root, train=train, transform=transform)

        
        
        
    
