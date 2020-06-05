import gzip
import os.path

from urllib.parse import urljoin
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets

# Within package imports
from .data_loader import register_dataset_obj, register_data_params
from . import utils
from .data_loader import DatasetParams

@register_data_params('mnistm')
class MNISTMParams(DatasetParams):
    
    num_channels = 3
    image_size   = 32
    mean         = 0.5
    std          = 0.5
    num_cls      = 10

@register_dataset_obj('mnistm')
class MNISTM(datasets.ImageFolder):

    """MNISTM
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
download=False):

        self.root = root
        self.train = train
        if self.train:
            self.root = os.path.join(root, 'train')
        else:
            self.root = os.path.join(root, 'test')

        super(MNISTM, self).__init__(self.root, transform=transform, target_transform=target_transform)
        
        
    
