import os.path

from PIL import Image
from torch.utils import data
from scipy.io import loadmat

from .utils import register_dataset_obj, register_data_params
from .utils import DatasetParams

@register_data_params('synnum')
class SYNNUMParams(DatasetParams):
    
    num_channels = 3
    image_size   = 32
    #mean = 0.1307
    #std = 0.30
    #mean         = 0.254
    #std          = 0.369
    mean = 0.5
    std = 0.5
    num_cls      = 10

@register_dataset_obj('synnum')
class SYNNUM(data.Dataset):

    """SYNNUM 
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
download=False):

        self.root = root

        # self.train = train
        # if self.train:
        #     self.txt = os.path.join(root, '%s_train.txt'%self.src)
        # else:
        #     self.txt = os.path.join(root, '%s_test.txt'%self.src)

        self.mat = os.path.join(self.root, 'synth_test_32x32.mat')

        # self.img_path = []
        # self.labels = []
        self.transform = transform

        data = loadmat(os.path.join(self.root, 'synth_test_32x32.mat'))
        self.data = data['X'].transpose((3, 0, 1, 2))
        self.labels = data['y'].reshape(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        target = self.labels[index].astype(int)
        image = Image.fromarray(self.data[index]).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        return image, target
