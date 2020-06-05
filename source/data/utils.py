import logging
import os.path
from os.path import join
import numpy as np
import requests

import torch
from torchvision import transforms


logger = logging.getLogger(__name__)


def maybe_download(url, dest):
    """Download the url to dest if necessary, optionally checking file
    integrity.
    """
    if not os.path.exists(dest):
        logger.info('Downloading %s to %s', url, dest)
        download(url, dest)


def download(url, dest):
    """Download the url to dest, overwriting dest if it already exists."""
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def load_data_multi(name, dset, batch=64, rootdir='', num_channels=3,
                    image_size=32, download=True, shuffle=True, kwargs={}):
    '''
    [14/05/2019] modified by zxh: adding arg: shuffle
    '''
    if dset != 'train':
        shuffle = False

    if isinstance(name, list):  # load multi
        dataset_list = []
        for i in range(len(name)):
            dataset_list.append(
                get_dataset(name[i], join(rootdir, name[i]), dset, image_size, num_channels, download=download))
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        dataset = get_dataset(name, rootdir, dset, image_size, num_channels,
                              download=download)

    if len(dataset) == 0:
        return None
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch,
                                         shuffle=shuffle, **kwargs)
    return loader


def get_dataset_multi(name, dset, rootdir='', num_channels=3,
                      image_size=32, download=True):
    if isinstance(name, list):  # load multi
        dataset_list = []
        for i in range(len(name)):
            dataset_list.append(
                get_dataset(name[i], join(rootdir, name[i]), dset, image_size, num_channels, download=download))
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        dataset = get_dataset(name, rootdir, dset, image_size, num_channels,
                              download=download)

    if len(dataset) == 0:
        return None
    return dataset


def get_transform(params, image_size, num_channels):
    # Transforms for PIL Images: Gray <-> RGB
    Gray2RGB = transforms.Lambda(lambda x: x.convert('RGB'))
    RGB2Gray = transforms.Lambda(lambda x: x.convert('L'))

    transform = []
    # Does size request match original size?
    # if not image_size == params.image_size:
    transform.append(transforms.Resize((image_size, image_size)))

    # Does number of channels requested match original?
    if not num_channels == params.num_channels:
        if num_channels == 1:
            transform.append(RGB2Gray)
        elif num_channels == 3:
            transform.append(Gray2RGB)
        else:
            print('NumChannels should be 1 or 3', num_channels)
            raise Exception

    transform += [transforms.ToTensor(),
                  transforms.Normalize((params.mean,), (params.std,))]

    return transforms.Compose(transform)


def get_target_transform(params):
    t_uniform = transforms.Lambda(lambda x: x[:, 0]
    if isinstance(x, (list, np.ndarray)) and len(x) == 2 else x)
    return t_uniform


data_params = {}


def register_data_params(name):
    def decorator(cls):
        data_params[name] = cls
        return cls

    return decorator


dataset_obj = {}


def register_dataset_obj(name):
    def decorator(cls):
        dataset_obj[name] = cls
        return cls

    return decorator


class DatasetParams(object):
    "Class variables defined."
    num_channels = 1
    image_size = 16
    mean = 0.5
    std = 0.5
    num_cls = 10
    target_transform = None


def get_dataset(name, rootdir, dset, image_size, num_channels, download=True):
    is_train = (dset == 'train')
    print('get dataset:', name, rootdir, dset)

    params = data_params[name]
    transform = get_transform(params, image_size, num_channels)
    target_transform = get_target_transform(params)
    target_transform = None
    return dataset_obj[name](rootdir, train=is_train, transform=transform,
                             target_transform=target_transform, download=download)
