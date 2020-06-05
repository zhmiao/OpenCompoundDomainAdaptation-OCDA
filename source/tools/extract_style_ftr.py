import os
import numpy as np

import torch

from ..models.models import get_model
from ..data.data_loader import load_data_multi

import pdb


def extract_style_features(src, tgt_list, base_model, style_model, num_cls, batch=128,
                           datadir="", outdir="", weights=None):

    if torch.cuda.is_available():
        kwargs = {'num_workers': 8, 'pin_memory': True}
    else:
        kwargs = {}

    net = get_model('StyleNet', num_cls=num_cls,
                    base_model=base_model, style_model=style_model,
                    weights_init=weights, eval=True)
    print(net)
    print('Extracting style features')

    src_data = load_data_multi(src, 'train', batch=batch,
                               rootdir=os.path.join(datadir, src), num_channels=net.num_channels,
                               image_size=net.image_size, download=False, shuffle=False, kwargs=kwargs)
    tgt_data = load_data_multi(tgt_list, 'train', batch=batch,
                               rootdir=datadir, num_channels=net.num_channels,
                               image_size=net.image_size, download=False, shuffle=False, kwargs=kwargs)

    net.eval()

    src_ftrs = extract_dataset(src_data, net)
    src_ftrs.tofile(os.path.join(outdir, 'src_style_ftr.bin')) # N x 512
    tgt_ftrs = extract_dataset(tgt_data, net)
    tgt_ftrs.tofile(os.path.join(outdir, 'tgt_style_ftr.bin'))


def extract_dataset(loader, net):

    np.random.seed(4325)
    ftrs = []

    for batch_idx, (data, _) in enumerate(loader):

        info_str = "[Extract] [{}/{} ({:.2f}%)]".format(
            batch_idx * len(data), len(loader.dataset), 100 * float(batch_idx) / len(loader))

        if torch.cuda.is_available():
            data = data.cuda()

        data.require_grad = False

        with torch.no_grad():
            style_ftr = net.style_net(data.clone()) # Bx512

        ftrs.append(style_ftr.detach().cpu().numpy())

        if batch_idx % 100 == 0:
            print(info_str)

    ftrs = np.concatenate(ftrs, axis=0)
    assert len(loader.dataset) == ftrs.shape[0], "{} vs {}".format(len(loader), ftrs.shape[0])
    return ftrs
