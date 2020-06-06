import os
import numpy as np

import torch

from ..models.utils import get_model
from ..data.utils import load_data_multi

import pdb


def extract_domain_factor_features(args):

    src = args.src
    tgt_list = args.tgt_list
    base_model = args.base_model
    domain_factor_model = args.domain_factor_model
    num_cls = args.num_cls
    batch = args.batch
    datadir = args.datadir
    outdir = args.outdir_domain_factor
    weights = args.domain_factor_net_file

    if torch.cuda.is_available():
        kwargs = {'num_workers': 8, 'pin_memory': True}
    else:
        kwargs = {}

    net = get_model('DomainFactorNet', num_cls=num_cls,
                    base_model=base_model, domain_factor_model=domain_factor_model,
                    weights_init=weights, eval=True)
    print(net)
    print('Extracting domain_factor features')

    src_data = load_data_multi(src, 'train', batch=batch,
                               rootdir=os.path.join(datadir, src), num_channels=net.num_channels,
                               image_size=net.image_size, download=False, shuffle=False, kwargs=kwargs)
    tgt_data = load_data_multi(tgt_list, 'train', batch=batch,
                               rootdir=datadir, num_channels=net.num_channels,
                               image_size=net.image_size, download=False, shuffle=False, kwargs=kwargs)

    net.eval()

    src_ftrs = extract_dataset(src_data, net)
    src_ftrs.tofile(os.path.join(outdir, 'src_domain_factor_ftr.bin')) # N x 512
    tgt_ftrs = extract_dataset(tgt_data, net)
    tgt_ftrs.tofile(os.path.join(outdir, 'tgt_domain_factor_ftr.bin'))


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
            domain_factor_ftr = net.domain_factor_net(data.clone()) # Bx512

        ftrs.append(domain_factor_ftr.detach().cpu().numpy())

        if batch_idx % 100 == 0:
            print(info_str)

    ftrs = np.concatenate(ftrs, axis=0)
    assert len(loader.dataset) == ftrs.shape[0], "{} vs {}".format(len(loader), ftrs.shape[0])
    return ftrs
