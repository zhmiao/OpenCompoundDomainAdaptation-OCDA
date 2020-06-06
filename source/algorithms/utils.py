from os.path import join
import numpy as np
from tqdm import tqdm
import torch
from ..data.utils import load_data_multi
from ..models.utils import get_model


def class_count (data):

    try:
        labels = np.array(data.dataset.labels)
    except AttributeError:
        labels = np.array(data.dataset.targets)

    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num


def compute_source_centroids(args):

    data = args.src
    datadir = join(args.datadir, args.src)
    weights = args.src_net_file
    model = args.base_model
    num_cls = args.num_cls
    save_path = args.centroids_src_file
    batch = args.batch
    dset = 'train'
    base_model = None
    feat_dim = 512

    print('Calculating centroids.')

    # Setup GPU Usage
    if torch.cuda.is_available():
        kwargs = {'num_workers': 8, 'pin_memory': True}
    else:
        kwargs = {}

    net = get_model(model, num_cls=num_cls, weights_init=weights, feat_dim=feat_dim)

    # Load data
    train_data = load_data_multi(data, dset, batch=batch,
                                 rootdir=datadir, num_channels=net.num_channels,
                                 image_size=net.image_size, download=True, kwargs=kwargs)

    net.eval()

    centroids = np.zeros((num_cls, feat_dim))

    for idx, (data, target) in tqdm(enumerate(train_data)):

        # setup data and target #
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        data.require_grad = False
        target.require_grad = False

        # forward pass
        _, x = net(data.clone())

        # add feed-forward feature to centroid tensor
        for i in range(len(target)):
            label = target[i]
            centroids[label.item()] += x[i].detach().cpu().numpy()

    # Average summed features with class count
    centroids /= np.array(class_count(train_data)).astype(float).reshape((-1, 1))

    np.save(save_path, centroids)
