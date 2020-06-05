import os
def set_np_threads(n):
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
set_np_threads(4)


import numpy as np
import yaml
from os.path import join
from argparse import ArgumentParser

from source.algorithms.utils import compute_source_centroids
from source.algorithms.train_source_net import train_source
from source.algorithms.train_dme_net import train_dme_multi
from source.algorithms.train_style_net import train_style_multi
from source.algorithms.extract_style_ftr import extract_style_features
from source.algorithms.train_scheduled_dme_net import train_scheduled_dme_multi
from source.algorithms.test_cond_net import load_and_test_net


def main(args):

    ############################################################################################################

    ##################
    # Initialization #
    ##################
    # set gpu
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # read configuration
    np.random.seed(4325)
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config.items():
        setattr(args, k, v)

    # setup output file directories
    setattr(args, 'outdir_source', 'results/{}_to_{}'\
            .format(args.src, args.tgt))
    setattr(args, 'outdir_dme', '{}/dme'\
            .format(args.outdir_source))  # depends on outdir_source
    setattr(args, 'outdir_style', '{}/disentangle_dispell_{}_rec_{}'\
            .format(args.outdir_dme, args.gamma_dispell, args.gamma_rec))  # depends on outdir_dme
    setattr(args, 'outdir_scheduled', '{}/{}'\
            .format(args. outdir_style, args.config.split('/')[-1][:-5]))
    setattr(args, 'src_net_file', '{}/{}_net_{}.pth'\
            .format(args.outdir_source, args.base_model, args.src))
    setattr(args, 'centroids_src_file', '{}/centroids_src.npy'\
            .format(args.outdir_source))
    setattr(args, 'dme_net_file', '{}/dme_{}_net_{}_{}.pth'\
            .format(args.outdir_dme, args.base_model, args.src, args.tgt))
    setattr(args, 'style_net_file', '{}/StyleNet_{}_net_{}_{}.pth'\
            .format(args.outdir_style, args.style_model, args.src, args.tgt))
    setattr(args, 'scheduled_net_file', '{}/scheduled_{}_net_{}_{}.pth'\
            .format(args.outdir_scheduled, args.base_model, args.src, args.tgt))
    setattr(args, 'src_ftr_fn', '{}/src_style_ftr.bin'\
            .format(args.outdir_style))
    setattr(args, 'tgt_ftr_fn', '{}/tgt_style_ftr.bin'\
            .format(args.outdir_style))

    ############################################################################################################

    #######################
    # 1. Train Source Net #
    #######################
    if os.path.isfile(args.src_net_file):
        print('Skipping source net training, exists:', args.src_net_file)
    else:
        train_source(args)

    ################################
    # 2. Compute Initial Centroids #
    ################################
    if os.path.isfile(args.centroids_src_file):
        print('Skipping source centroids computation, exists:', args.centroids_src_file)
    else:
        compute_source_centroids(args)

    ####################
    # 3. Train DME Net #
    ####################
    if os.path.isfile(args.dme_net_file):
        print('Skipping dme training, exists:', args.dme_net_file)
    else:
        train_dme_multi(args)

    ##################################
    # 4. Train Disentangle Style Net #
    ##################################
    if os.path.isfile(args.style_net_file):
        print('Skip disentangle training, exists: {}.'.format(args.style_net_file))
    else:
        train_style_multi(args)

    #################################
    # 5. Preparation for Scheduling #
    #################################

    # Extract source and style features And Save
    if os.path.isfile(args.src_ftr_fn) and os.path.isfile(args.tgt_ftr_fn):
        print("Skip feature extraction, exists: {} and {}.".format(args.src_ftr_fn, args.tgt_ftr_fn))
    else:
        extract_style_features(args)

    # Loade source and target style features
    src_ftr = np.fromfile(args.src_ftr_fn, dtype=np.float32).reshape(-1, 512)
    tgt_ftr = np.fromfile(args.tgt_ftr_fn, dtype=np.float32).reshape(-1, 512)

    # Calculate style feature centroids from source
    # And calculate distances of target feature to the centroids
    if args.norm_style:
        src_ftr /= np.linalg.norm(src_ftr, axis=1, keepdims=True)
        tgt_ftr /= np.linalg.norm(tgt_ftr, axis=1, keepdims=True)
        src_center = src_ftr.mean(axis=0)[:, np.newaxis]
        dist = 1. - tgt_ftr.dot(src_center).squeeze()
    else:
        src_center = src_ftr.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(tgt_ftr - src_center, axis=1)

    # Based on style feature distances to the target, calculate data order
    setattr(args, 'sort_idx', np.argsort(dist))

    #########################################
    # 6. Domain Adaptiation With Scheduling #
    #########################################
    if os.path.isfile(args.scheduled_net_file):
        print('Skipping scheduled training, exists: {}'.format(args.scheduled_net_file))
    else:
        train_scheduled_dme_multi(args)

    #################
    # 7. Evaluation #
    #################
    test_list = args.tgt_list + ['synnum']

    for tgt in test_list:
        tgt_datadir = join(args.datadir, tgt)
        print('----------------')
        print('Test set:', tgt)
        print('----------------')
        print('Evaluating {}->{} dme model: {}'.format(args.src, tgt, args.scheduled_net_file))
        load_and_test_net(args, tgt, tgt_datadir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()
    main(args)
