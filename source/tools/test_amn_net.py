import torch

from os.path import join

from ..data.data_loader import load_data_multi
from ..models.models import get_model

import numpy as np
from tqdm import tqdm

import pdb

def test(loader, net, centers=False, feat_exp=False, out_dir=None, dom=None, cal_conf_mat=False):

    net.eval()

    test_loss = 0
    correct = 0
    total_labels = np.empty(0)

    if feat_exp:
        total_feat = np.empty((0, net.classifier.in_dims))
    if cal_conf_mat:
        conf_mat = np.zeros((net.classifier.out_dims, net.classifier.out_dims))

    N = len(loader.dataset)

    for idx, (data, target) in tqdm(enumerate(loader)):
        
        # setup data and target #
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        data.require_grad = False
        target.require_grad = False
        
        # forward pass
        if hasattr(net, 'tgt_net'):
            score, x = net.tgt_net(data.clone(), with_ft=True)
        else:
            score, x = net(data.clone(), with_ft=True)

        ###########################
        # storing direct feature

        if centers:
            
            direct_feature = x.clone()

            # set up visual memory
            # keys_memory = torch.from_numpy(net.centroids).float().cuda().clone()
            keys_memory = net.centroids.detach().clone()

            # computing memory feature by querying and associating visual memory
            values_memory = score.clone()
            values_memory = values_memory.softmax(dim=1)
            memory_feature = torch.matmul(values_memory, keys_memory)

            # computing concept selector
            concept_selector = net.fc_selector(x.clone()).tanh()
            x = direct_feature + concept_selector * memory_feature

            if feat_exp:
                total_feat = np.append(total_feat, x.detach().cpu().numpy(), axis=0)
            
            total_labels = np.append(total_labels, target.detach().cpu().numpy(), axis=0)

        # apply cosine norm classifier
        score = net.classifier(x.clone())
        ###########################
        
        # pdb.set_trace()

        # compute loss
        if centers:
            test_loss += net.cls_criterion(score.clone(), target).item()
        else:
            test_loss += net.criterion_cls(score.clone(), target).item()
        
        # compute predictions and true positive count
        _, pred = torch.max(score.clone(), 1)  # get the index of the max log-probability
        correct += (pred == target).cpu().sum().item()

        if cal_conf_mat:
            for i in range(len(target)):
                label = target[i]
                p = pred[i]
                conf_mat[label, p] += 1.
        
    if feat_exp:
        np.savez(join(out_dir, '%s'%dom), feature=total_feat, labels=total_labels)

    if cal_conf_mat:
        class_count = np.array([[len(total_labels[total_labels == i])] for i in range(net.classifier.out_dims)])
        conf_mat = conf_mat/class_count
        np.save(join(out_dir, 'conf_mat_%s'%dom), conf_mat)

    test_loss /= len(loader) # loss function already averages over batch size
    print('[Evaluate] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, N, 100. * correct / N))


def load_and_test_net(data, datadir, weights, model, num_cls, batch, 
                      dset='test', base_model=None, centers=False, feat_exp=False,
                      outdir=None, conf_mat=False, centroids_path=None, feat_dim=512):
    
    # Setup GPU Usage
    if torch.cuda.is_available(): 
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    # pdb.set_trace()
    # Eval tgt from AddaNet or TaskNet model #
    if model == 'AddaNet':
        net = get_model(model, num_cls=num_cls, weights_init=weights, 
                        model=base_model, feat_dim=feat_dim)
        # net = net.tgt_net
    elif model == 'DmeNet':
        net = get_model(model, num_cls=num_cls, weights_init=weights, 
                        model=base_model, centroids_path=centroids_path, feat_dim=feat_dim)
        # net = net.tgt_net
    else:
        net = get_model(model, num_cls=num_cls, weights_init=weights)

    # Load data
    test_data = load_data_multi(data, dset, batch=batch, 
                                rootdir=datadir, num_channels=net.num_channels,
                                image_size=net.image_size, download=True, kwargs=kwargs)

    if test_data is None:
        print('skipping test')
    else:
        test(test_data, net, centers=centers, feat_exp=feat_exp, out_dir=outdir, dom=data, cal_conf_mat=conf_mat)
