import torch

from copy import deepcopy
from os.path import join

from ..data.utils import load_data_multi
from ..models.utils import get_model

import numpy as np


def test(loader, net, domain_factor_net, centers=False, domain_factor_cond=0, feat_exp=False, out_dir=None, dom=None):

    net.eval()
    domain_factor_net.eval()

    test_loss = 0
    correct = 0
    total_feat = np.empty((0, net.classifier.in_dims))
    total_labels = np.empty(0)
   
    N = len(loader.dataset)

    for idx, (data, target) in enumerate(loader):
        
        # setup data and target #
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        data.require_grad = False
        target.require_grad = False
        
        # forward pass
        if hasattr(net, 'tgt_net'):
            score, x = net.tgt_net(data.clone())
        else:
            score, x = net(data.clone())

        ###########################
        # storing direct feature

        if centers:
            
            direct_feature = x.clone()

            # set up visual memory
            keys_memory = net.centroids.detach().clone()

            # computing memory feature by querying and associating visual memory
            values_memory = score.clone()
            values_memory = values_memory.softmax(dim=1)
            memory_feature = torch.matmul(values_memory, keys_memory)

            if domain_factor_cond == 0:
                # computing concept selector
                concept_selector = net.fc_selector(x.clone()).tanh()
                class_enhancer = concept_selector * memory_feature
                x = direct_feature + class_enhancer
            elif domain_factor_cond == 1:
                with torch.no_grad():
                    domain_factor_ftr = domain_factor_net(data).detach()
                domain_factor_selector = net.domain_factor_selector(x).tanh()
                x = direct_feature + domain_factor_selector * domain_factor_ftr
            elif domain_factor_cond == 2:
                # computing concept selector
                concept_selector = net.fc_selector(x.clone()).tanh()
                with torch.no_grad():
                    domain_factor_ftr = domain_factor_net(data.clone()).detach()
                domain_factor_selector = net.domain_factor_selector(x.clone()).tanh()
                x = direct_feature + concept_selector * memory_feature + 0.01 * domain_factor_selector * domain_factor_ftr
            elif domain_factor_cond == 3:
                with torch.no_grad():
                    domain_factor_ftr = domain_factor_net(data.clone()).detach()
                domain_indicator = net.domain_factor_selector(domain_factor_ftr.clone()).tanh()
                x = direct_feature + domain_indicator * memory_feature
            elif domain_factor_cond == 4:
                # computing concept selector
                concept_selector = net.fc_selector(x.clone()).tanh()
                with torch.no_grad():
                    domain_factor_ftr = domain_factor_net(data.clone()).detach()
                domain_factor_selector = net.domain_factor_selector(domain_factor_ftr.clone()).tanh()
                x = direct_feature + domain_factor_selector * concept_selector * memory_feature
            else:
                raise Exception("No such domain_factor_cond: {}".format(domain_factor_cond))

            if feat_exp:
                total_feat = np.append(total_feat, x.clone().detach().cpu().numpy(), axis=0)
                total_labels = np.append(total_labels, target.clone().detach().cpu().numpy(), axis=0)

        # apply cosine norm classifier
        score = net.classifier(x.clone())
        ###########################
        
        # compute loss
        if centers:
            test_loss += net.cls_criterion(score.clone(), target).item()
        else:
            test_loss += net.criterion_cls(score.clone(), target).item()
        
        # compute predictions and true positive count
        pred = torch.argmax(score.clone(), 1)  # get the index of the max log-probability
        correct += (pred == target).cpu().sum().item()
        
    if feat_exp:
        np.savez(join(out_dir, '%s' % dom), feature=total_feat, labels=total_labels)

    test_loss /= len(loader) # loss function already averages over batch size
    print('[Evaluate] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, N, 100. * correct / N))


def load_and_test_net(args, data, datadir):

    weights = args.scheduled_net_file
    domain_factor_weights = args.domain_factor_net_file
    domain_factor_model = args.domain_factor_model
    num_cls = args.num_cls
    domain_factor_cond = args.domain_factor_cond
    batch = args.batch
    dset = 'test'
    base_model = args.base_model
    centers = True
    centroids_path = args.centroids_src_file
    feat_exp = False
    outdir = None

    # Setup GPU Usage
    if torch.cuda.is_available(): 
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    # pdb.set_trace()
    net = get_model('MannNet', num_cls=num_cls, weights_init=weights,
                    model=base_model, use_domain_factor_selector=(domain_factor_cond != 0), centroids_path=centroids_path)

    domain_factor_net = deepcopy(get_model('DomainFactorNet', num_cls=num_cls,
                                   base_model=base_model, domain_factor_model=domain_factor_model,
                                   weights_init=domain_factor_weights, eval=True).domain_factor_net)

    domain_factor_net.eval()

    # Load data
    test_data = load_data_multi(data, dset, batch=batch, 
                                rootdir=datadir, num_channels=net.num_channels,
                                image_size=net.image_size, download=True, kwargs=kwargs)

    if test_data is None:
        print('skipping test')
    else:
        test(test_data, net, domain_factor_net, centers=centers, domain_factor_cond=domain_factor_cond, feat_exp=feat_exp, out_dir=outdir, dom=data)

