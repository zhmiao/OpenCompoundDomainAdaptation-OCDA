import os
from os.path import join
from copy import deepcopy

# Import from torch
import torch
import torch.optim as optim

# Import from within Package 
from ..models.models import get_model
from ..data.data_loader import load_data_multi, get_dataset_multi
from ..data.sampler import DomainScheduledSampler

import pdb


def train_epoch(loader_src, loader_tgt, net, style_net, opt_net, opt_dis,
                opt_selector_content, opt_selector_style, opt_classifier, epoch, the=0.6, style_cond=0):
   
    log_interval = 10  # specifies how often to display
  
    N = len(loader_tgt.dataset)
    joint_loader = zip(loader_src, loader_tgt)

    net.train()
    style_net.eval()
   
    last_update = -1

    for batch_idx, ((data_s, _), (data_t, _)) in enumerate(joint_loader):
        
        if len(data_s) == 1 or len(data_t) == 1:  # BN protection
            continue

        # log basic dme train info
        info_str = "[Train Schedule Dme] Epoch: {} [{}/{} ({:.2f}%)]".format(epoch, batch_idx * len(data_t),
                                                                             N, 100 * batch_idx * len(data_t) / N)
   
        ########################
        # Setup data variables #
        ########################
        if torch.cuda.is_available():
            data_s = data_s.cuda()
            data_t = data_t.cuda()

        data_s.require_grad = False
        data_t.require_grad = False

        ##########################
        # Optimize discriminator #
        ##########################

        # extract and concat features
        score_s, x_s = net.src_net(data_s.clone())
        score_t, x_t = net.tgt_net(data_t.clone())

        ###########################
        # storing direct feature
        direct_feature = x_t.clone()

        # set up visual memory
        keys_memory = net.centroids.detach().clone()

        # computing memory feature by querying and associating visual memory
        values_memory = score_t.clone()
        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        if style_cond == 0:
            # computing concept selector
            concept_selector = net.fc_selector(x_t.clone()).tanh()
            x_t = direct_feature + concept_selector * memory_feature
        elif style_cond == 1:
            with torch.no_grad():
                style_ftr = style_net(data_t).detach()
            style_selector = net.style_selector(x_t).tanh()
            x_t = direct_feature + style_selector * style_ftr
        elif style_cond == 2:
            # computing concept selector
            concept_selector = net.fc_selector(x_t.clone()).tanh()
            with torch.no_grad():
                style_ftr = style_net(data_t.clone()).detach()
            style_selector = net.style_selector(x_t.clone()).tanh()
            x_t = direct_feature + concept_selector * memory_feature + 0.01 * style_selector * style_ftr
        elif style_cond == 3:
            with torch.no_grad():
                style_ftr = style_net(data_t.clone()).detach()
            domain_indicator = net.style_selector(style_ftr.clone()).tanh()
            x_t = direct_feature + domain_indicator * memory_feature
        elif style_cond == 4:
            # computing concept selector
            concept_selector = net.fc_selector(x_t.clone()).tanh()
            with torch.no_grad():
                style_ftr = style_net(data_t.clone()).detach()
            style_selector = net.style_selector(style_ftr.clone()).tanh()
            x_t = direct_feature + style_selector * concept_selector * memory_feature
        else:
            raise Exception("No such style_cond: {}".format(style_cond))

        # apply cosine norm classifier
        score_t = net.classifier(x_t.clone())
        ###########################

        f = torch.cat((score_s, score_t), 0)
        
        # predict with discriminator
        pred_concat = net.discriminator(f.clone())

        # prepare real and fake labels: source=1, target=0
        target_dom_s = torch.ones(len(data_s), requires_grad=False).long()
        target_dom_t = torch.zeros(len(data_t), requires_grad=False).long()
        label_concat = torch.cat((target_dom_s, target_dom_t), 0).cuda()

        # compute loss for disciminator
        loss_dis = net.gan_criterion(pred_concat.clone(), label_concat)

        # zero gradients for optimizer
        opt_dis.zero_grad()

        # loss backprop
        loss_dis.backward()

        # optimize discriminator
        opt_dis.step()

        # compute discriminator acc
        pred_dis = torch.squeeze(pred_concat.max(1)[1])
        acc = (pred_dis == label_concat).float().mean()
        
        # log discriminator update info
        info_str += " acc: {:0.1f} D: {:.3f}".format(acc.item()*100, loss_dis.item())

        ###########################
        # Optimize target network #
        ###########################

        # only update net if discriminator is strong
        if acc.item() > the:
            
            last_update = batch_idx
        
            # extract target features
            score_t, x_t = net.tgt_net(data_t.clone())

            ###########################
            # storing direct feature
            direct_feature = x_t.clone()

            # set up visual memory
            keys_memory = net.centroids.detach().clone()

            # computing memory feature by querying and associating visual memory
            values_memory = score_t.clone()
            values_memory = values_memory.softmax(dim=1)
            memory_feature = torch.matmul(values_memory, keys_memory)

            if style_cond == 0:
                # computing concept selector
                concept_selector = net.fc_selector(x_t.clone()).tanh()
                x_t = direct_feature + concept_selector * memory_feature
            elif style_cond == 1:
                with torch.no_grad():
                    style_ftr = style_net(data_t).detach()
                style_selector = net.style_selector(x_t).tanh()
                x_t = direct_feature + style_selector * style_ftr
            elif style_cond == 2:
                # computing concept selector
                concept_selector = net.fc_selector(x_t.clone()).tanh()
                with torch.no_grad():
                    style_ftr = style_net(data_t.clone()).detach()
                style_selector = net.style_selector(x_t.clone()).tanh()
                x_t = direct_feature + concept_selector * memory_feature + 0.01 * style_selector * style_ftr
            elif style_cond == 3:
                with torch.no_grad():
                    style_ftr = style_net(data_t).detach()
                domain_indicator = net.style_selector(style_ftr).tanh()
                x_t = direct_feature + domain_indicator * memory_feature
            elif style_cond == 4:
                # computing concept selector
                concept_selector = net.fc_selector(x_t.clone()).tanh()
                with torch.no_grad():
                    style_ftr = style_net(data_t.clone()).detach()
                style_selector = net.style_selector(style_ftr.clone()).tanh()
                x_t = direct_feature + style_selector * concept_selector * memory_feature
            else:
                raise Exception("No such style_cond: {}".format(style_cond))

            # apply cosine norm classifier
            score_t = net.classifier(x_t.clone())
            ###########################

            ###########################
            # predict with discriinator
            ###########################
            pred_tgt = net.discriminator(score_t)
            
            # create fake label
            label_tgt = torch.ones(pred_tgt.size(0), requires_grad=False).long().cuda()
            
            # compute loss for target network
            loss_gan_t = net.gan_criterion(pred_tgt, label_tgt)

            # zero out optimizer gradients
            opt_dis.zero_grad()
            opt_net.zero_grad()

            opt_selector_content.zero_grad()
            opt_classifier.zero_grad()

            if opt_selector_style:
                opt_selector_style.zero_grad()

            # loss backprop
            loss_gan_t.backward()

            # optimize tgt network
            opt_net.step()
            opt_selector_content.step()
            opt_classifier.step()
            if opt_selector_style:
                opt_selector_style.step()

            # log net update info
            info_str += " G: {:.3f}".format(loss_gan_t.item()) 

        ###########
        # Logging #
        ###########
        if batch_idx % log_interval == 0:
            print(info_str)

    return last_update


def train_scheduled_dme_multi(src, tgt, base_model, style_model, num_cls, tgt_list,
                              sort_idx, power, initial_ratio, schedule_strategy,
                              num_epoch=200, batch=128, datadir="", outdir="",
                              src_weights=None, style_weights=None, lr=1e-5, betas=(0.9, 0.999),
                              weight_decay=0, style_cond=0, centroids_path=None):

    """Main function for training DME."""

    ###########################
    # Setup cuda and networks #
    ###########################

    # setup cuda
    if torch.cuda.is_available():
        kwargs = {'num_workers': 8, 'pin_memory': True}
    else:
        kwargs = {}

    # setup network 
    net = get_model('DmeNet', model=base_model, num_cls=num_cls,
                    src_weights_init=src_weights,
                    use_style_selector=(style_cond != 0),
                    centroids_path=centroids_path)

    style_net = deepcopy(get_model('StyleNet', num_cls=num_cls,
                                   base_model=base_model, style_model=style_model,
                                   weights_init=style_weights, eval=True).style_net)

    style_net.eval()

    # print network and arguments
    print(net)
    print('Training Scheduled Dme {} model for {}->{}'.format(base_model, src, tgt))

    #######################################
    # Setup data for training and testing #
    #######################################
    train_src_data = load_data_multi(src, 'train', batch=batch,
                                     rootdir=join(datadir, src), num_channels=net.num_channels,
                                     image_size=net.image_size, download=False, kwargs=kwargs)

    train_tgt_set = get_dataset_multi(tgt_list, 'train', rootdir=datadir, num_channels=net.num_channels,
                                      image_size=net.image_size, download=False)

    ######################
    # Optimization setup #
    ######################
    opt_net = optim.Adam(net.tgt_net.parameters(), lr=lr, 
                         weight_decay=weight_decay, betas=betas)
    opt_dis = optim.Adam(net.discriminator.parameters(), lr=lr, 
                         weight_decay=weight_decay, betas=betas)
    opt_selector_content = optim.Adam(net.fc_selector.parameters(), lr=lr*0.1, 
                                      weight_decay=weight_decay, betas=betas)
    opt_classifier = optim.Adam(net.classifier.parameters(), lr=lr*0.1,
                                weight_decay=weight_decay, betas=betas)
    if style_cond != 0:
        opt_selector_style = optim.Adam(net.style_selector.parameters(), lr=lr*0.1,
                                        weight_decay=weight_decay, betas=betas)
    else:
        opt_selector_style = None

    #########
    # Train #
    #########
    # scheduled_ratio = lambda ep: (1. - initial_ratio) / ((num_epoch) ** power) * (ep ** power) + initial_ratio
    # scheduled_ratio = lambda ep: (1. - initial_ratio) / ((num_epoch + 30) ** power) * (ep ** power) + initial_ratio
    # scheduled_ratio = lambda ep: (1. - initial_ratio) / ((num_epoch + 30) ** power) * (ep ** power) + initial_ratio
    # Best
    # scheduled_ratio = lambda ep: (1. - initial_ratio) / ((num_epoch - 30) ** power) * (ep ** power) + initial_ratio
    scheduled_ratio = lambda ep: (1. - initial_ratio) / ((num_epoch - 30) ** power) * (ep ** power) + initial_ratio

    # train_tgt_loader = load_data_multi(tgt_list, 'train', batch=batch,
    #                                    rootdir=datadir, num_channels=net.num_channels,
    #                                    image_size=net.image_size, download=True, kwargs=kwargs)

    for epoch in range(num_epoch):

        # if epoch % 5 == 0:
        #     os.makedirs(outdir, exist_ok=True)
        #     outfile = join(outdir, 'scheduled_{:s}_net_{:s}_{:s}_ep_{}.pth'.format(
        #                    base_model, src, tgt, epoch))
        #     print('Saving to', outfile)
        #     net.save(outfile)

        # Calculate current domain ratio
        ratio = scheduled_ratio(epoch)

        actual_lr = ratio * lr

        for param_group in opt_net.param_groups:
            param_group['lr'] = actual_lr
        for param_group in opt_dis.param_groups:
            param_group['lr'] = actual_lr
        for param_group in opt_selector_content.param_groups:
            param_group['lr'] = actual_lr * 0.1
        for param_group in opt_classifier.param_groups:
            param_group['lr'] = actual_lr * 0.1
        if style_cond != 0:
            for param_group in opt_net.param_groups:
                param_group['lr'] = actual_lr * 0.1

        if ratio < 1:
            # Use sampler for data loading
            print('Epoch: {}, using sampler'.format(epoch))
            sampler = DomainScheduledSampler(train_tgt_set, sort_idx, ratio,
                                             initial_ratio, schedule_strategy, seed=epoch)
            train_tgt_loader = torch.utils.data.DataLoader(train_tgt_set, batch_size=batch,
                                                           shuffle=False, sampler=sampler, **kwargs)
        else:
            print('Epoch: {}, using default'.format(epoch))
            train_tgt_loader = torch.utils.data.DataLoader(train_tgt_set, batch_size=batch, shuffle=True, **kwargs)

        err = train_epoch(train_src_data, train_tgt_loader, net, style_net, opt_net, opt_dis, opt_selector_content,
                          opt_selector_style, opt_classifier, epoch, style_cond=style_cond)
        # if err == -1:
        #     print("No suitable discriminator")
        #     break
       
    ##############
    # Save Model #
    ##############
    os.makedirs(outdir, exist_ok=True)
    outfile = join(outdir, 'scheduled_{:s}_net_{:s}_{:s}.pth'.format(
        base_model, src, tgt))
    print('Saving to', outfile)
    net.save(outfile)

