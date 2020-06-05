import os
from os.path import join
import collections

# Import from torch
import torch
import torch.optim as optim

# Import from within Package 
from ..models.models import get_model
from ..data.data_loader import load_data_multi

import pdb


def soft_cross_entropy(input, target, size_average=True):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def train_epoch(loader_src, loader_tgt, net, opt_style, opt_decoder, opt_dis_cls, epoch,
                gamma_dispell, gamma_rec, num_cls, fake_label_type):
   
    log_interval = 10  # specifies how often to display
  
    N = min(len(loader_src.dataset), len(loader_tgt.dataset)) 
    joint_loader = zip(loader_src, loader_tgt)

    # Only make discriminator trainable
    net.discriminator_cls.train()
    net.style_net.eval()
    # net.style_net.train()
    net.decoder.eval()
    net.tgt_net.eval()

    last_update = -1
    for batch_idx, ((data_s, cls_s_gt), (data_t, cls_t_gt)) in enumerate(joint_loader):
        
        # log basic dme train info
        info_str = "[Train Style Net] Epoch: {} [{}/{} ({:.2f}%)]".format(epoch, batch_idx*len(data_t),
                                                                          N, 100 * batch_idx / N)
   
        ########################
        # Setup data variables #
        ########################
        if torch.cuda.is_available():
            data_s = data_s.cuda()
            data_t = data_t.cuda()

        data_s.require_grad = False
        data_t.require_grad = False

        ##########################
        # Optimize Discriminator #
        ##########################

        # extract features and logits
        data_cat = torch.cat((data_s, data_t), dim=0).detach()

        with torch.no_grad():  # content branch

            _, content_ftr_s = net.tgt_net(data_s.clone())
            content_ftr_s = content_ftr_s.detach()

            logit_t_pseudo, content_ftr_t = net.tgt_net(data_t.clone())
            logit_t_pseudo = logit_t_pseudo.detach()
            content_ftr_t = content_ftr_t.detach()

            style_ftr = net.style_net(data_cat.clone())  # style branch
            style_ftr = style_ftr.detach()

        # predict classes with discriminator using style feature
        pred_cls_from_style = net.discriminator_cls(style_ftr.clone())

        # prepare class labels
        cls_t_pseudo = logit_t_pseudo.argmax(dim=1)
        pseudo_acc = (cls_t_pseudo == cls_t_gt.cuda()).float().mean()  # acc of pseudo label
        info_str += " pseudo_acc: {:0.1f}".format(pseudo_acc.item() * 100)
        cls_real = torch.cat((cls_s_gt.cuda(), cls_t_pseudo), dim=0).cuda()  # real

        # compute loss for class disciminator
        loss_dis_cls = net.gan_criterion(pred_cls_from_style, cls_real)

        # zero gradients for optimizer
        opt_dis_cls.zero_grad()
        # loss backprop
        loss_dis_cls.backward()
        # optimize discriminator
        opt_dis_cls.step()

        # compute discriminator acc
        pred_dis_cls = torch.squeeze(pred_cls_from_style.argmax(1))
        acc_cls = (pred_dis_cls == cls_real).float().mean()
        
        # log discriminator update info
        info_str += " D_acc: {:0.1f} D_loss: {:.3f}".format(acc_cls.item()*100, loss_dis_cls.item())

        ##########################
        # Optimize Style Network #
        ##########################

        if acc_cls.item() > 0.3:

            # Make style net trainable
            net.discriminator_cls.eval()
            # net.discriminator_cls.train()
            net.style_net.train()
            net.decoder.train()

            # update style net
            last_update = batch_idx

            ###############
            # GAN loss - Style should not include class information
            # Calculate styles again and predict classes with it
            style_ftr = net.style_net(data_cat.clone())
            pred_cls_from_style = net.discriminator_cls(style_ftr.clone())

            # Calculate loss using random class labels
            if fake_label_type == 'random':
                cls_fake = torch.randint(0, num_cls, (cls_real.size(0),)).long().cuda()
                loss_gan_style = net.gan_criterion(pred_cls_from_style, cls_fake)
            elif fake_label_type == 'uniform':
                cls_fake = torch.ones((cls_real.size(0), num_cls), dtype=torch.float32).cuda() * 1. / num_cls
                loss_gan_style = soft_cross_entropy(pred_cls_from_style, cls_fake)
            else:
                raise Exception("No such fake_label_type: {}".format(fake_label_type))

            ###############
            # reconstruction loss - However, style should be able to help reconstruct the data into domain specific appearences

            # Concate source and target contents
            cls_ftr = torch.cat((content_ftr_s, content_ftr_t), 0).detach()
            # Concate contents and styles of each sample and feed into decoder
            combined_ftr = torch.cat((cls_ftr, style_ftr), dim=1)

            data_rec = net.decoder(combined_ftr)

            # Calculate reconstruction loss based on the decoder outputs
            loss_rec = net.rec_criterion(data_rec, data_cat)

            loss = gamma_dispell * loss_gan_style + gamma_rec * loss_rec

            opt_dis_cls.zero_grad()
            opt_style.zero_grad()
            opt_decoder.zero_grad()

            loss.backward()

            opt_style.step()
            opt_decoder.step()

            info_str += " G_loss: {:.3f}".format(loss_gan_style.item())
            info_str += " R_loss: {:.3f}".format(loss_rec.item())

        ###########
        # Logging #
        ###########
        if batch_idx % log_interval == 0:
            print(info_str)

    return last_update


def train_style_multi(src, tgt, base_model, style_model, num_cls, tgt_list, num_epoch=200,
                      batch=128, datadir="", outdir="",
                      dme_weights=None, lr=1e-5, betas=(0.9,0.999),
                      weight_decay=0, gamma_dispell=1., gamma_rec=10., fake_label_type='random'):

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
    net = get_model('StyleNet', num_cls=num_cls,
                    base_model=base_model, style_model=style_model,
                    weights_init=dme_weights)
    
    # print network and arguments
    print(net)
    print('Training style {} model for {}->{}'.format(style_model, src, tgt))

    #######################################
    # Setup data for training and testing #
    #######################################

    train_src_data = load_data_multi(src, 'train', batch=batch, 
                                     rootdir=join(datadir, src), num_channels=net.num_channels,
                                     image_size=net.image_size, download=False, kwargs=kwargs)

    train_tgt_data = load_data_multi(tgt_list, 'train', batch=batch, 
                                     rootdir=datadir, num_channels=net.num_channels,
                                     image_size=net.image_size, download=False, kwargs=kwargs)

    ######################
    # Optimization setup #
    ######################
    opt_style = optim.Adam(net.style_net.parameters(),
                           lr=lr, weight_decay=weight_decay, betas=betas)
    opt_decoder = optim.Adam(net.decoder.parameters(),
                           lr=lr, weight_decay=weight_decay, betas=betas)
    opt_dis_cls = optim.Adam(net.discriminator_cls.parameters(), lr=lr,
                             weight_decay=weight_decay, betas=betas)

    ##############
    # Train Dme #
    ##############
    for epoch in range(num_epoch):

        if epoch % 5 == 0:
            os.makedirs(outdir, exist_ok=True)
            outfile = join(outdir, 'StyleNet_{:s}_net_{:s}_{:s}_ep_{}.pth'.format(
                style_model, src, tgt, epoch))
            print('Saving to', outfile)
            net.save(outfile)

        err = train_epoch(train_src_data, train_tgt_data, net, opt_style, opt_decoder, opt_dis_cls,
                          epoch, gamma_dispell, gamma_rec, num_cls, fake_label_type)

        # if err == -1:
        #     print("No suitable discriminator")
        #     break

    ######################
    # Save Total Weights #
    ######################
    os.makedirs(outdir, exist_ok=True)
    outfile = join(outdir, 'StyleNet_{:s}_net_{:s}_{:s}.pth'.format(
        style_model, src, tgt))
    print('Saving to', outfile)
    net.save(outfile)

