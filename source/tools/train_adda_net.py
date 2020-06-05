import os
from os.path import join

# Import from torch
import torch
import torch.optim as optim

# Import from within Package 
from ..models.models import get_model
from ..data.data_loader import load_data_multi

import pdb

def train_epoch(loader_src, loader_tgt, net, opt_net, opt_dis, epoch, the = 0.6):
   
    log_interval = 10 # specifies how often to display
  
    N = min(len(loader_src.dataset), len(loader_tgt.dataset)) 
    joint_loader = zip(loader_src, loader_tgt)
      
    net.train()
   
    last_update = -1
    for batch_idx, ((data_s, _), (data_t, _)) in enumerate(joint_loader):
        
        # log basic adda train info
        info_str = "[Train Adda] Epoch: {} [{}/{} ({:.2f}%)]".format(
            epoch, batch_idx*len(data_t), N, 100 * batch_idx / N)
   
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

        # zero gradients for optimizer
        opt_dis.zero_grad()

        # extract and concat features
        score_s = net.src_net(data_s)
        score_t = net.tgt_net(data_t)
        f = torch.cat((score_s, score_t), 0)
        
        # predict with discriminator
        pred_concat = net.discriminator(f)

        # prepare real and fake labels: source=1, target=0
        target_dom_s = torch.ones(len(data_s), requires_grad=False).long()
        target_dom_t = torch.zeros(len(data_t), requires_grad=False).long()
        label_concat = torch.cat((target_dom_s, target_dom_t), 0).cuda()

        # compute loss for disciminator
        loss_dis = net.gan_criterion(pred_concat, label_concat)
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
        
            # zero out optimizer gradients
            opt_dis.zero_grad()
            opt_net.zero_grad()

            # extract target features
            score_t = net.tgt_net(data_t)

            # predict with discriinator
            pred_tgt = net.discriminator(score_t)
            
            # create fake label
            label_tgt = torch.ones(pred_tgt.size(0), requires_grad=False).long().cuda()
            
            # compute loss for target network
            loss_gan_t = net.gan_criterion(pred_tgt, label_tgt) 
            loss_gan_t.backward()

            # optimize tgt network
            opt_net.step()

            # log net update info
            info_str += " G: {:.3f}".format(loss_gan_t.item()) 

        ###########
        # Logging #
        ###########
        if batch_idx % log_interval == 0:
            print(info_str)

    return last_update

def train_adda_multi(src, tgt, model, num_cls, tgt_list, num_epoch=200,
        batch=128, datadir="", outdir="", 
        src_weights=None, weights=None, lr=1e-5, betas=(0.9,0.999),
        weight_decay=0):

    """Main function for training ADDA."""

    ###########################
    # Setup cuda and networks #
    ###########################

    # setup cuda
    if torch.cuda.is_available():
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    # setup network 
    net = get_model('AddaNet', model=model, num_cls=num_cls,
            src_weights_init=src_weights)
    
    # print network and arguments
    print(net)
    print('Training Adda {} model for {}->{}'.format(model, src, tgt))

    #######################################
    # Setup data for training and testing #
    #######################################

    train_src_data = load_data_multi(src, 'train', batch=batch, 
        rootdir=join(datadir, src), num_channels=net.num_channels, 
        image_size=net.image_size, download=True, kwargs=kwargs)

    train_tgt_data = load_data_multi(tgt_list, 'train', batch=batch, 
        rootdir=datadir, num_channels=net.num_channels, 
        image_size=net.image_size, download=True, kwargs=kwargs)

    ######################
    # Optimization setup #
    ######################
    opt_net = optim.Adam(net.tgt_net.parameters(), lr=lr, 
            weight_decay=weight_decay, betas=betas)
    opt_dis = optim.Adam(net.discriminator.parameters(), lr=lr, 
            weight_decay=weight_decay, betas=betas)

    ##############
    # Train Adda #
    ##############
    for epoch in range(num_epoch):
        err = train_epoch(train_src_data, train_tgt_data, net, opt_net, opt_dis, epoch) 
        if err == -1:
            print("No suitable discriminator")
            break
       
    ##############
    # Save Model #
    ##############
    os.makedirs(outdir, exist_ok=True)
    outfile = join(outdir, 'adda_{:s}_net_{:s}_{:s}.pth'.format(
        model, src, tgt))
    print('Saving to', outfile)
    net.save(outfile)

