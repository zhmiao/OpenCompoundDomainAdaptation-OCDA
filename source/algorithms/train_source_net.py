import os
from os.path import join

import torch
import torch.optim as optim

from ..models.utils import get_model
from ..data.utils import load_data_multi


def train_epoch(loader, net, opt_net, epoch):

    log_interval = 10  # specifies how often to display

    net.train()

    for batch_idx, (data, target) in enumerate(loader):

        # make data variables
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        data.require_grad = False
        target.require_grad = False
        
        # zero out gradients
        opt_net.zero_grad()
       
        # forward pass
        score, x = net(data.clone())
        loss_cls = net.criterion_cls(score.clone(), target)
        loss_ctr = net.criterion_ctr(x.clone(), target)
        loss = loss_cls + 0.1 * loss_ctr

        # backward pass
        loss.backward()
        
        # optimize classifier and representation
        opt_net.step()
       
        # Logging
        if batch_idx % log_interval == 0:
            print('[Train Source AMN] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch, batch_idx * len(data), len(loader.dataset),
                          100. * batch_idx / len(loader), loss.item()), end="")
            _, pred = torch.max(score, 1)
            correct = (pred == target).cpu().sum().item()
            acc = correct / len(pred) * 100.0
            print('  Acc: {:.2f}'.format(acc))


def train_source(args):

    """Train a classification net and evaluate on test set."""

    data = args.src
    datadir = join(args.datadir, args.src)
    model = args.base_model
    num_cls = args.num_cls
    outdir = args.outdir_source
    num_epoch = args.src_num_epoch
    batch = args.batch
    lr = args.src_lr
    betas = tuple(args.betas)
    weight_decay = args.weight_decay
    feat_dim = 512

    # Setup GPU Usage
    if torch.cuda.is_available(): 
        kwargs = {'num_workers': 8, 'pin_memory': True}
    else:
        kwargs = {}

    ############
    # Load Net #
    ############
    net = get_model(model, num_cls=num_cls, feat_dim=feat_dim)
    print('-------Training net--------')
    print(net)

    ############################
    # Load train and test data # 
    ############################
    train_data = load_data_multi(data, 'train', batch=batch, 
                                 rootdir=datadir, num_channels=net.num_channels,
                                 image_size=net.image_size, download=True, kwargs=kwargs)
    
    ###################
    # Setup Optimizer #
    ###################
    opt_net = optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
    #########
    # Train #
    #########
    print('Training {} model for {}'.format(model, data))
    for epoch in range(num_epoch):
        train_epoch(train_data, net, opt_net, epoch)
    
    ############
    # Save net #
    ############
    os.makedirs(outdir, exist_ok=True)
    outfile = join(outdir, '{:s}_net_{:s}.pth'.format(model, data))
    print('Saving to', outfile)
    net.save(outfile)

    return net
