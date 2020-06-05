import torch

from ..data.data_loader import load_data_multi
from ..models.models import get_model

def test(loader, net):

    # import pdb
    # pdb.set_trace()

    net.eval()
    test_loss = 0
    correct = 0
   
    N = len(loader.dataset)
    for idx, (data, target) in enumerate(loader):
        
        # setup data and target #
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        data.require_grad = False
        target.require_grad = False
        
        # forward pass
        score = net(data)
        
        # compute loss
        test_loss += net.criterion(score, target).item()
        
        # compute predictions and true positive count
        _, pred = torch.max(score, 1) # get the index of the max log-probability
        correct += (pred == target).cpu().sum().item()
        
    test_loss /= len(loader) # loss function already averages over batch size
    print('[Evaluate] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, N, 100. * correct / N))


def load_and_test_net_reg(data, datadir, weights, model, num_cls, batch, 
        dset='test', base_model=None):
    
    # Setup GPU Usage
    if torch.cuda.is_available(): 
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    # Eval tgt from AddaNet or TaskNet model #
    if model == 'AddaNet':
        net = get_model(model, num_cls=num_cls, weights_init=weights, 
                        model=base_model)
        net = net.tgt_net
    else:
        net = get_model(model, num_cls=num_cls, weights_init=weights)

    # Load data
    test_data = load_data_multi(data, dset, batch=batch, 
        rootdir=datadir, num_channels=net.num_channels, 
        image_size=net.image_size, download=True, kwargs=kwargs)

    if test_data is None:
        print('skipping test')
    else:
        test(test_data, net)
