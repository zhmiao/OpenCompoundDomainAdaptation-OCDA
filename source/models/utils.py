import torch.nn as nn
from torch.nn import init
import torch

models = {}
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def get_model(name, **args):

    net = models[name](**args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def init_weights(obj):
    for m in obj.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()

