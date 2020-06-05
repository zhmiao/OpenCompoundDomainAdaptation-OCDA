import torch
import torch.nn as nn
from torchvision.models.resnet import model_urls, BasicBlock, ResNet 
from .utils import register_model

@register_model('ResNet18')
class ResNet18(ResNet):

    num_channels = 3
    image_size = 224
    name = 'ResNet18'
    out_dim = 512 # dim of last feature layer

    def __init__(self, num_cls=5, weights_init=None):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_cls)
        self.criterion = nn.CrossEntropyLoss()

        if weights_init is not None:
            self.load(weights_init)

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)
