import torch
import torch.nn as nn
from .utils import register_model
from .utils import init_weights
from . import cos_norm_classifier, disc_centroids_loss

import torchvision.models as models


class MemoryNet(nn.Module):

    num_channels = 3
    image_size = 32
    name = 'MemoryNet'

    "Basic class which does classification."
    def __init__(self, num_cls=10, weights_init=None, feat_dim=512):
        super(MemoryNet, self).__init__()
        self.num_cls = num_cls
        self.setup_net()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_ctr = disc_centroids_loss.create_loss(feat_dim, num_cls)
        if weights_init is not None:
            self.load(weights_init)
        else:
            init_weights(self)

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

@register_model('AMN')
class AMNClassifier(MemoryNet):

    """Classifier used for SVHN source experiment"""

    num_channels = 3
    image_size = 32
    name = 'AMN'
    out_dim = 512 # dim of last feature layer

    def setup_net(self):
        self.conv_params = nn.Sequential (
                nn.Conv2d(self.num_channels, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
    
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512),
                )

        self.classifier = cos_norm_classifier.create_model(512, self.num_cls)

    def forward(self, x, with_ft=True):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)

        feat = x.clone()
        score = self.classifier(x)

        if with_ft:
            return score, feat
        else:
            return score

@register_model('AMN_res')
class AMNClassifier_res(MemoryNet):

    """Classifier used for face source experiment"""

    num_channels = 3
    image_size = 224
    name = 'AMN_res'
    out_dim = 512  # dim of last feature layer

    def setup_net(self):
        resnet18 = models.resnet18(pretrained=False)
        modules_resnet18 = list(resnet18.children())[:-1]
        self.feat_model = nn.Sequential(*modules_resnet18)
        self.classifier = cos_norm_classifier.create_model(512, self.num_cls)

    def forward(self, x, with_ft=True):
        x = self.feat_model(x)
        x = torch.squeeze(x)

        feat = x.clone()
        score = self.classifier(x)

        if with_ft:
            return score, feat
        else:
            return score
