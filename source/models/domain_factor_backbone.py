import torch.nn as nn

from .utils import register_model


@register_model('DomainFactorBackbone')
class DomainFactorBackbone(nn.Module):

    def __init__(self):
        super(DomainFactorBackbone, self).__init__()
        self.num_channels = 3
        self.setup_net()
        
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

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        return x
