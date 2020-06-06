from copy import deepcopy
import torch
import torch.nn as nn
from .utils import register_model, get_model
from . import cos_norm_classifier


@register_model('DomainFactorNet')
class DomainFactorNet(nn.Module):

    """Defines a Dynamic Meta-Embedding Network."""

    def __init__(self, num_cls=10, base_model='LeNet',
                 domain_factor_model='LeNet', content_weights_init=None, weights_init=None, eval=False, feat_dim=512):

        super(DomainFactorNet, self).__init__()
        self.name = 'DomainFactorNet'
        self.base_model = base_model
        self.domain_factor_model = domain_factor_model
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.CrossEntropyLoss()
        self.rec_criterion = nn.SmoothL1Loss()
      
        self.setup_net()
        if weights_init is not None:
            self.load(weights_init, eval=eval)
        elif content_weights_init is not None:
            self.load_content_net(content_weights_init)
        else:
            raise Exception('MannNet must be initialized with weights.')

    def forward(self, x):
        pass

    def setup_net(self):
        """Setup source, target and discriminator networks."""
        self.tgt_net = get_model(self.base_model, num_cls=self.num_cls, feat_dim=self.feat_dim)
        # self.domain_factor_net = get_model(self.content_model, num_cls=self.num_cls)
        self.domain_factor_net = get_model(self.domain_factor_model)  # cls: dummy

        # self.discriminator_cls = nn.Sequential(  # classifier
        #         nn.Linear(self.feat_dim, 500),  # feature dim: 512 (hard code)
        #         nn.ReLU(),
        #         nn.Linear(500, 500),
        #         nn.ReLU(),
        #         nn.Linear(500, self.num_cls),
        #         )

        self.discriminator_cls = cos_norm_classifier.create_model(512, self.num_cls)

        self.decoder = Decoder(input_dim=1024)

        self.image_size = self.tgt_net.image_size
        self.num_channels = self.tgt_net.num_channels

    def load(self, init_path, eval=False):

        """
        Load weights from pretrained tgt model
        and initialize DomainFactorNet from pretrained tgr model.
        """

        net_init_dict = torch.load(init_path)

        print('Load weights.')
        self.load_state_dict(net_init_dict, strict=False)
        load_keys = set(net_init_dict.keys())
        self_keys = set(self.state_dict().keys())
        missing_keys = self_keys - load_keys
        unused_keys = load_keys - self_keys
        print("missing keys: {}".format(sorted(list(missing_keys))))
        print("unused_keys: {}".format(sorted(list(unused_keys))))

        if not eval:
            print('Initialize domain_factor net weights from tgt net.')
            tgt_weights = deepcopy(self.tgt_net.state_dict())
            self.domain_factor_net.load_state_dict(tgt_weights, strict=False)
            load_keys_sty = set(tgt_weights.keys())
            self_keys_sty = set(self.domain_factor_net.state_dict().keys())
            missing_keys_sty = self_keys_sty - load_keys_sty
            unused_keys_sty = load_keys_sty - self_keys_sty
            print("missing keys: {}".format(sorted(list(missing_keys_sty))))
            print("unused_keys: {}".format(sorted(list(unused_keys_sty))))

        print('Initialize class discriminator weights from tgt net.')
        self.discriminator_cls.weight.data = self.tgt_net.state_dict()['classifier.weight'].data.clone()


    def load_content_net(self, init_path):
        self.tgt_net.load(init_path)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def save_domain_factor_net(self, out_path):
        torch.save(self.domain_factor_net.state_dict(), out_path)


class Decoder(nn.Module):

    def __init__(self, input_dim=1024):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4096))

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 3, kernel_size=1))

    def forward(self, x):
        assert x.size(1) == self.input_dim
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.decoder(x)
        return x
