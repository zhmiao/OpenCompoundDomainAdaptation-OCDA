import numpy as np
import torch
import torch.nn as nn
from .utils import register_model, get_model
from . import cos_norm_classifier


@register_model('MannNet')
class MannNet(nn.Module):

    """Defines a Dynamic Meta-Embedding Network."""

    def __init__(self, num_cls=10, model='LeNet', src_weights_init=None,
                 weights_init=None, use_domain_factor_selector=False, centroids_path=None, feat_dim=512):

        super(MannNet, self).__init__()

        self.name = 'MannNet'
        self.base_model = model
        self.num_cls = num_cls
        self.feat_dim = feat_dim
        self.use_domain_factor_selector = use_domain_factor_selector
        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.CrossEntropyLoss()

        self.centroids = torch.from_numpy(np.load(centroids_path)).float().cuda()
        assert self.centroids is not None
        self.centroids.requires_grad = False

        self.setup_net()

        if weights_init is not None:
            self.load(weights_init)
        elif src_weights_init is not None:
            self.load_src_net(src_weights_init)
        else:
            raise Exception('MannNet must be initialized with weights.')

    def forward(self, x_s, x_t):

        """Pass source and target images through their respective networks."""

        score_s, x_s = self.src_net(x_s, with_ft=True)
        score_t, x_t = self.tgt_net(x_t, with_ft=True)

        if self.discrim_feat:
            d_s = self.discriminator(x_s.clone())
            d_t = self.discriminator(x_t.clone())
        else:
            d_s = self.discriminator(score_s.clone())
            d_t = self.discriminator(score_t.clone())

        return score_s, score_t, d_s, d_t

    def setup_net(self):

        """Setup source, target and discriminator networks."""
        self.src_net = get_model(self.base_model, num_cls=self.num_cls, feat_dim=self.feat_dim)
        self.tgt_net = get_model(self.base_model, num_cls=self.num_cls, feat_dim=self.feat_dim)

        input_dim = self.num_cls

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            )

        self.fc_selector = nn.Linear(self.feat_dim, self.feat_dim)

        if self.use_domain_factor_selector:
            self.domain_factor_selector = nn.Linear(self.feat_dim, self.feat_dim)

        self.classifier = cos_norm_classifier.create_model(self.feat_dim, self.num_cls)

        self.image_size = self.src_net.image_size
        self.num_channels = self.src_net.num_channels

    def load(self, init_path):
        """Loads full src and tgt models."""
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def load_src_net(self, init_path):
        """Initialize source and target with source weights."""
        self.src_net.load(init_path)
        self.tgt_net.load(init_path)

        net_init_dict = torch.load(init_path)
        classifier_weights = net_init_dict['classifier.weight']

        self.classifier.weight.data = classifier_weights.data.clone()

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def save_tgt_net(self, out_path):
        torch.save(self.tgt_net.state_dict(), out_path)

