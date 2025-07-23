'''https://github.com/AntixK/PyTorch-VAE'''
from .base import AE
from .batch_norm import BatchNormAE
from .layer_norm import LayerNormAE
from .instance_norm import InstanceNormAE
from .group_norm import GroupNormAE
from .continual_norm import ContinualNormAE
from .vis_norm import VisNormAE
vis_models = {
    'AE': AE,
    'bnAE': BatchNormAE,
    'lnAE': LayerNormAE,
    'inAE': InstanceNormAE,
    'gnAE': GroupNormAE,
    'cnAE': ContinualNormAE,
    'visAE': VisNormAE
}
