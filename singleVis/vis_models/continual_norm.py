'''
[ICLR'22] Continual Normalization: Rethinking Batch Normalization for Online Continual Learning
adapt from https://github.com/phquang/Continual-Normalization
'''
import torch
import torch.nn as nn
from .base import BaseVisModel

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F


class CN(_BatchNorm):
    """official implementation"""
    def __init__(self, num_features, G=8, eps = 1e-5, momentum = 0.1, affine=True):
        super(CN, self).__init__(num_features, eps, momentum, affine=affine)
        self.G = G

    def forward(self, input):
        out_gn = F.group_norm(input, self.G, None, None, self.eps)
        out = F.batch_norm(out_gn, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)
        return out

# TODO debug
class ContinualNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-5, momentum=0.1) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.momentum = momentum

        # Initialize parameters
        self.gamma = nn.Parameter(torch.ones(1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, num_features))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.ones(1, num_features))

    def forward(self, x):
        # group norm
        batch_size, num_features = x.size()
        num_channels_per_group = num_features // self.num_groups
        x = x.view(batch_size, self.num_groups, num_channels_per_group)

        # compute mean and variance per group
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, unbiased=False, keepdim=True)
        # normalize x within each group
        x_normalized = (x-mean)/ torch.sqrt(var+self.eps)
        # reshape back to the original shape
        x_normalized = x_normalized.view(batch_size, num_features)

        # batch norm
        if self.training:
            mean = x_normalized.mean(dim=0, keepdim=True)
            var = x_normalized.var(dim=0, unbiased=False, keepdim=True)
            self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*mean
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_normalized = (x_normalized-mean)/torch.sqrt(var+self.eps)

        # scale and shift using learned parameters
        out = self.gamma*x_normalized + self.beta
        return out


class ContinualNormAE(BaseVisModel):

    def __init__(self, encoder_dims, decoder_dims):
        super(ContinualNormAE, self).__init__()
  
        assert len(encoder_dims) > 1
        assert len(decoder_dims) > 1
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Build Encoder
        modules = list()
        for i in range(0, len(self.encoder_dims)-2):
            if i==0:
                modules.append(
                    nn.Sequential(
                    nn.Linear(self.encoder_dims[i], self.encoder_dims[i+1]),
                    CN(self.encoder_dims[i+1]),
                    nn.ReLU(True) 
                    )
                )
            else:
                modules.append(
                    nn.Sequential(
                    nn.Linear(self.encoder_dims[i], self.encoder_dims[i+1]),
                    nn.BatchNorm1d(self.encoder_dims[i+1]),
                    # CN(self.encoder_dims[i+1]),
                    nn.ReLU(True) 
                    )
                )
        modules.append(nn.Linear(self.encoder_dims[-2], self.encoder_dims[-1]))
        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = list()
        for i in range(0, len(self.decoder_dims)-2):
            if i == 0:
                modules.append(
                    nn.Sequential(
                        nn.Linear(self.decoder_dims[i], self.decoder_dims[i+1]),
                        CN(self.decoder_dims[i+1]),
                        nn.ReLU(True)
                    )
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(self.decoder_dims[i], self.decoder_dims[i+1]),
                        nn.BatchNorm1d(self.decoder_dims[i+1]),
                        nn.ReLU(True)
                    )
                )
        modules.append(nn.Linear(self.decoder_dims[-2], self.decoder_dims[-1]))
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, edge):
        embedding = self.encoder(edge)
        recon = self.decoder(embedding)
        return embedding, recon


# class ContinualNormAE(BaseVisModel):

#     def __init__(self, encoder_dims, decoder_dims):
#         super(ContinualNormAE, self).__init__()
  
#         assert len(encoder_dims) > 1
#         assert len(decoder_dims) > 1
#         self.encoder_dims = encoder_dims
#         self.decoder_dims = decoder_dims

#         # Build Encoder
#         modules = list()
#         # modules.append(
#         #     nn.Sequential(
#         #     nn.Linear(self.encoder_dims[0], self.encoder_dims[1]),
#         #     nn.InstanceNorm1d(self.encoder_dims[1]),
#         #     nn.ReLU(True) 
#         #     )
#         # )
#         for i in range(0, len(self.encoder_dims)-3):
#             modules.append(
#                 nn.Sequential(
#                 nn.Linear(self.encoder_dims[i], self.encoder_dims[i+1]),
#                 nn.InstanceNorm1d(self.encoder_dims[i+1]),
#                 nn.ReLU(True) 
#                 )
#             )
#         modules.append(
#             nn.Sequential(
#             nn.Linear(self.encoder_dims[-3], self.encoder_dims[-2]),
#             nn.BatchNorm1d(self.encoder_dims[-2]),
#             nn.ReLU(True) 
#             )
#         )
#         modules.append(nn.Linear(self.encoder_dims[-2], self.encoder_dims[-1]))
#         self.encoder = nn.Sequential(*modules)

#         # Build Decoder
#         modules = list()
#         modules.append(
#             nn.Sequential(
#                 nn.Linear(self.decoder_dims[0], self.decoder_dims[1]),
#                 nn.InstanceNorm1d(self.decoder_dims[1]),
#                 nn.ReLU(True)
#             )
#         )
#         for i in range(1, len(self.decoder_dims)-2):
#             modules.append(
#                 nn.Sequential(
#                     nn.Linear(self.decoder_dims[i], self.decoder_dims[i+1]),
#                     nn.InstanceNorm1d(self.decoder_dims[i+1]),
#                     nn.ReLU(True)
#                 )
#             )
#         # modules.append(
#         #     nn.Sequential(
#         #         nn.Linear(self.decoder_dims[-3], self.decoder_dims[-2]),
#         #         nn.InstanceNorm1d(self.decoder_dims[i+1]),
#         #         nn.ReLU(True)
#         #     )
#         # )
#         modules.append(nn.Linear(self.decoder_dims[-2], self.decoder_dims[-1]))
#         self.decoder = nn.Sequential(*modules)

#     def forward(self, edge_to, edge_from):
#         outputs = dict()
#         embedding_to = self.encoder(edge_to)
#         embedding_from = self.encoder(edge_from)
#         recon_to = self.decoder(embedding_to)
#         recon_from = self.decoder(embedding_from)
        
#         outputs["umap"] = (embedding_to, embedding_from)
#         outputs["recon"] = (recon_to, recon_from)

#         return outputs