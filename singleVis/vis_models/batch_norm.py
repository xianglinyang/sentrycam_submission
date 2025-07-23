import torch.nn as nn
from .base import BaseVisModel

'''
1. layer norm or batch norm
2. relu or leakyrelu
3. whether to add relu in the last layer?
'''

class BatchNormAE(BaseVisModel):

    def __init__(self, encoder_dims, decoder_dims):
        super(BatchNormAE, self).__init__()
  
        assert len(encoder_dims) > 1
        assert len(decoder_dims) > 1
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Build Encoder
        modules = list()
        for i in range(0, len(self.encoder_dims)-2):
            modules.append(
                nn.Sequential(
                nn.Linear(self.encoder_dims[i], self.encoder_dims[i+1]),
                nn.BatchNorm1d(self.encoder_dims[i+1]),
                nn.ReLU(True) 
                )
            )
        modules.append(nn.Linear(self.encoder_dims[-2], self.encoder_dims[-1]))
        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = list()
        for i in range(0, len(self.decoder_dims)-2):
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