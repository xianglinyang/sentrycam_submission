from torch import nn
from abc import abstractmethod

class BaseVisModel(nn.Module):

    def __init__(self) -> None:
        super(BaseVisModel, self).__init__()
    
    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    # Calculate the size of trainable weights
    def cal_param_size(self):
        total_bytes = 0
        for param in self.parameters():
            if param.requires_grad:
                num_bytes = param.element_size() * param.nelement()
                total_bytes += num_bytes
        size = total_bytes / (1024 * 1024)
        print(f"Total Trainable Parameters Size: {size:.2f} MB")
        return size


class AE(BaseVisModel):

    def __init__(self, encoder_dims, decoder_dims):
        super(AE, self).__init__()
  
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
                    nn.ReLU(True)
                )
                
            )
        modules.append(nn.Linear(self.decoder_dims[-2], self.decoder_dims[-1]))
        self.decoder = nn.Sequential(*modules)

    def forward(self, edge_to, edge_from):
        outputs = dict()
        embedding_to = self.encoder(edge_to)
        embedding_from = self.encoder(edge_from)
        recon_to = self.decoder(embedding_to)
        recon_from = self.decoder(embedding_from)
        
        outputs["umap"] = (embedding_to, embedding_from)
        outputs["recon"] = (recon_to, recon_from)

        return outputs
    