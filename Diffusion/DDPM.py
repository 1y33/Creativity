import torch
import torch.nn as nn
import torch.nn.functional as F
import VAEs.VAE as VAE


class DiffusionModel(nn.Module):
    def __init__(self,image_size,reduction):
        super().__init__()
        
        self.image_size = image_size
        self.reduction = reduction
        self.latent_space = image_size // 2**reduction
        
        self.vae = self._generateVAE(reduction)
        
        
        self.setParams()        
    def _generateVAE(self,reduction):
        encoder,decoder = VAE.generateBackbones(reduction)
        vae = VAE.VAE(encoder,decoder)
        return vae
    
    def setParams(self,learning_rate):
        self.learning_rate = learning_rate
        self.vae.setParams(learing_rate= self.learning_rate)
        
        
        
        
        