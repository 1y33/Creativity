import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import TransformerLayer

class ViTEncoder(nn.Module):
    def __init__(self,image_size, patch_size,n_layers,n_heads,d_embed):
        super().__init__()
        assert image_size % patch_size == 0, 'image size must be divisible by patch size'
        
        self.num_patches = (image_size // patch_size) ** 2
        
        self.PatchEmbedding = nn.Conv2d(3,d_embed,kernel_size=patch_size,stride=patch_size)
        self.PositionalEmbedding = nn.Parameter(torch.randn(1,self.num_patches,d_embed),requires_grad=True)
        self.modules = nn.ModuleList([TransformerLayer(d_embed,n_heads) for _ in range(n_layers)])
        
    def forward(self,x):
        x = self.PatchEmbedding(x) # batchsize, d_embed, num_patches, num_patches
        x = x.flatten(2).transpose(1,2) # batchsize, numpatchs, d_embed
        x = x + self.PositionalEmbedding
        for module in self.modules:
            x = module(x)
        
        return x
    
    
