import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import SelfAttention

class VAEResicualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.group_norm_1  = nn.GroupNorm(32,in_channels)
        self.conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.dropout_1 = nn.Dropout(0.3)
        
        self.group_norm_2  = nn.GroupNorm(32,out_channels)
        self.conv_2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.dropout_2 = nn.Dropout(0.3)
        
        if in_channels == out_channels:
            self.residual_block = nn.Identity()
        else:
            self.residual_block = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)

    def forward(self,x):
        residue = x

        x = F.silu(self.group_norm_1(x))
        x = self.conv_1(x)
        x = self.dropout_1(x)

        x = F.silu(self.group_norm_2(x))
        x = self.conv_2(x)
        x = self.dropout_2(x)

        return x + self.residual_block(residue)

class VAEAttentionBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()

        self.group_norm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(channels,n_heads=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        residual = x

        n,c,h,w = x.shape

        x = x.view(n,c,h*w)
        x = x.transpose(-1,-2)

        x = self.attention(x)
        x = x.transpose(-1,-2)
        x = x.view(n,c,h,w)
        x = self.group_norm(x)
        x = self.dropout(x)
        
        return x + residual

class VAEEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
    
    def forward(self,x,train=False):
        x = self.backbone(x)
        mean,log_var =  torch.chunk(x,chunk=2,dim=1)
        log_var = torch.clamp(log_var,-30,30)
        variance = log_var.exp()
        stdev = variance.sqrt()

        noise = torch.randn_like(x)

        x = mean + stdev*noise
        x *= 0.18215
        
        if train==True:
            return x,mean,log_var
        else:
            return x
    
class VAEDecoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
    
    def forward(self,x):
        x /= 0.18215
        x = self.backbone(x)

        return x
    
class VAE(nn.Module):
    def __init__(self,encoder_backbone,decoder_backbone):
        super().__init__()
        
        self.encoder = VAEEncoder(encoder_backbone)
        self.decoder = VAEDecoder(decoder_backbone)

    def forward(self,x,train=True):
        if train == False:
            x = self.encoder(x,train)
            x = self.decoder(x)
            return x
        else:
            x,mean,log_var = self.encoder(x,train)
            x = self.decoder(x)
            return x,mean,log_var
    
    def calculateLoss(self,reconstructed_x,x,mean,log_var):
        reconstruction_loss = F.mse_loss(reconstructed_x,x,reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        total_loss = reconstruction_loss + kl_loss

        return total_loss,reconstruction_loss,kl_loss
