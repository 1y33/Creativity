import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self,d_in,d_out,d_hidden=None,activation=F.silu):
        super().__init__()
        
        if d_hidden == None:
            d_hidden = d_in * 4
    
        self.fc_1 = nn.Linear(d_in,d_hidden)
        self.fc_2 = nn.Linear(d_hidden,d_out)
        self.activation = activation
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self,d_embed,n_heads,in_proj_bias=True,out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed,3*d_embed,bias = in_proj_bias)
        self.out_proj = nn.Linear(3*d_embed,d_embed,bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads

    def forward(self,x:torch.Tensor,causal_mask=False)->torch.Tensor:
        batch,seq_len,d_embed = x.shape

        interim_shape = (batch,seq_len,self.n_heads,self.d_head)
        
        q,k,v = self.in_proj(x).chunk(3,dim=-1)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        
        weight = q @ k.transpose(-1,-2)

        if causal_mask:
            mask = torch.ones_like(weight,dtype=torch.bool).triu(1)
            weight.mask_fill_(mask,-torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)

        output = weight @ v
        output = output.transpose(1,2).reshape(batch,seq_len,d_embed)
        output = self.out_proj(output)

        return output

class CrossAttention(nn.Module):
    def __init__(self,d_embed,d_cross,n_heads,in_proj_bias=True,out_proj_bias=True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed,d_embed,bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross,d_embed,bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross,d_embed,bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads

    def forward(self,x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        batch,seq_len,d_embed = x.shape
        interim_shape = (batch,-1,self.n_heads,self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        
        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)

        output = weight @ v
        output = output.transpose(1,2).contiguous().view(batch,seq_len,d_embed)
        output = self.out_proj(output)
        
        return output

class TransformerLayer(nn.Module):
    def __init__(self, d_embed,n_heads,causal_mask=False):
        super().__init__()
        
        self.attention = SelfAttention(d_embed,n_heads)
        self.causal_mask = causal_mask 
        
        self.norm_1 = nn.LayerNorm(d_embed,eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_embed,eps=1e-6)
        
        self.ffn = MLP(d_embed,d_embed)
        
    def forward(self,x):
        residue = x
        x = self.norm_1(x)
        x = self.attention(x)
        x = residue + x
        
        residue = x
        x = self.norm_2(x)
        x = self.ffn(x)
        x = residue + x
        
        return x 
        
        