import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(q,k,v,scale_factor):
    att = q@k.t
    att = att/scale_factor
    att = torch.softmax(att,dim=1)
    att = att@v
    return att