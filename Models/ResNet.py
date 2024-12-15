import torch
import torch.nn as nn
import Conv as Conv

# ResNet    
# Maybe add More Models here and create this file to be only CNN models


class ResNet(nn.Module):
    def __init__(self,image_size,n_blocks = 4,n_layers = [2,3,3,2],features=[64,128,256,512],n_classes=10):
        super().__init__()
        self.n_blocks = n_blocks # 4
        self.n_layers = n_layers # [2,2,2,2]
        self.n_classes = n_classes # 10
        self.features = features # 
        self.image_size = image_size # 256*256
        self.final_size = image_size // 2**n_blocks 
        assert len(n_blocks) == len(n_layers) and len(n_layers) == len(features) ; "Different dimension, cant build model"
                
        self.input_proj = nn.Conv2d(3,self.features[0],kernel_size=3,stride=0,padding=1) # First input projection
        self.model = nn.ModuleList([self._creatBlock(self.n_layers[i],3,64) for i in range(self.n_blocks)])
        self.output_proj = nn.Linear(self.features[-1]*self.final_size**2,self.n_classes)
        
        
    def _creatBlock(self,n_layers,in_channels,out_channels):
        return nn.Sequential(
            Conv.NResidual(n_layers,in_channels,out_channels),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.input_proj(x)
        for block in self.model:
            x = block(x)
        x = x.view(x.size(0),-1)
        x = self.output_proj(x)
        return x
        
        
        