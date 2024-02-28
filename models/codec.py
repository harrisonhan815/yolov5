import torch
import torch.nn as nn



class codec(nn.Module):
    def __init__(self):

        super().__init__()

        self.codec = nn.Conv2d(1,1,1,1)
        nn.init.constant_(self.codec.weight, 1)
        nn.init.constant_(self.codec.bias, 0)

    def forward(self, x):
        out = self.codec(x)
        #out = x
        return out
    

class hyper_model(nn.Module):
    def __init__(self, codec, model):

        super().__init__()

        self.codec = codec
        self.model = model

    def forward(self, x):
        x = self.codec(x)
        out = self.model(x)


        return out
