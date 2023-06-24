import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class Adapter(nn.Module):

    def __init__(self):
        super(Adapter, self).__init__()
        self.linear1 = nn.Linear(50*256, 256)

        #self.linear2 = nn.Linear(50*256,256)

        self.linear3 = nn.Linear(768,256)
        self.combine = nn.Linear(256*2,256)
        self.sigmod = nn.ReLU()
    
    def forward(self, prompt, x_v):

        
        prompt = self.linear1(torch.reshape(prompt, (prompt.shape[0],-1)))
        #prompt = self.linear2(prompt)
        #prompt = self.linear3(prompt)
        x_v = torch.cat((x_v,prompt),dim=1)

        x_v = self.combine(x_v)
        #x_v = self.sigmod(x_v)

        return x_v