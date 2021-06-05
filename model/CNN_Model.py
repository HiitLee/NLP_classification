'''
 @Author : juhyounglee
 @Datetime : 2021/05/17
 @File : CNN_Model.py
 @Last Modify Time : 2020/08/01
 @Contact : dlwngud3028@naver.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


class CNN(nn.Module):
    
    def __init__(self, args, weight=None):
        super(CNN,self).__init__()
        self.args = args
        
        Ci = 1
        if(weight==None):
            self.embed = nn.Embedding(args.vocab, args.CNN_embed)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()

        Kernel=[]
        for K in args.CNN_kernels:
            Kernel.append( K + 1 if K % 2 == 0 else K)


        self.convs_1d = nn.ModuleList([nn.Conv2d(1, args.CNN_filter, (k, args.CNN_embed), padding=(k//2,0)) for k in Kernel])
        
        self.dropout = nn.Dropout(args.CNN_dropout)
        self.fc1 = nn.Linear(args.CNN_filter*len(args.CNN_kernels), args.num_class)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, text,text1):
        embeds = self.embed(text)   # (N,W,D)
        embeds = embeds.unsqueeze(1)  # (N,Ci,W,D)
        
        cnn_x = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        cnn_x = torch.cat(cnn_x, 1)
        
        x = self.dropout(cnn_x)     # (N,len(Ks)*Co)
        logit = self.fc1(x)     # (N,C)
        return logit

