'''
 @Author : juhyounglee
 @Datetime : 2021/05/17 
 @File : CLSTM_Model.py
 @Last Modify Time : 2020/08/01
 @Contact : dlwngud3028@naver.com
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

class CLSTM(nn.Module):
    
    def __init__(self, args, weight=None):
        super(CLSTM, self).__init__()
        self.args = args
        
        if(weight==None):
            self.embed = nn.Embedding(args.CLSTM1_dim, args.CLSTM1_embed)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()
        
        Kernel=[]
        for K in args.CLSTM1_kernel:
            Kernel.append( K + 1 if K % 2 == 0 else K)

        self.convs_1d = nn.ModuleList([nn.Conv2d(1, args.CLSTM1_filter, (k, args.CLSTM1_embed), padding=(k//2,0)) for k in Kernel])
        self.bilstm = nn.LSTM(args.CLSTM1_filter*len(args.CLSTM1_kernel),args.CLSTM1_filter*len(args.CLSTM1_kernel) , dropout=0.1, num_layers=args.CLSTM1_lstm_layer, bidirectional=True)
        self.hidden2label = nn.Linear(args.CLSTM1_filter*len(args.CLSTM1_kernel) * 2, args.num_class)
        self.dropout = nn.Dropout(0.1)
        
    def conv_and_pool(self, input, conv):
        cnn_x = conv(input)
        cnn_x = F.relu(cnn_x)
        cnn_x = cnn_x.squeeze(3)
        return cnn_x

    def forward(self, input):
        input = self.embed(input)
        embeds = input.unsqueeze(1)
        embeds = self.dropout(embeds)
        cnn_x = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = torch.transpose(cnn_x, 1,2)

        bilstm_out,(final_hidden_state, final_cell_state) =self.bilstm(cnn_x)
        bilstm_out = torch.transpose(bilstm_out, 2,1)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)

        logit = self.hidden2label(bilstm_out)
        
        return logit
    
