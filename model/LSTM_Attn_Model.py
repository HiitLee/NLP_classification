'''
 @Author : juhyounglee
 @Datetime : 2021/05/17
 @File : LSTM_Attn_Model.py
 @Last Modify Time : 2020/08/01
 @Contact : dlwngud3028@naver.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
Neural Networks model : ensemble with Hierarchical C-SLTM, C-LSTM and C-LSTM 
"""
class LSTM_Attn(nn.Module):
    
    def __init__(self, args, weight):
        super(LSTM_Attn, self).__init__()
        self.args = args
        
        if(weight==None):
            self.embed = nn.Embedding(args.vocab, args.LSTM_Attn_embed)
        else:
            self.embed = nn.Embedding.from_pretrained(weight,freeze=False).cuda()

        self.w = nn.Parameter(torch.zeros(args.LSTM_Attn_embed*2))
        self.bilstm = nn.LSTM(args.LSTM_Attn_embed, args.LSTM_Attn_hidden, num_layers=args.LSTM_Attn_layer, bidirectional=True)
        self.hidden2label = nn.Linear(args.LSTM_Attn_hidden*2, args.num_class)
        self.m = nn.LogSoftmax(dim=1)
        
    def forward(self, input,seq_lengths):
        input = self.embed(input)
        packed_input = pack_padded_sequence(input, seq_lengths.cpu().numpy(), batch_first=True)
        r1_output, (final_hidden_state1,final_cell_state1) = self.bilstm(input)
        r_output, (final_hidden_state,final_cell_state) = self.bilstm(packed_input)
        r_output, input_sizes = pad_packed_sequence(r_output, batch_first=True)

        alpha = F.softmax(torch.matmul(r_output, self.w)).unsqueeze(-1)
        out = r_output * alpha 

        out = torch.sum(out , 1)
        out = F.relu(out)
        out = self.hidden2label(out)
        return self.m(out)
