'''
 @Author : juhyounglee
 @Datetime : 2021/05/17
 @File : LSTM_Model.py
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
class LSTM(nn.Module):
    
    def __init__(self, args, weight=None):
        super(LSTM, self).__init__()
        self.args = args
        
        if(weight==None):
            self.embed = nn.Embedding(args.vocab,args.LSTM_embed)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()


        self.bilstm = nn.LSTM(args.LSTM_embed, args.LSTM_hidden, num_layers=args.LSTM_layer, bidirectional=True)
        self.hidden2label1 = nn.Linear(args.LSTM_hidden*2, args.num_class)
        self.m = nn.LogSoftmax(dim=1)
        
    def forward(self, input,seq_lengths):
        input = self.embed(input)
        packed_input = pack_padded_sequence(input, seq_lengths.cpu().numpy(), batch_first=True)
        r_output, (final_hidden_state,final_cell_state) = self.bilstm(packed_input)
        r_output, input_sizes = pad_packed_sequence(r_output, batch_first=True)
        output = torch.cat([final_hidden_state[0], final_hidden_state[1]],1)
        out1 = self.hidden2label1(output)
        return self.m(out1)

