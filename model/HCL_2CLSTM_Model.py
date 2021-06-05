
'''
 @Author : juhyounglee
 @Datetime : 2021/05/17
 @File : HCL_2CLSTM_Model.py
 @Last Modify Time : 2020/08/01
 @Contact : dlwngud3028@naver.com
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


"""
Neural Networks model : ensemble with Hierarchical C-SLTM, C-LSTM and C-LSTM 
"""
class CLSTM1(nn.Module):
    
    def __init__(self, args, weight=None):
        super(CLSTM1, self).__init__()
        self.args = args
        
        if(weight==None):
            self.embed = nn.Embedding(args.vocab, args.CLSTM1_embed)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()

        
        KK=[]
        for K in args.CLSTM1_kernel:
            KK.append( K + 1 if K % 2 == 0 else K)

        self.convs_1d = nn.ModuleList([nn.Conv2d(1, args.CLSTM1_filter, (k, args.CLSTM1_embed), padding=(k//2,0)) for k in KK])
        
            
        self.bilstm = nn.LSTM(args.CLSTM1_filter*len(args.CLSTM1_kernel),args.CLSTM1_filter*len(args.CLSTM1_kernel) ,  num_layers=args.CLSTM1_lstm_layer,dropout = args.dropout, bidirectional=True)
        # gru
        # linear
        self.hidden2label = nn.Linear(args.CLSTM1_filter*len(args.CLSTM1_kernel) * 2, args.num_class)
        #  dropout
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
    
class CLSTM2(nn.Module):
    
    def __init__(self, args, weight=None):
        super(CLSTM2, self).__init__()
        self.args = args
        if(weight==None):
            self.embed = nn.Embedding(args.vocab, args.CLSTM2_embed)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()

        KK=[]
        for K in args.CLSTM2_kernel:
            KK.append( K + 1 if K % 2 == 0 else K)


        self.convs_1d = nn.ModuleList([nn.Conv2d(1,args.CLSTM2_filter, (k, args.CLSTM2_embed), padding = (k//2,0) ) for k in KK])  
        self.bilstm = nn.LSTM(args.CLSTM2_dim*len(args.CLSTM2_kernel), args.CLSTM2_dim*len(args.CLSTM2_kernel),  num_layers=args.CLSTM2_lstm_layer,dropout = args.dropout, bidirectional=True)
        # gru
        # linear
        self.hidden2label = nn.Linear(args.CLSTM2_dim*len(args.CLSTM2_kernel)*2, args.num_class)
        #  dropout
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
    
    
class wordCLSTM(nn.Module):
    def __init__(self, args, weight=None):

        super(wordCLSTM, self).__init__()
        
        self.args = args
        if(weight==None):
            self.embed = nn.Embedding(args.vocab, args.wordCLSTM_embed)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()

        KK=[]
        for K in args.wordCLSTM_kernel:
            KK.append( K + 1 if K % 2 == 0 else K)


        self.convs_1d = nn.ModuleList([nn.Conv2d(1,args.wordCLSTM_filter, (k, args.wordCLSTM_embed), padding = (k//2,0) ) for k in KK])
        self.bilstm = nn.LSTM(args.wordCLSTM_filter*len(args.wordCLSTM_kernel), args.wordCLSTM_filter, num_layers=args.wordCLSTM_lstm_layer, dropout = args.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(args.wordCLSTM_filter*len(args.wordCLSTM_kernel)*2, args.num_class)
        self.dropout = nn.Dropout(args.dropout)
      
    def conv_and_pool(self, input, conv):
        cnn_x = conv(input)
        cnn_x = F.relu(cnn_x)
        cnn_x = cnn_x.squeeze(3)

        return cnn_x

    def forward(self, input):
        input = self.embed(input)
        embeds = input.unsqueeze(1)
        embeds = self.dropout(embeds)
#        cnn_2x=[]
#        for conv in self.convs_1d:
#            cnn_2x.append(self.conv_and_pool(embeds, conv))

        cnn_x = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = torch.transpose(cnn_x, 1,2)
        bilstm_out,(final_hidden_state, final_cell_state) =self.bilstm(cnn_x)
        bilstm_out = torch.transpose(bilstm_out, 2,1)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        return bilstm_out.unsqueeze(0)

    
    
    
class sentCLSTM(nn.Module):

    def __init__(self, args):
        super(sentCLSTM, self).__init__()
        self.args = args
        KK=[]
        for K in args.sentCLSTM_kernel:
            KK.append( K + 1 if K % 2 == 0 else K)


        self.convs_1d = nn.ModuleList([nn.Conv2d(1,args.sentCLSTM_filter, (k, args.sentCLSTM_embed), padding = (k//2,0) ) for k in KK])
        self.bilstm = nn.LSTM(args.sentCLSTM_dim*len(args.sentCLSTM_kernel), args.sentCLSTM_dim*len(args.sentCLSTM_kernel), num_layers=args.sentCLSTM_lstm_layer,dropout = args.dropout, bidirectional=True)
        # gru
        # linear
        self.hidden2label = nn.Linear(args.sentCLSTM_dim*len(args.sentCLSTM_kernel)*2, args.num_class)
        #  dropout
        self.dropout = nn.Dropout(0.1)

    def conv_and_pool(self, input, conv):
        cnn_x = conv(input)
        cnn_x = F.relu(cnn_x)
        cnn_x = cnn_x.squeeze(3)
        return cnn_x
    
    def forward(self, input):
        input = torch.transpose(input, 1,0)
        embeds = input.unsqueeze(1)
        embeds = self.dropout(embeds)
        cnn_x = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = torch.transpose(cnn_x, 1,2)
       # print("##:", cnn_x.shape)
        bilstm_out,(final_hidden_state, final_cell_state) =self.bilstm(cnn_x)
        bilstm_out = torch.transpose(bilstm_out, 2,1)
        
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = self.hidden2label(bilstm_out)

        return bilstm_out
    
    
class HCL_CLSTM_CLSTM(nn.Module):
    
    def __init__(self, args,weight=None, name='HCL'):
        super(HCL_CLSTM_CLSTM, self).__init__()
        self.wordCLSTM = wordCLSTM(args,weight)
        self.senCLSTM = sentCLSTM(args)
        self.CLSTM1= CLSTM1(args,weight)
        self.CLSTM2 = CLSTM2(args,weight)
        
        self.mode = name

        self.relu = nn.ReLU()

    def forward(self, text, text1):
      
        s = None
        for i in range(0, int(text.size(1))):
            _s = self.wordCLSTM(text[:,i,:])
            if(s is None):
                s = _s
            else:
                s = torch.cat((s,_s),0)    
        logits = self.senCLSTM(s)

        if(self.mode == 'HCL_CLSTM'):
            logits_CLSTM1 = self.CLSTM1(text1)
            logits = logits+logits_CLSTM1

        if(self.mode == 'HCL_2CLSTM'):
            print("222222222")
            logits_CLSTM1 = self.CLSTM1(text1)
            logits_CLSTM2 = self.CLSTM2(text1)
            logits = logits+logits_CLSTM1+logits_CLSTM2
 
        return F.log_softmax(logits, dim=1)
