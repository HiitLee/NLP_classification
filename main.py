from gensim import models
import csv
import fire
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.CNN_Model import CNN
from model.HCL_2CLSTM_Model import HCL_CLSTM_CLSTM
from model.CLSTM_Model import CLSTM
from model.LSTM_Model import LSTM
from model.LSTM_Attn_Model import LSTM_Attn
from trainer import Trainer

from sklearn.model_selection import StratifiedKFold
import argparse
import os
import sys
import datetime
import torch.nn.functional as F
import torch.nn as nn
from config import config
args = config.parser.parse_args()


def embed_index(word2index):
    for i in range(0, len(embed_lookup.wv.index2word)):
        try:
            word2index[embed_lookup.wv.index2word[i]] = i
        except KeyError:
            word2index[embed_lookup.wv.index2word[i]] = i
    return word2index

word2index={}
embedding = "random"
if(embedding == "fastText"):
    embed_lookup = models.fasttext.load_facebook_model("cc.ko.300.bin")
    word2index = embed_index(word2index)
elif(embedding == "glove"):
    embed_lookup = models.fasttext.load_facebook_model("cc.ko.300.bin")
    word2index = embed_index(word2index)


tempidx={}


def tokenizer_HCL(inputText):
    if(args.tokenizer == 'char'):
        reviews_words = list(inputText)
    elif(args.tokenizer == 'word'):
         reviews_words = inputText.split(' ')
    tokenized_reviews = []

    if(embedding == "fastText" or embedding == "glove"):
        sentences = []
        for word in reviews_words:
            if('.' in word  or '?' in word  or '!' in word ):
                word = word.replace('.', '').replace('!','').replace('?','')
                try:
                    sentences.append(word2index[word])
                except:
                    sentences.append(0)
                if(len(sentences) >= 1):
                    tokenized_reviews.append(sentences)
                sentences = []
                continue
            try:
                idx = word2index[word]
            except: 
                idx = 0
            sentences.append(idx)
        if(len(sentences)>=1):
            tokenized_reviews.append(sentences)
    else:
        sentences = []
        for word in reviews_words:
            if('.' in word  or '?' in word  or '!' in word ):
                word = word.replace('.', '').replace('!','').replace('?','')
                if(word not in tempidx):
                    tempidx[word] = len(tempidx)+1
                    sentences.append(tempidx[word])
                else:
                    sentences.append(tempidx[word])
                if(len(sentences) >=1):
                    tokenized_reviews.append(sentences)
                sentences = []
                continue
            if(word not in tempidx):
                tempidx[word] = len(tempidx)+1
                idx = tempidx[word]
            else: 
                idx = tempidx[word]
            sentences.append(idx)
        if(len(sentences) >=1):
            tokenized_reviews.append(sentences)
    return tokenized_reviews


def tokenizer(inputText):
    if(args.tokenizer == 'char'):
        reviews_words = list(inputText)
    elif(args.tokenizer == 'word'):
        reviews_words = inputText.split(' ')

    tokenized_reviews = []
    if(embedding == "fastText" or embedding == "glove"):
        for word in reviews_words:
            if('"' in word or "'" in word or '?' in word or '!' in word or '.' in word or ',' in word):
                word = word.replace('.','').replace("'",'').replace('"','').replace('?','').replace('!','').replace(',','')
                try:
                    idx = word2index[word]
                except:
                    idx = 0
                tokenized_reviews.append(idx)
    else:
        reviews_words = list(inputText)
        for word in reviews_words:
            if(word not in tempidx):
                tempidx[word] = len(tempidx)+1
                idx = tempidx[word]
            else:
                idx = tempidx[word]
            tokenized_reviews.append(idx)
    return tokenized_reviews





def main():

    print("Reading Train set.....")
    fr = open(args.data_train_file, 'r', encoding='utf-8')
    lines = csv.reader(fr,  delimiter='\t')
    X_train=[]
    y_train=[]
    X_train1=[]
    meanSen = []
    meanLine = []
    meanLine2 = []
    meanLineTrain=[]
    meanLineTest=[]
    print('args_model:', args.model)
    print('args_token#;', args.tokenizer)
    print('lr #:', args.lr)
    for line in lines:
        if('label' in line[0]):
            continue
        if('HCL' in args.model):
            w2i = list(tokenizer_HCL(line[1]))
            for subsentence in w2i:
                meanSen.append(len(subsentence))
           # if('LSTM' in args.model):
            w2i2 = list(tokenizer(line[1]))
            meanLine2.append(len(w2i2))
            X_train1.append(w2i2)
        else:
            w2i = list(tokenizer(line[1]))

        meanLine.append(len(w2i))
        if(len(w2i)<=0):
            meanLineTrain.append([2])
        else:
            meanLineTrain.append([len(w2i)])
        X_train.append(w2i)
        y_train.append([int(line[0])])
    fr.close()

    print("Reading Test set......")

    fr = open(args.data_test_file, 'r', encoding='utf-8')
    lines = csv.reader(fr,  delimiter='\t')
    X_test=[]
    X_test1=[]
    y_test=[]

    for line in lines:
        if('label' in line[0]):
            continue
        if('HCL' in args.model):
            w2i = list(tokenizer_HCL(line[1]))
            for subsentence in w2i:
                meanSen.append(len(subsentence))
            #if('LSTM' in args.model):
            w2i2 = list(tokenizer(line[1]))
            meanLine2.append(len(w2i2))
            X_test1.append(w2i2)
        else:
            w2i = list(tokenizer(line[1]))
        meanLine.append(len(w2i))
        if(len(w2i)<=0):
            meanLineTest.append([2])
        else:
            meanLineTest.append([len(w2i)])
        X_test.append(w2i)
        y_test.append([int(line[0])])
    fr.close()

    if('HCL' in args.model):
        maxLen = int(np.max(meanSen))
        maxSen1 = int(np.max(meanLine2))
        print('maxLen:', maxLen)
    maxSen = int(np.max(meanLine))
#    if(args.maxSen != 0):
#        maxSen = int(args.maxSen)
#        for i in range(0, len(X_train)):
#            #print(X_train[i])
#            #print(X_train1[i])
#            if(maxSen <= len(X_train[i])):
#                X_train[i] = X_train[i][:maxSen]
#                if('LSTM' in args.model):
#                    X_train1[i] = X_train1[i][:maxSen]
#        for i in range(0, len(X_test)):
#            if(maxSen <= len(X_test[i])):
#                X_test[i] = X_test[i][:maxSen]
#                if('LSTM' in args.model):
#                    X_test1[i] = X_test1[i][:maxSen]
        
    print('maxSen:', maxSen)
    if('HCL' in args.model):
        for i  in range(0,len(X_train)):
            for k in range(0, len(X_train[i])):
                n_pad = maxLen - len(X_train[i][k])
                X_train[i][k].extend([0]*n_pad)
            n_pad = maxSen - len(X_train[i])
            for l in range(0, n_pad):
                temp=[]
                temp.extend([0]*maxLen)
                X_train[i].append(temp)
        for i  in range(0,len(X_test)):
            for k in range(0, len(X_test[i])):
                n_pad = maxLen - len(X_test[i][k])
                X_test[i][k].extend([0]*n_pad)
            n_pad = maxSen - len(X_test[i])
            for l in range(0, n_pad):
                temp=[]
                temp.extend([0]*maxLen)
                X_test[i].append(temp)
        for i  in range(0, len(X_train1)):
            n_pad = maxSen1 - len(X_train1[i])
            X_train1[i].extend([0]*n_pad)
        for i in range(0, len(X_test1)):
            n_pad = maxSen1 - len(X_test1[i])
            X_test1[i].extend([0]*n_pad)
    else:
        for i  in range(0, len(X_train)):
            n_pad = maxSen - len(X_train[i])
            X_train[i].extend([0]*n_pad)
        for i in range(0, len(X_test)):
            n_pad = maxSen - len(X_test[i])
            X_test[i].extend([0]*n_pad)



    print("pre-trained embedding loading........")

    k=0
#    weights = list()
#    for i in range(0, len(embed_lookup.wv.vocab)):
#        cc = embed_lookup.wv.index2word[i]
#        try:
#            weights.append(np.ndarray.tolist(embed_lookup[cc]))
#        except KeyError:
#            weights.append(np.ndarray.tolist(np.random.rand(300,)))
#        k+=1                                                                   
#    weights = np.array(weights, dtype=np.float32)
#    weights = torch.from_numpy(weights)
#    weights = torch.FloatTensor(weights)

    weights = None
    print("pre-trained embedding loading success....")

    print("length of trainset: ", len(X_train))
    print("length of testset: ", len(X_test))


    print("args.sampling##:", args.sampling)
    if(args.sampling == True):
        print("data sampling ............")
        skf = StratifiedKFold(n_splits=10)
        X_sample=[]
        X_sample1=[]
        y_sample=[]
        for train_idx, test_idx in skf.split(X_train, y_train):
            gg={}
            for k in test_idx:
                X_sample.append(X_train[k])
                if('HCL' in args.model):
                    X_sample1.append(X_train1[k])

            for k in test_idx:
                y_sample.append(y_train[k])
                if(y_train[k][0] not in gg):
                    gg[y_train[k][0]] =1
                else:
                    gg[y_train[k][0]]+=1

            for key, value in gg.items():
                print(key, value)
            break
        X_train = X_sample
        X_train1 = X_train1
        y_train = y_train
        print("data sampling complete..........")


    class TDataset(Dataset):

        def __init__(self, data,data1, labels,seq ):
            super().__init__()
            self.data = data
            self.data1 = data1
            self.labels = labels
            self.seq = seq
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x = torch.LongTensor(self.data[idx])
            y = torch.LongTensor(self.data1[idx])
            c = torch.LongTensor(self.labels[idx])
            z = self.seq[idx]
            return x,y,c,z



    if('HCL' in args.model):
        model = HCL_CLSTM_CLSTM(args,weights,args.model)
    elif(args.model == 'CLSTM'):
        model = CLSTM(args,weights)
    elif(args.model =='CNN'):
        model = CNN(args,weights)
    elif(args.model == 'LSTM'):
        model = LSTM(args,weights)
    elif(args.model == 'GRU'):
        model = GRU(args,weights)
    elif(args.model == 'LSTM_Attn'):
        model = LSTM_Attn(args,weights)
    elif(args.model == 'GRU_Attn'): 
        model = GRU_Attn(args,weights)
    else:
        model = CNN(args, weights) 
    batch = args.batch

    if(args.optim == 'Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif(args.optim == 'SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if('HCL' in args.model):
        data_iter = DataLoader(dataset=TDataset(X_train, X_train1, y_train,meanLineTrain), batch_size=args.batch, shuffle=True)
        data_test_iter = DataLoader(dataset=TDataset(X_test,X_test1, y_test, meanLineTest), batch_size=args.batch, shuffle=False)
    else:
        data_iter = DataLoader(dataset=TDataset(X_train, X_train, y_train,meanLineTrain), batch_size=args.batch, shuffle=True)
        data_test_iter = DataLoader(dataset=TDataset(X_test,X_test, y_test, meanLineTest), batch_size=args.batch, shuffle=False)

    loss_sum =0.


    model.cuda()
    model.train()
    steps=0
    criterion = nn.CrossEntropyLoss()  
    print("training..............")
    print('args.model: ', args.model)
    print("#############:", args.mode)
    trainer = Trainer(args, data_iter, data_test_iter, model, optimizer, criterion)
    if(args.mode == 'train'):
        trainer.train_epoch()
    elif(args.mode == 'eval'):
        trainer.eval()

if __name__ == '__main__':
    fire.Fire(main())
