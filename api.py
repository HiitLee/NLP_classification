#-*- coding: utf-8 -*-

import json, re, abusive, socket, context_abusive
from flask import Flask, render_template, request
import flask

#ipaddress = socket.gethostbyname(socket.gethostname())
ipaddress='0.0.0.0'
app = Flask(__name__)

import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

   
embed_lookup = torch.load('./data/embed_lookup.pt')
weights = torch.load('./data/weights.pt')
final_attn = entireContext()
#final_attn.load_state_dict(torch.load("./data/abusive_detection0.pth"))
final_attn.load_state_dict(torch.load('./data/abusive_detection.pt', map_location='cpu'))
final_attn.eval()

#final_attn = torch.load('./data/abusive_detection0.pth',map_location='cpu')
@app.route('/index')
def index():
    result = {}
    return render_template('index.html', result = result)

@app.route('/abusive_test', methods=['POST'])
def abusive_test():
    if request.method == 'POST':
        ttext = request.form
    result = {}
    if ttext['input_text'] == "":
        print("DDDDDDD")
        return render_template('index.html', result = result)

    input_sentence = ttext['input_text']
    input_no_punc = re.sub("[!@#$%^&*().?\"~/<>:;'{}]","",input_sentence)
    result['input'] = input_sentence
    
    abusive_word_list = []
    abusive_word_list += abusive.matching_blacklist(abusive_dict_set, input_sentence)
    abusive_word_list += abusive.matching_blacklist(abusive_dict_set, input_no_punc)
    abusive_word_list = list((set(abusive_word_list)))    
    
    if len(abusive_word_list) == 0:
        result['tag'] = 0
        result['abusive_words'] = 'non_abusive_words'
        input_sentence = clean_text(input_sentence)
        context_result =test_accuracy_full_batch3(input_sentence, 2,final_attn)
        print("############:", context_result[0].item())


        print(context_result[0].item())
        if context_result[0].item() == 1.0 or context_result[0].item() == 2.0:
            result['tag'] = 0
        elif context_result[0].item() ==0.0:
            result['tag'] = 1
    else:
        result['tag'] = 1
        result['abusive_words'] = abusive_word_list

    return render_template('index.html', result = result)

@app.route('/abusive/get_abusiveness', methods=['POST'])
def get_abusiveness():
    try:
        _json = json.loads(request.data)
    except ValueError:
        return redirect(request.url)
    if 'text' not in _json:
        return redirect(request.url)
    result = {}
    
    input_sentence = _json['text'].lower()
    input_no_punc = re.sub("[!@#$%^&*().?\"~/<>:;'{}]","",input_sentence)
    result['input'] = input_sentence
    
    abusive_word_list = []
    abusive_word_list += abusive.matching_blacklist(abusive_dict_set, input_sentence)
    abusive_word_list += abusive.matching_blacklist(abusive_dict_set, input_no_punc)
    
    
    if len(abusive_word_list) == 0:
        abusive_word_list += abusive.edit_distancing(abusive_dict_set, input_sentence)
        abusive_word_list += abusive.n_gram_token(abusive_dict_set, input_sentence)        
        

    abusive_word_list = abusive.remove_whitelist(whitelist_dict_set, abusive_word_list)
    abusive_word_list = list((set(abusive_word_list)))    
        
    if len(abusive_word_list) == 0:
        result['tag'] = 0
    else:
        result['tag'] = 1
        result['abusive_words'] = abusive_word_list
    return json.dumps(result)

if __name__ == '__main__':
   app.run(ipaddress, debug = True, threaded = True)
