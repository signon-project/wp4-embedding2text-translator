#!/bin/python
# using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_cors import CORS
import json

import os
import sys
import torch
import numpy as np
from tokeniser import Tokeniser
from slt_model import SLTmodel


# creating the flask app
app = Flask(__name__)
CORS(app)

#global args, model, accelerator = None, None, None
lang_mapping = {
    'spa': 'es_XX',
    'eng': 'en_XX',
    'gle': 'ga_GA',
    'nld': 'nl_XX',
    'dut': 'nl_XX',
    'vgt': 'vgt', # Flemish SL
    'ssp': 'ssp', # Spanish SL
    'bfi': 'bfi', # British SL
    'dse': 'dse' # British SL
}
        
@app.route('/', methods=['POST'])
def process():
    global model
    #if request.method == 'POST':
    json_data = request.get_json(force = True)

    ##### TODO: embedding-ak json-etik jaso
    embeddings = json_data['SourceLanguageProcessing']['SLR']['embedding']
        
    src_lang = json_data['App']['sourceLanguage']
    trg_lang = json_data['App']['translationLanguage']
  
    input = torch.asarray(embeddings).unsqueeze(0)
    input = torch.cat((input, torch.zeros((1, input.size(1), 1024-128))), dim=2)
    input_mask = ~(input != torch.zeros(1024))[..., 0]

    res = model.forward(x=input, src_mask=input_mask, tgt_lang=lang_mapping[trg_lang.lower()])#, decoder_input_ids=decoder_input_ids)
    translation = model.tokeniser.decode(res, skip_special_tokens=True)

    print(translation)
    return(jsonify({'translationText': translation}))

def init():
    global model
    pretrained_embeddings_vocab = 'pretrained_embeddings_vrt_news_vocabulary.txt'
    tokeniser = Tokeniser(pretrained_embeddings_vocab)
    weights = torch.load('./model/best.ckpt', map_location='cpu')['model_state']
        
    model = SLTmodel(tokeniser)
    model.load_state_dict(weights)
    
# driver function
if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=5003)
