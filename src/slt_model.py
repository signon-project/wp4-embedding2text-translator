import os
import torch
import json
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from tokeniser import Tokeniser
from beam_search import beam_search
from utils import Embeddings, subsequent_mask, MaskedNorm, get_activation

class SgnEmbed(torch.nn.Module):
    def __init__(self):
        super(SgnEmbed, self).__init__()
        input_size = 1024
        embedding_dim = 1024
        self.ln = torch.nn.Linear(input_size, embedding_dim)

        self.norm = MaskedNorm(
            norm_type='batch', num_groups=16, num_features=embedding_dim
        )

        self.activation = get_activation('softsign')

    def forward(self, x):
        return self.ln(x)

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.output_size = None
        

    def forward(self, encoder_input, src_mask, trg_embed, trg_mask=None):
        #print(trg_mask.size(), 'trg_mask', subsequent_mask(trg_embed.size(1)).type_as(trg_mask).size(),'submask')
        """ trg_mask = trg_mask & subsequent_mask(trg_embed.size(1)).type_as(trg_mask)
        trg_mask = trg_mask.squeeze(0) """
        """ print(trg_mask.size(), 'trg_mask')
        
        print(trg_mask) """
        """ print(encoder_input.size(), trg_embed.size(), trg_mask.size())
        print('---------------------') """
        #print(trg_mask.size())
    
        output = self.model(
            inputs_embeds=encoder_input,
            attention_mask=torch.ones((encoder_input.size(0), encoder_input.size(1))),#src_mask,#.squeeze(dim=1),
            decoder_inputs_embeds=trg_embed,
            decoder_attention_mask=trg_mask
            #decoder_attention_mask=trg_mask.squeeze(dim=1) if len(trg_mask.size()) == 3 else trg_mask,
        )
        return output.logits

class SLTmodel(torch.nn.Module):
    def __init__(self, tokeniser):
        super(SLTmodel, self).__init__()
        self.translation_beam_size = 6
        self.translation_max_output_length = 100
        self.translation_beam_alpha = 1
        pretrained_embeddings_path = 'embedding_table_vrt_news.npy'
        pretrained_embeddings_bias_path = 'embedding_table_bias_vrt_news.npy'
        
        self.tokeniser = tokeniser

        self.txt_eos_index = self.tokeniser.eos_id()
        self.txt_pad_index = self.tokeniser.pad_id()
        self.txt_bos_index = self.tokeniser.bos_id()
        
        self.txt_embed = Embeddings(
            embedding_dim=1024,
            num_heads=16,
            activation_type='softsign',
            norm_type='batch',
            vocab_size=len(self.tokeniser),
            padding_idx=self.txt_pad_index,
        )
        self.sgn_embed = SgnEmbed()
        self.decoder = BaseModel()

        embeddings = torch.as_tensor(np.load(pretrained_embeddings_path))
       
        new_embeddings = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        new_embeddings.weight.data = embeddings
        new_embeddings.padding_idx = tokeniser.pad_id()
        self.txt_embed.lut = new_embeddings
        with torch.no_grad():
            self.decoder.model.lm_head = torch.nn.Linear(embeddings.shape[1], embeddings.shape[0], bias=False)
            self.decoder.model.lm_head.weight.copy_(embeddings.clone())
            self.decoder.model.final_logits_bias = torch.as_tensor(np.load(pretrained_embeddings_bias_path))
        self.decoder.model.config.vocab_size = embeddings.shape[0]
        self.decoder.output_size = embeddings.shape[0]

    def forward(self, x, src_mask, trg_mask=None, tgt_lang='nl_XX'):#, decoder_input_ids):#trg_mask):
        bos_index=self.tokeniser.encode(tgt_lang, add_special_tokens=False)[0][0]

        stacked_txt_output, _ = beam_search(
            size=self.translation_beam_size,
            encoder_input=x,
            src_mask=src_mask,
            embed=self.txt_embed,
            max_output_length=self.translation_max_output_length,
            alpha=self.translation_beam_alpha,
            eos_index=self.tokeniser.eos_id(),
            pad_index=self.tokeniser.pad_id(),
            #bos_index=self.tokeniser.bos_id(),
            bos_index=bos_index,
            decoder=self.decoder,
        )

        return stacked_txt_output
