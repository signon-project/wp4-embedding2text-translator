import sentencepiece as spm
#from transformers import MBartTokenizerFast
from transformers import AutoTokenizer

class Tokeniser:
    def __init__(self, pretrained_embeddings_vocab):
        super(Tokeniser).__init__()        
        
        self.token_to_id = {}
        self.id_to_token = {}
        
        """self.use_pretrained_embeddings = cfg["use_pretrained_embeddings"]
        if self.use_pretrained_embeddings: # not tested
            pretrained_embeddings_vocab = cfg['pretrained_embeddings_vocab']
            self.mapping = {}
            with open(pretrained_embeddings_vocab, 'r') as f:
                for count, line in enumerate(f):
                    # old id, token
                    id, token = line.strip().split('\t')
                    self.mapping[int(id)] = count
            self.rev_mapping = {v:k for k,v in self.mapping.items()}
             
            self.tokeniser = MBartTokenizerFast.from_pretrained("facebook/mbart-large-cc25")
            self.tokeniser.src_lang = 'de_DE'
            self.tokeniser.tgt_lang = 'de_DE'
            self.UNK_TOKEN = 3
            self.BOS_TOKEN = 0
            self.EOS_TOKEN = 2
            self.PAD_TOKEN = 1
        else:"""

        # Fitxategi batean kargatutako hiztegia irakurri
        """vocab_path = cfg['txt_vocab']
        with open(vocab_path, 'r') as f:
            for line in f:
                id, token = line.strip().split(' ') # zenbaki token parea lerro bakoitzeko
                new_idx = len(self.id_to_token)
                self.id_to_token[new_idx] = token
                self.token_to_id[token] = new_idx
        tokeniser_file = cfg['tokeniser_path']
        self.tokeniser = spm.SentencePieceProcessor()
        self.tokeniser = spm.SentencePieceProcessor()
        self.tokeniser.Load(tokeniser_file)
        self.UNK_TOKEN = self.tokeniser.unk_id()
        self.BOS_TOKEN = self.tokeniser.bos_id()
        self.EOS_TOKEN = self.tokeniser.eos_id()
        self.PAD_TOKEN = self.tokeniser.pad_id()"""
            
        self.tokeniser = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokeniser.src_lang = 'nl_XX'
        self.tokeniser.tgt_lang = 'nl_XX'
        self.UNK_TOKEN = 3
        self.BOS_TOKEN = 0
        self.EOS_TOKEN = 2
        self.PAD_TOKEN = 1

        pretrained_embeddings_vocab = pretrained_embeddings_vocab
        self.mapping = {}
        with open(pretrained_embeddings_vocab, 'r') as f:
            for count, line in enumerate(f):
                # old id, token
                id, token = line.strip().split('\t')
                self.mapping[int(id)] = count
        self.rev_mapping = {v:k for k,v in self.mapping.items()}

        """ self.rev_mapping = self.tokeniser.vocab
        self.mapping = {v:k for k, v in self.rev_mapping.items()} """

    def encode(self, x, add_special_tokens=True):
        """ if self.use_pretrained_embeddings:
            with self.tokeniser.as_target_tokenizer():
                y = self.tokeniser(x, add_special_tokens=False).input_ids
            y = [self.mapping.get(elem, self.mapping[3]) for elem in y]
        else:
            y = self.tokeniser.encode_as_ids(x)
        return y """
        with self.tokeniser.as_target_tokenizer():
            item = self.tokeniser(x, add_special_tokens=add_special_tokens)
        y = item['input_ids']
        mask = item['attention_mask']
        y = [self.mapping.get(elem, self.mapping[3]) for elem in y]
        return y, mask

    def decode(self, x, skip_special_tokens=True):
        """ if self.use_pretrained_embeddings:
            if isinstance(x[0], int):
                x = [self.rev_mapping[z] for z in x]
                decoded_txt = self.tokeniser.decode(x, skip_special_tokens=True)
            else:
                x = [[self.rev_mapping[elem] for elem in z] for z in x]
                decoded_txt = self.tokeniser.batch_decode(x, skip_special_tokens=True)
        else:
            if isinstance(x[0], int):
                decoded_txt = []
                for id in x:
                    token = self.tokeniser.id_to_piece(int(id))
                    if token == '</s>': break
                    decoded_txt.append(token)
            else:
                decoded_txt = []
                for i in range(len(x)):
                    sent = []
                    for id in x[i]:
                        token = self.tokeniser.id_to_piece(int(id))
                        if token == '</s>': break
                        if token != '':
                            sent.append(token)
                    decoded_txt.append(sent)
                
        return decoded_txt """
        if isinstance(x[0], int):
            x = [self.rev_mapping[z] for z in x]
            decoded_txt = self.tokeniser.decode(x, skip_special_tokens=skip_special_tokens)
        else:
            #print(self.rev_mapping)
            #print(x)
            x = [[self.rev_mapping[elem] for elem in z] for z in x]
            decoded_txt = self.tokeniser.batch_decode(x, skip_special_tokens=skip_special_tokens)
     
        return decoded_txt

    def __len__(self):
        return len(self.mapping)

    def bos_id(self):
        return self.mapping[self.tokeniser.vocab[self.tokeniser.tgt_lang]]#self.BOS_TOKEN

    def eos_id(self):
        return self.EOS_TOKEN

    def pad_id(self):
        return self.PAD_TOKEN

    def unk_id(self):
        return self.UNK_TOKEN
