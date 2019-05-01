from bertviz import attention, visualization
from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import h5py
import torch
import random

bert_version = 'bert-large-uncased'
model = BertModel.from_pretrained(bert_version)
tokenizer = BertTokenizer.from_pretrained(bert_version)

sent='I eat pizza with mushroom.'

tokens = tokenizer.tokenize(sent)
tokens_a_delim = ['[CLS]'] + tokens + ['[SEP]']
token_ids =  tokenizer.convert_tokens_to_ids(tokens_a_delim)
tokens_tensor = torch.tensor([token_ids])
token_type_tensor = torch.LongTensor([[0] * len(tokens_a_delim)])

_, _, attn_data_list =  model(tokens_tensor, token_type_ids=token_type_tensor)
attentions=[attn['attn_probs'].squeeze().detach().numpy() for attn in attn_data_list]
