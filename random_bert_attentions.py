from bertviz import attention, visualization
from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import h5py
import torch
import random

bert_version = 'bert-base-uncased'
model = BertModel.from_pretrained(bert_version)
tokenizer = BertTokenizer.from_pretrained(bert_version)

for name, param in model.named_parameters():
    if param.requires_grad:
        param.data.uniform_(-0.1, 0.1)

filename = '/scratch/sb6416/Ling3340/extract_tree/UD_English-PUD/en_pud-ud-test.conllu'
with open(filename, 'r') as f:
    data = f.readlines()
i=0
sentences={}

for i in range(1,len(data)):
    line=data[i]
    if line[0]=='#':
        if line[2:9] == "sent_id":
            sentence_id = line[12:]
            
        if line[2:6] == "text":
            sentence =line[9:]
            sentences[sentence_id[:-1]] =sentence[:-1]
        
#print(sentences)
attention={}
for sent in sentences :
    tokens_sent = sentences[sent]
    tokens = tokenizer.tokenize(tokens_sent)
    #print(tokens_sent, tokens)
    tokens_a_delim = ['[CLS]'] + tokens + ['[SEP]']
    token_ids =  tokenizer.convert_tokens_to_ids(tokens_a_delim)
    #print(token_ids, len(token_ids))
    tokens_tensor = torch.tensor([token_ids])
    token_type_tensor = torch.LongTensor([[0] * len(tokens_a_delim)])
    #print(token_type_tensor)
    _, _, attn_data_list =  model(tokens_tensor, token_type_ids=token_type_tensor)
    attn_tensor = torch.stack([attn_data['attn_probs'] for attn_data in attn_data_list])
    attention[sent] = attn_tensor.data.numpy()

L=len(sentences)

sent_id = []
attentions = []
for idx in attention:
    sent_id.append(idx)
    attentions.append(attention[idx])
    

f= h5py.File('attn.h5','w')
dt = h5py.special_dtype(vlen=np.dtype('float64','float64'))
dataset = f.create_dataset('vlen',(L,12,1,12,), dtype=dt)
dataset.value
for i in range(len(attentions)):
    dataset[i]=attentions[i]
dataset.value
f.close()    
