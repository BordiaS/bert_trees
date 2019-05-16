from bertviz import attention, visualization
from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import h5py
import json 
import torch

bert_version = 'bert-large-uncased'
model = BertModel.from_pretrained(bert_version)
tokenizer = BertTokenizer.from_pretrained(bert_version)
output_path='/misc/vlgscratch4/BowmanGroup/datasets/ptb_trees/trees_'+bert_version+'_wsj.hdf5'

def create_dict(file_name, attn_dict, prefix):
    model.eval()
    with open(filename, 'r') as f_wsj:
        data = json.load(f_wsj)

    for key, datum in data.items():
        sent = datum['sentence']
        tokens = tokenizer.tokenize(sent)
        tokens_a_delim = ['[CLS]'] + tokens + ['[SEP]']
        token_ids =  tokenizer.convert_tokens_to_ids(tokens_a_delim)
        tokens_tensor = torch.tensor([token_ids])
        token_type_tensor = torch.LongTensor([[0] * len(tokens_a_delim)])       
        _, _, attn_data_list =  model(tokens_tensor, token_type_ids=token_type_tensor)
        attn_tensor = torch.stack([attn_data['attn_probs'] for attn_data in attn_data_list])
        attention[prefix+key] = attn_tensor.data.numpy()

    return attn_dict

attention={}
filename = '/misc/vlgscratch4/BowmanGroup/datasets/ptb_trees/ptb3-wsj-test.json'    
attention=create_dict(filename,attention,'test_')

#filename = '/misc/vlgscratch4/BowmanGroup/datasets/ptb_trees/ptb3-wsj-train.json'
#attention=create_dict(filename,attention,'train_')

#filename = '/misc/vlgscratch4/BowmanGroup/datasets/ptb_trees/ptb3-wsj-dev.json'
#attention=create_dict(filename,attention,'dev_')



L=len(attention)
print("len attention: ", L)
print("writing weights to the file!!")
with h5py.File(output_path,'w') as f:
    for idx in attention:
        f.create_dataset(idx,data=attention[idx],dtype='float64')
f.close()
print("done") 
