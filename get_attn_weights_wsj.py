from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
import h5py
import json 
import torch
import tqdm


def create_dict(model, tokenizer, file_name, prefix):
    model.eval()
    with open(file_name, 'r') as f_wsj:
        data = json.load(f_wsj)

    attn = {}
    for key, datum in data.items():
        sent = datum['sentence']
        tokens = tokenizer.tokenize(sent)
        tokens_a_delim = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = tokenizer.convert_tokens_to_ids(tokens_a_delim)
        tokens_tensor = torch.tensor([token_ids])
        token_type_tensor = torch.LongTensor([[0] * len(tokens_a_delim)])       
        _, _, attn_data_list = model(tokens_tensor, token_type_ids=token_type_tensor)
        attn_tensor = torch.stack([attn_data['attn_probs'] for attn_data in attn_data_list])
        attn[prefix+key] = attn_tensor.data.numpy()

    return attn


def save_attn_weights(model, bert_version, output_path, prefix):
    tokenizer = BertTokenizer.from_pretrained(bert_version)

    filename = '/scratch/sb6416/Ling3340/extract_tree/UD_English-PUD/en_pud-ud-test.conllu'
    with open(filename, 'r') as f_wsj:
        data = json.load(f_wsj)

    attention = {}
    model.eval()
    with torch.no_grad():
        for key, datum in tqdm.tqdm(data.items(), total=len(data)):
            sent = datum['sentence']
            tokens = tokenizer.tokenize(sent)
            tokens_a_delim = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = tokenizer.convert_tokens_to_ids(tokens_a_delim)
            tokens_tensor = torch.tensor([token_ids])
            token_type_tensor = torch.LongTensor([[0] * len(tokens_a_delim)])
            _, _, attn_data_list = model(tokens_tensor, token_type_ids=token_type_tensor)
            attn_tensor = torch.stack([attn_data['attn_probs'] for attn_data in attn_data_list])
            attention[prefix+key] = attn_tensor.data.numpy()

    print("writing weights to the file!!")

    with h5py.File(output_path, 'w') as f:
        for idx in attention:
            f.create_dataset(idx, data=attention[idx], dtype='float64')
    f.close()
    print("done")


#filename = '/misc/vlgscratch4/BowmanGroup/datasets/ptb_trees/ptb3-wsj-train.json'
#attention=create_dict(filename,attention,'train_')

#filename = '/misc/vlgscratch4/BowmanGroup/datasets/ptb_trees/ptb3-wsj-dev.json'
#attention=create_dict(filename,attention,'dev_')


def main():
    bert_version = 'bert-large-uncased'
    model = BertModel.from_pretrained(bert_version)
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    output_path = '/misc/vlgscratch4/BowmanGroup/datasets/ptb_trees/trees_' + bert_version + '_wsj.hdf5'

    filename = '/misc/vlgscratch4/BowmanGroup/datasets/ptb_trees/ptb3-wsj-test.json'
    attn = create_dict(model, tokenizer, filename, 'test_')

    length = len(attn)
    print("len attention: ", length)
    print("writing weights to the file!!")
    with h5py.File(output_path, 'w') as f:
        for idx in attn:
            f.create_dataset(idx, data=attn[idx], dtype='float64')
    f.close()
    print("done")


if __name__ == "__main__":
    main()
