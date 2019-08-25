from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
import h5py
import tqdm
import torch


def save_attn_weights(model, bert_version, output_path):
    tokenizer = BertTokenizer.from_pretrained(bert_version)

    filename = '/scratch/sb6416/Ling3340/extract_tree/UD_English-PUD/en_pud-ud-test.conllu'
    with open(filename, 'r') as f:
        data = f.readlines()
    sentences = {}

    for i in range(1, len(data)):
        line = data[i]
        if line[0] == '#':
            if line[2:9] == "sent_id":
                sentence_id = line[12:]

            if line[2:6] == "text":
                sentence = line[9:]
                sentences[sentence_id[:-1]] = sentence[:-1]

    # print(sentences)
    attention = {}
    model.eval()
    with torch.no_grad():
        for sent in tqdm.tqdm(sentences):
            tokens_sent = sentences[sent]
            tokens = tokenizer.tokenize(tokens_sent)
            # print(tokens_sent, tokens)
            tokens_a_delim = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = tokenizer.convert_tokens_to_ids(tokens_a_delim)
            # print(token_ids, len(token_ids))
            tokens_tensor = torch.tensor([token_ids])
            token_type_tensor = torch.LongTensor([[0] * len(tokens_a_delim)])
            # print(token_type_tensor)
            _, _, attn_data_list = model(tokens_tensor, token_type_ids=token_type_tensor)
            attn_tensor = torch.stack([attn_data['attn_probs'] for attn_data in attn_data_list])
            attention[sent] = attn_tensor.data.numpy()

    print("writing weights to the file!!")

    with h5py.File(output_path, 'w') as f:
        for idx in attention:
            f.create_dataset(idx, data=attention[idx], dtype='float64')
    f.close()
    print("done")


def main():
    bert_version = 'bert-large-uncased'
    model = BertModel.from_pretrained(bert_version)
    save_attn_weights(
        model=model,
        bert_version=bert_version,
        output_path='attn.h5',
    )
