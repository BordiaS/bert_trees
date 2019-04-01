import argparse
import h5py
import json
import numpy as np
import os
import tqdm
import unicodedata

import bert_trees.extract_mst


def export_tree(attn_arr, root_i):
    mst = bert_trees.extract_mst.get_mst_from_attn(attn_arr, root_i)
    out_arc_ls = [[root_i, -1]]
    for arc in mst.values():
        out_arc_ls.append([arc.tail, arc.head])
    return sorted(out_arc_ls, key=lambda _: _[0])


def load_ud_json(path):
    with open(path) as f:
        raw_data = json.loads(f.read())
        data = {}
        for k, v in raw_data.items():
            data[k] = v
            for arc in v["dependencies"]:
                if arc[1] == 0:
                    assert "root" not in v
                    v["root"] = arc[0]
    return data


def load_tokens(path):
    with open(path, "r") as f:
        tokens = json.loads(f.read())
    return tokens


def adjust_tree(tree, gold_map, start_adj=1):
    extended_tree = tree[:]
    node_map = {-1: start_adj - 1}
    new_node = len(tree)
    for i, tup in enumerate(gold_map):
        assert i not in node_map
        node_map[i] = tup[0] + start_adj
        if len(tup) > 1:
            for j in tup[1:]:
                assert new_node not in node_map
                node_map[new_node] = j + start_adj
                extended_tree.append((new_node, i))
                new_node += 1

    new_tree = sorted([
        [node_map[i], node_map[j]]
        for i, j in extended_tree
    ], key=lambda tup_: tup_[0])

    return new_tree, node_map


def strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def get_token_map(raw_gold_tokens, raw_bert_tokens):
    gold_tokens = [strip_accents(s.lower()) for s in raw_gold_tokens]
    bert_tokens = [s.replace("##", "") for s in raw_bert_tokens[1:-1]]

    gold_i = 0
    bert_i = 0
    gold_result = []
    bert_result = []
    while True:
        if gold_i == len(gold_tokens) and bert_i == len(bert_tokens):
            break
        elif gold_tokens[gold_i] == bert_tokens[bert_i]:
            gold_result.append((gold_i,))
            bert_result.append((bert_i,))
            gold_i += 1
            bert_i += 1
        elif len(bert_tokens[bert_i]) != len(gold_tokens[gold_i]):
            sub_bert_i_ls = [bert_i]
            sub_gold_i_ls = [gold_i]
            sub_gold_token = gold_tokens[gold_i]
            sub_bert_token = bert_tokens[bert_i]
            bert_i += 1
            gold_i += 1
            while len(sub_bert_token) != len(sub_gold_token):
                if len(sub_bert_token) < len(sub_gold_token):
                    sub_bert_i_ls.append(bert_i)
                    sub_bert_token += bert_tokens[bert_i]
                    bert_i += 1
                else:
                    sub_gold_i_ls.append(gold_i)
                    sub_gold_token += gold_tokens[gold_i]
                    gold_i += 1
            assert sub_bert_token == sub_gold_token
            bert_result.append(tuple(sub_bert_i_ls))
            gold_result.append(tuple(sub_gold_i_ls))
        else:
            raise Exception
    return gold_result, bert_result


def get_remapped_bert_root(raw_gold_root, gold_map):
    gold_root = raw_gold_root - 1
    for i, gold_tup in enumerate(gold_map):
        if gold_root in gold_tup:
            return i


def compress_bert_attn(attn_arr, bert_map):
    x = attn_arr
    x2 = np.copy(x)
    selector = np.zeros(len(x2)).astype(bool)
    for tup in bert_map:
        selector[tup[0]] = 1
        if len(tup) > 1:
            tup = np.array(tup)
            x2[tup[0], :] = x2[tup, :].sum(0)
            x2[tup[1:], :] = 0
            x2[:, tup[0]] = x2[:, tup].sum(1)
            x2[:, tup[1:]] = 0
    x3 = x2[np.ix_(selector, selector)]
    return x3


def mass_extract(ud_input_path, bert_input_path, tokens_path, output_base_path):
    os.makedirs(output_base_path, exist_ok=True)
    data = load_ud_json(ud_input_path)
    tokens = load_tokens(tokens_path)
    f = h5py.File(bert_input_path, "r")
    print("Loading bert attentions")
    full_arr_dict = {
        sent_id: np.squeeze(np.array(f[sent_id.strip()]), 1)
        for sent_id in data
    }
    num_layers, num_heads, _, _ = full_arr_dict[list(full_arr_dict.keys())[0]].shape

    for layer_i in tqdm.trange(num_layers):
        for head_i in tqdm.trange(num_heads):
            # print(layer_i, head_i)
            output_dict = {}
            for sent_id, datum in tqdm.tqdm(data.items()):
                arr = full_arr_dict[sent_id]
                raw_gold_tokens = datum["tokens"]
                raw_bert_tokens = tokens[sent_id.strip()]
                gold_map, bert_map = get_token_map(
                    raw_gold_tokens,
                    raw_bert_tokens,
                )
                attn_arr = arr[layer_i, head_i, 1:-1, 1:-1]
                compressed_bert_attn = compress_bert_attn(attn_arr, bert_map)
                bert_root = get_remapped_bert_root(datum["root"], gold_map)
                raw_tree = export_tree(compressed_bert_attn, bert_root)
                adjusted_tree, node_map = adjust_tree(raw_tree, gold_map, 1)
                output_dict[sent_id] = {"dependencies": adjusted_tree}
            file_name = f"bert_large__layer{layer_i:02d}__head{head_i:02d}.json"
            with open(os.path.join(output_base_path, file_name), "w") as f:
                f.write(json.dumps(output_dict))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ud_input_path", required=True)
    parser.add_argument("--bert_input_path", required=True)
    parser.add_argument("--tokens_path", required=True)
    parser.add_argument("--output_base_path", required=True)
    args = parser.parse_args()
    mass_extract(
        ud_input_path=args.ud_input_path,
        bert_input_path=args.bert_input_path,
        tokens_path=args.tokens_path,
        output_base_path=args.output_base_path,
    )


if __name__ == "__main__":
    main()
