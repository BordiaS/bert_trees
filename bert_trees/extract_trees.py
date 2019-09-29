import argparse
import h5py
import json
import numpy as np
import os
import tqdm
import unicodedata

import extract_mst

TOKENS_SLICE_DICT = {
    "bert": slice(1, -1),
    "xlnet": slice(0, -2),
    "roberta": slice(1, -2),
}

SPECIAL_MAPS = {
    "roberta": {
        ('âĢ', 'ľ'): ("“",),
        ('âĢ', 'Ŀ'): ("”",),
        # ('âĢ', 'Ļ'): "’",
        ('âĢ', 'Ļ', 's'): ("’s",),
        ("âĢĶ",): ("—",),
        ("isn","âĢ", "Ļ", "t"): ("is", "n’t",),
        ('âĢ', 'Ļ', 've'): ("’ve",),
        ('don', 'âĢ', 'Ļ', 't'): ('do', 'n’t'),
        ('doesn', 'âĢ', 'Ļ', 't'): ('does', 'n’t'),
        ('âĢ', 'Ļ', 'm'): ('’m',),
        ('âĢ', 'Ļ', 're'): ('’re',),
        ('âĢ', 'Ļ', 'd'): ('’d',),
        ('can', 'âĢ', 'Ļ', 't'): ('ca', 'n’t'),
        ('âĢ¦',): ('…',),
        ('Â£',): ('£',),
        ('âĢĵ',): ('–',),
        ('didn', 'âĢ', 'Ļ', 't'): ('did', 'n’t'),
        ('Ben', 'o', 'Ã®', 't'): ('Benoit',),
        ('âĤ¬',): ('€',),
        ('won', 'âĢ', 'Ļ', 't'): ('wo', 'n’t'),
        ('Gonz', 'Ã¡', 'lez'): ('Gonzalez',),
        ('Z', 'Ã¡', 'hor', 'ie'): ('Zahorie',),
        ('Moj', 'm', 'ÃŃ', 'r'): ('Mojmir',),
        ('Me', 'Ã¤', 'n', 'ki', 'eli'): ('Meankieli',),
        ('b', 'j', 'Ã³', 'rr'): ('bjorr',),
        ('Rh', 'Ã´', 'ne',): ('Rhone',),
        ('Y', 'uc', 'at', 'Ã¡n'): ("Yucatan",),
        ('Sh', 'Åį', 'wa',): ("Showa",),
        ('R', 'Ã³', 's'): ('Ros',),
        ('F', 'j', 'Ã¶', 'gur',): ('Fjogur',),
        ('P', 'ÃŃ', 'an', 'Ã³'): ('Piano',),
        ('Pet', 'Ã©n'): ("Peten",),
        ('Hall', 'str', 'Ã¶', 'm'): ("Hallstrom",),
        ('M', 'Ã´', 'mone'): ("Momone",),
        ('Ir', 'Ã¨', 'ne',): ("Irene",),
        ('K', 'Ã¼', 'hn'): ("Kuhn",),
        ('Kr', 'Ã¤', 'ts', 'ch', 'mer'): ('Kratschmer',),
        ('G', 'Ã¼', 'n', 'ter'): ('Gunter',),
        ('D', 'Ã¼', 'nd', 'ar'): ('Dundar',),
        ('O', 'uv', 'ri', 'Ã¨re'): ("Ouvriere",),
        ('Dur', 'Ã¡n',): ("Duran",),
        ('Ã', 'ģ', 'ng', 'el'): ("Angel",),
        ('F', 'Ã¡', 't', 'ima'): ("Fatima",),
        ('B', 'Ã¡', 'Ã±', 'ez'): ("Banez",),
        ('Bar', 'Ã³n'): ("Baron",),
        ('S', 'Ã¡n', 'che', 'z'): ("Sanchez",),
        ('Ãī', 'v', 'ole',): ("Evole",),
        ('W', 'Ã¼r', 'tt', 'ember', 'g'): ("Wurttemberg",),
        ('Ãĸ', 't', 'zi'): ("Otzi",),
        ('S', 'Ã¼', 'db', 'aden'): ("Sudbaden",),
        ('G', 'aud', 'ÃŃ',): ("Gaudi",),
        ('gr', 'Ã¢', 'ce',): ("grace",),
        ('C', 'Ã©s', 'ar'): ("Cesar",),
    },
    "xlnet": {
        ('', '.', '.', '.'): [('…',), ('...',)],
    },
    "bert": {},
}


def match_special(bert_tokens, bert_i, gold_tokens, gold_i, model_name):
    special_map = SPECIAL_MAPS[model_name.split("-")[0]]
    for k, v in special_map.items():
        if tuple(bert_tokens[bert_i: bert_i + len(k)]) == k:
            if isinstance(v, tuple):
                return k, v
            else:
                for v_choice in v:
                    if tuple(gold_tokens[gold_i: gold_i + len(v_choice)]) == v_choice:
                        return k, v_choice
                raise Exception()
    return None


def add_root(datum):
    for arc, _ in datum["dependencies"]:
        if arc[1] == 0:
            datum["root"] = arc[0]
            return
    raise Exception()


def get_tokens_slice(model_name):
    model_type = model_name.split("-")[0]
    return TOKENS_SLICE_DICT[model_type]


def get_relevant_slice(model_name, num_tokens):
    model_type = model_name.split("-")[0]
    tokens_slice = TOKENS_SLICE_DICT[model_type]
    if model_type in ["bert", "roberta"]:
        return slice(tokens_slice.start, num_tokens + tokens_slice.stop)
    elif model_type == "xlnet":
        return slice(-num_tokens + tokens_slice.start,
                     tokens_slice.stop)


def export_tree(attn_arr, root_i):
    mst = extract_mst.get_mst_from_attn(attn_arr, root_i)
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


def get_token_map(raw_gold_tokens, raw_bert_tokens, model_name):
    gold_tokens = [strip_accents(s) for s in raw_gold_tokens]
    if "uncased" in model_name:
        gold_tokens = [s.lower() for s in gold_tokens]
    tokens_slice = get_tokens_slice(model_name)
    if model_name.startswith("bert"):
        bert_tokens = [strip_accents(s.replace("##", "")) for s in raw_bert_tokens[tokens_slice]]
    elif model_name.startswith("xlnet"):
        bert_tokens = [strip_accents(s.replace("▁", "")) for s in raw_bert_tokens[tokens_slice]]
    elif model_name.startswith("roberta"):
        bert_tokens = [s.replace("Ġ", "") for s in raw_bert_tokens[tokens_slice]]
    else:
        raise KeyError()

    gold_i = 0
    bert_i = 0
    gold_result = []
    bert_result = []
    while True:
        if gold_i == len(gold_tokens) and bert_i == len(bert_tokens):
            break
        elif match_special(bert_tokens, bert_i,
                           gold_tokens, gold_i, model_name=model_name):
            roberta_special, gold_special = match_special(
                bert_tokens, bert_i,
                gold_tokens, gold_i, model_name=model_name,
            )

            assert tuple(gold_tokens[gold_i: gold_i + len(gold_special)]) == gold_special

            gold_result.append(tuple([gold_i+j for j in range(len(gold_special))]))
            bert_result.append(tuple([bert_i+j for j in range(len(roberta_special))]))
            gold_i += len(gold_special)
            bert_i += len(roberta_special)
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
            assert sub_bert_token == sub_gold_token, (
                bert_tokens, gold_tokens,
            )
            bert_result.append(tuple(sub_bert_i_ls))
            gold_result.append(tuple(sub_gold_i_ls))
        else:
            print(gold_tokens)
            print(bert_tokens)
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


def mass_extract(ud_input_path, bert_input_path, tokens_path, output_base_path, model_name):
    os.makedirs(output_base_path, exist_ok=True)
    data = load_ud_json(ud_input_path)
    tokens = load_tokens(tokens_path)
    tokens_slice = get_tokens_slice(model_name)
    output_dict_dict = {}

    with h5py.File(bert_input_path, "r") as f_in:
        for sent_id, datum in tqdm.tqdm(data.items()):
            add_root(datum)
            raw_gold_tokens = datum["tokens"]
            raw_bert_tokens = tokens[sent_id.strip()]
            gold_map, bert_map = get_token_map(
                raw_gold_tokens,
                raw_bert_tokens,
                model_name=model_name,
            )

            arr = np.array(f_in[sent_id.strip()])
            num_layers, num_heads, _, _ = arr.shape

            for layer_i in tqdm.trange(num_layers):
                for head_i in tqdm.trange(num_heads):
                    attn_arr = arr[layer_i, head_i, tokens_slice, tokens_slice]
                    compressed_bert_attn = compress_bert_attn(attn_arr, bert_map)
                    bert_root = get_remapped_bert_root(datum["root"], gold_map)
                    raw_tree = export_tree(compressed_bert_attn, bert_root)
                    adjusted_tree, node_map = adjust_tree(raw_tree, gold_map, 1)

                    output_dict_dict_key = layer_i, head_i
                    if output_dict_dict_key not in output_dict_dict:
                        output_dict_dict[output_dict_dict_key] = {}
                    output_dict_dict[output_dict_dict_key][sent_id] = {
                        "dependencies": adjusted_tree,
                    }

    for (layer_i, head_i), output_dict in output_dict_dict.items():
        file_name = f"tree_{model_name}_layer{layer_i:02d}__head{head_i:02d}.json"
        with open(os.path.join(output_base_path, file_name), "w") as f:
            f.write(json.dumps(output_dict))


def get_undirected_attn(attn_matrix):
    zeros_mat = np.zeros_like(attn_matrix)

    for i in range(attn_matrix.shape[0]):
        for j in range(i, attn_matrix.shape[1]):
            zeros_mat[i][j] = attn_matrix[i][j] \
                if (attn_matrix[i][j] > attn_matrix[j][i]) else attn_matrix[j][i]
    return zeros_mat            


def exclude_diagonals(attn_matrix):
    for i in range(attn_matrix.shape[0]):
        attn_matrix[i][i]=-1
    return attn_matrix


def get_max_relations(attn_matrix,no_diagonal=False):
    if no_diagonal:
        attn_matrix=exclude_diagonals(attn_matrix)
    max_relations = attn_matrix.argmax(axis=1)+1
    return list(zip(range(1,len(max_relations)+1),max_relations.tolist())) 


def mass_extract_dependencies(ud_input_path, bert_input_path, tokens_path, output_base_path,
                              model_name, undirected=False):
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
                    model_name=model_name,
                )
                attn_arr = arr[layer_i, head_i, 1:-1, 1:-1]
                compressed_bert_attn = compress_bert_attn(attn_arr, bert_map)
                if undirected:
                    compressed_bert_attn = get_undirected_attn(compressed_bert_attn)
                relations = get_max_relations(compressed_bert_attn)
                output_dict[sent_id] = {"dependencies": relations}
            if undirected:
                file_name = f"undirected_rel_{model_name}_layer{layer_i:02d}__head{head_i:02d}.json"
            else:
                file_name = f"nodiag_rel_{model_name}_layer{layer_i:02d}__head{head_i:02d}.json"
            with open(os.path.join(output_base_path, file_name), "w") as f:
                f.write(json.dumps(output_dict))
            print("done writing to: ", file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ud_input_path", required=True, \
                        default='/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/ud_eng_pud_with_type.json')
    parser.add_argument("--bert_input_path", required=True, \
                       default='/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/bert-large-uncased.hdf5')
    parser.add_argument("--tokens_path", required=True, \
                       default='/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/bert-large-uncased')
    parser.add_argument("--output_base_path", required=True, \
                       default='/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/bert_large_ud')
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--model_name", action="store_true")
    args = parser.parse_args()
    print("model_name: ", args.model_name)
    mass_extract(
        ud_input_path=args.ud_input_path,
        bert_input_path=args.bert_input_path,
        tokens_path=args.tokens_path,
        output_base_path=args.output_base_path,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
