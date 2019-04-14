import json

def load_gt_trees(gt_file):
    pass

def save_gt_trees(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    i=0
    sentences={}

    arcs = []
    tokens = []
    sent = {}
    for i in range(len(data)):
        line=data[i]

        if not line.strip():
            continue


        if line[0]=='#':
            if line[2:9] == "sent_id":

                if sent:
                    sent['dependencies'] = arcs
                    sent['tokens'] = tokens
<<<<<<< HEAD
                    sent['dep_type'] = dep_type
=======
>>>>>>> a4df8c416d6466e8b9912bea8d1f6a2663ae666e
                    sentences[sentence_id] = sent
                    
                    sent = {}
                    arcs = []
                    tokens = []
                sentence_id = line[12:]

            if line[2:6] == "text":
                sentence =line[9:]
                sent['sentence'] = sentence
            #sentences[sentence_id[:-1]] =sentence[:-1]
        else:
            columns = line.rstrip("\r\n").split()
            try:
<<<<<<< HEAD
                dep_type = columns[7].strip()
                arc = ((int(columns[0]), int(columns[6])), dep_type)
                print(dep_type)
=======
                arc = (int(columns[0]), int(columns[6]))
>>>>>>> a4df8c416d6466e8b9912bea8d1f6a2663ae666e
            except:
                print("line: ", line)
                continue
                #arc = (int(float(columns[0])), int(float(columns[6])))
            arcs.append(arc)
            tokens.append(columns[1].strip())



<<<<<<< HEAD
    with open('/scratch/sb6416/Ling3340/extract_tree/UD_English-PUD/ud_eng_pud_with_type.json', 'w') as f:
=======
    with open('/scratch/sb6416/Ling3340/extract_tree/UD_English-PUD/ud_eng_pud.json', 'w') as f:
>>>>>>> a4df8c416d6466e8b9912bea8d1f6a2663ae666e
        json.dump(sentences,f)


if __name__ == "__main__":
    save_gt_trees('/scratch/sb6416/Ling3340/extract_tree/UD_English-PUD/en_pud-ud-test.conllu')
    print("done")
