import json
def trees_in_json(filename,json_file):
    with open(filename, 'r') as f:
        data = f.readlines()
    i=0
    sentences={}
    arcs = []
    tokens = []
    sent = {}
    sentence_id =0
    for i in range(len(data)):
        line=data[i]
        if not line.strip():
            sentence  = " ".join(tokens)

            sent["sentence"] =sentence
            sent['dependencies'] = arcs
            sent['tokens'] = tokens
            sentences[sentence_id] = sent
            sentence_id+=1
            sent = {}
            arcs=[]
            arcs_with_tokens=[]
            tokens=[]
            continue

        columns = line.rstrip("\r\n").split()
        dep_type = columns[7]
        arc_with_tokens =((columns[0], columns[6]), dep_type)
        arc = ((int(columns[0]), int(columns[6])), dep_type)
        arcs.append(arc)
        tokens.append(columns[1].strip())
    
    with open(json_file, 'w') as f:
        json.dump(sentences,f)
   
