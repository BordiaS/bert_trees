import json

wsj_path='/scratch/sb6416/Ling3340/WSJ/ptb.jsonl'

with open(wsj_path) as fwsj:
    for line in fwsj:
        line_dict = eval(line)
        print(line_dict['sentence1_binary_parse'])
        sys.exit()
