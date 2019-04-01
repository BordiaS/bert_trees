import json
import sys

def get_rb_indices(sent):
   ''' 
       sent : list of tokens in the sentence

       returns:
	  rb_tree : list of right branching arc indices
   ''' 
   max_index = len(sent)  
   
   rb_tree = [(i, max_index) for i in range(1,max_index)]
   return set(rb_tree)


def get_lb_indices(sent):
    ''' 
       sent : list of tokens in the sentence

       returns:
           lb_tree: list of left branching arc indices
   '''
    length = len(sent) + 1
    lb_tree = [(1, i) for i in range(2, length)]
    return set(lb_tree)


def get_undirected_arcs(tree):
    '''
        To make F1 calculation easier, 
        for all undirected trees, we make the arc go from left to right
    '''
    tmp_tree = []
    for arc in tree:
        if arc[0] > arc [1]:
            tmp_tree.append((arc[1], arc[0]))
        else:
            tmp_tree.append(arc)

    return tmp_tree


def mean(x):
    return sum(x) / len(x)


def eval(ref_trees, sys_trees, mode='sys'):
    '''
        return:
            mean F1 score
    '''
    prec_list = []
    reca_list = []
    f1_list = []

    assert len(ref_trees)==len(sys_trees), "ref and sys should have same number of sentences"

    for i in ref_trees:
        
        ref_tree = set(tuple(x) for x in ref_trees[i]['dependencies'])

        if mode!='sys':
            ref_tree = get_undirected_arcs(ref_tree)
        if mode=='rb':
            sys_tree = get_rb_indices(ref_trees[i]['tokens'])
        elif mode=='lb':
            sys_tree = get_lb_indices(ref_trees[i]['tokens'])
        else:
            sys_tree = set(tuple(x) for x in sys_trees[i]['dependencies'])
        overlap = sys_tree.intersection(ref_tree)
        prec = float(len(overlap)) / (len(sys_tree) + 1e-8)
        reca = float(len(overlap)) / (len(ref_tree) + 1e-8)

        if len(sys_tree) == 0:
            reca = 1.
            if len(ref_tree) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)   

        prec_list.append(prec)
        reca_list.append(reca)
        f1_list.append(f1) 

    return mean(f1_list)

if __name__ == "__main__":
    ref_tree_path = '/scratch/sb6416/Ling3340/extract_tree/UD_English-PUD/ud_eng_pud.json'
    sys_tree_path = '/scratch/sb6416/Ling3340/extract_tree/extracted_trees/v1/'+ sys.argv[1] #bert_large__layer00__head00.json' 

    with open(ref_tree_path, 'r') as f_ref:
        ref_trees = json.load(f_ref)

    with open(sys_tree_path, 'r') as f_sys:
        sys_trees = json.load(f_sys)

    print(sys.argv[1] + '\t' + str(eval(ref_trees, sys_trees, 'lb')))
