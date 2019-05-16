import json
import sys
from collections import Counter

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
    correct_type = Counter()
    total_type = Counter()
    assert len(ref_trees)==len(sys_trees), "ref and sys should have same number of sentences"

    

    for i in ref_trees:
        ref_tree = set(tuple(x[0]) for x in ref_trees[i]['dependencies'])
        
        if mode=='rb':
            sys_tree = get_rb_indices(ref_trees[i]['tokens'])
        elif mode=='lb':
            sys_tree = get_lb_indices(ref_trees[i]['tokens'])
        elif mode == 'one_plus':
            sys_tree = None
        else:
            sys_tree = set(tuple(x) for x in sys_trees[i]['dependencies'])
 
        #overlap = sys_tree.intersection(ref_tree)

        for ref_c in ref_trees[i]['dependencies']:
            if mode == 'one_plus':
                if ref_c[0][0] == (ref_c[0][1]) - 1: # or \
                    #ref_c[0][0] == (ref_c[0][1]) - 1:
                    correct_type[ref_c[1]] += 1
            elif mode == 'two_plus':
                if ref_c[0][0] == (ref_c[0][1]) - 2: #or \
                    #ref_c[0][0] == (ref_c[0][1]) - 2:
                    correct_type[ref_c[1]] += 1
            elif mode == 'three_plus':
                if ref_c[0][0] == (ref_c[0][1]) - 3: #or \
                    #ref_c[0][0] == (ref_c[0][1]) - 3:
                    correct_type[ref_c[1]] += 1
            elif mode == 'four_plus':
                if ref_c[0][0] == (ref_c[0][1]) - 4: # or \
                    #ref_c[0][0] == (ref_c[0][1]) - 4:
                    correct_type[ref_c[1]] += 1
            elif mode == 'five_plus':
                if ref_c[0][0] == (ref_c[0][1]) - 5: # or \
                    #ref_c[0][0] == (ref_c[0][1]) - 5:
                    correct_type[ref_c[1]] += 1 
            elif mode == 'undirected':
                if tuple(ref_c[0]) in sys_tree or \
			tuple([ref_c[0][1], ref_c[0][0]]) in sys_tree:
                    correct_type[ref_c[1]] += 1
            else:
                if tuple(ref_c[0]) in sys_tree:
                    correct_type[ref_c[1]] += 1
            total_type[ref_c[1]] += 1

        
        #prec = float(len(overlap)) / (len(sys_tree) + 1e-8)
        #reca = float(len(overlap)) / (len(ref_tree) + 1e-8)
        """
        if len(sys_tree) == 0:
            reca = 1.
            if len(ref_tree) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)   

        prec_list.append(prec)
        reca_list.append(reca)
        f1_list.append(f1) 
        """
    header_list = "acl,acl:relcl,advcl,advmod,amod,appos,aux,aux:pass,case,cc,ccomp,compound,conj,cop,csubj,csubj:pass,dep,det,det:predet,expl,fixed,flat,iobj,mark,nmod,nmod:npmod,nmod:poss,nmod:tmod,nsubj,nsubj:pass,nummod,obj,obl,obl:tmod,parataxis,punct,root,vocative,xcomp".split(',')
    value_list = []
    
    for key in header_list:
        if key in total_type:
            value_list.append(str(correct_type[key]*100.0/total_type[key]))
        else:
            value_list.append("0")
    header = ','.join(header_list)
    values = ','.join(value_list)
    if not f1_list:
        return 0, header, values
    return mean(f1_list), header, values


def corpus_stats(sys_trees, ref_tree, mode="undirected"):
    total_diagonal=0
    total_prev=0
    total_next=0
    for i in sys_trees:
        sys_tree = sys_trees[i]['dependencies']
        ref_tree = ref_trees[i]
        sen_len=len(ref_tree['tokens'])
        diagonals=0.
        for i in range(1, sen_len+1):
            if [i,i] in sys_tree:
                diagonals+=1

        diagonals/=(sen_len*1.0)

        prev_word_list =0
        next_word_list =0
        
        for i in range(2, sen_len+1):
            if [i-1,i] in sys_tree:
                prev_word_list += 1
        
        for i in range(1, sen_len):
            if [i, i+1] in sys_tree:
                next_word_list+=1

        if sen_len > 1:
            prev_word_list = prev_word_list*1.0/(sen_len-1)
            next_word_list=next_word_list*1.0/(sen_len-1)
           
        total_diagonal+=diagonals
        total_prev+=prev_word_list
        total_next+=next_word_list
    total_diagonal=total_diagonal*100/len(sys_trees)
    total_prev=total_prev*100/len(sys_trees)
    total_next=total_next*100/len(sys_trees)

    return total_diagonal, total_prev, total_next


if __name__ == "__main__":
    base_dir=sys.argv[1]
    ref_tree_path = '/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/ud_eng_pud_with_type.json'

    sys_tree_path = base_dir+'/'+ sys.argv[2] #bert_large__layer00__head00.json' 

    if len(sys.argv) > 3:
        mode=sys.argv[3].strip()
    else:
        mode='undirected'

    with open(ref_tree_path, 'r') as f_ref:
        ref_trees = json.load(f_ref)

    with open(sys_tree_path, 'r') as f_sys:
        sys_trees = json.load(f_sys)

    #total_diagonal, total_prev, total_next = corpus_stats(sys_trees, ref_trees)
    total_diagonal, total_prev, total_next=0,0,0
    mean_f1, header, values = eval(ref_trees, sys_trees,mode)
    print(sys.argv[2]+  ',' + str(mean_f1) + ','+str(total_diagonal)+','+str(total_prev)+','+str(total_next) + "," + values)
    #print(sys.argv[2]+','+str(total_diagonal)+','+str(total_prev)+','+str(total_next))
    '''
    print("one")
    mean_f1, header, values = eval(ref_trees, sys_trees, 'undirected')
    print(sys.argv[2]+  ',' + str(mean_f1) + ','+str(total_diagonal)+','+str(total_prev)+','+str(total_next) + "," + values)
    print("\ntwo")
    mean_f1, header, values = eval(ref_trees, sys_trees, 'none')
    print(sys.argv[2]+  ',' + str(mean_f1) + ','+str(total_diagonal)+','+str(total_prev)+','+str(total_next) + "," + values)
    print('\nthree')
    mean_f1, header, values = eval(ref_trees, sys_trees, 'three_plus')
    print(sys.argv[2]+  ',' + str(mean_f1) + ','+str(total_diagonal)+','+str(total_prev)+','+str(total_next) + "," + values)
    print('\nfour')
    mean_f1, header, values = eval(ref_trees, sys_trees, 'four_plus')
    print(sys.argv[2]+  ',' + str(mean_f1) + ','+str(total_diagonal)+','+str(total_prev)+','+str(total_next) + "," + values)
    print('\nfive')
    mean_f1, header, values = eval(ref_trees, sys_trees, 'five_plus')
    print(sys.argv[2]+  ',' + str(mean_f1) + ','+str(total_diagonal)+','+str(total_prev)+','+str(total_next) + "," + values)
    '''
