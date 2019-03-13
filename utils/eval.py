def get_rb_indices(sent):
   ''' 
       sent : list of tokens in the sentence

       returns:
	  rb_tree : list of right branching arc indices
   ''' 
   max_index = len(sent) - 1 
   
   rb_tree = [(i, max_index) for i in range(max_index)]
   return rb_tree


def get_lb_indices(sent):
    ''' 
       sent : list of tokens in the sentence

       returns:
           lb_tree: list of left branching arc indices
   '''
   length = len(sent)
   lb_tree = [(0, i) for i in range(1, length)]
   return lb_tree


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


def eval(ref_tree, sys_tree):
    '''
        return:
            mean F1 score
    '''
    prec_list = []
    reca_list = []
    f1_list = []

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

