import pandas as pd
from ext.min_spanning_arborescence import min_spanning_arborescence, Arc


def get_mst_from_attn(attn, root):
    """
    Extract an MST from an attention matrix and a root note (0-indexed)

    :param attn: L_in * L_out matrix
    :param root: root node (int)
    :return: List of Arcs (edges)
    """
    nodes = range(len(attn))
    graph = [
        Arc(i, -attn[i][j], j)
        for i in nodes
        for j in nodes
        if j != i
    ]
    return min_spanning_arborescence(graph, root)


def create_tree_df(mst, tokens, conll_offset_1=True):
    """Create Tree DataFrame in CoNLL_U format

    :param mst: MST (list of Arcs) from get_mst_from_attn
    :param tokens: List of tokens
    :param conll_offset_1: Whether to modify indices to 1-indexed
    :return: DataFrame
    """
    assert len(tokens) == len(mst) + 1
    tail_to_head = {
        arc.tail: arc.head
        for arc in mst.values()
    }
    tree_output = [
        (i, token, tail_to_head.get(i, -1))
        for i, token in enumerate(tokens)
    ]
    tree_df = pd.DataFrame(tree_output, columns=["id", "token", "head"])
    if conll_offset_1:
        tree_df["id"] += 1
        tree_df["head"] += 1
    return tree_df
