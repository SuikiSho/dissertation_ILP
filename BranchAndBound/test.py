import math
import numpy as np
from scipy import optimize

from bnb_Tree import BnBTree, BnBTreeNode
import branch_and_bound



def _test1():
    c = [3, 4, 1]
    A_ub = [[-1, -6, -2], [-2, 0, 0]]
    b_ub = [-5, -3]
    A_eq = None
    b_eq = None
    bounds = [(0, None), (0, None), (0, None)]
    tree = BnBTree()
    r = branch_and_bound.branch_and_bound(c, A_ub, b_ub, A_eq, b_eq, bounds, tree.root)
    print(r)
    print(tree)