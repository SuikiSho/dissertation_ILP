import numpy as np
from scipy.optimize import linprog

from branch_and_bound import BnBTreeNode, BranchAndBound



def _test1():
    c = np.array([3, 4, 1])
    A_ub = np.array([[-1, -6, -2], [-2, 0, 0]])
    b_ub = np.array([-5, -3])
    A_eq = None
    b_eq = None

    try:
        model = BranchAndBound(c, A_ub, b_ub, A_eq, b_eq)
        sol, val = model.solve()
        print("Optimal solution:", sol)
        print("Optimal value:", val)
    except ValueError as e:
        print("Error:", e)

def _test2():
    c = np.array([1, 1])
    A_ub = np.array([[2, 1]])
    b_ub = np.array([4.5])
    A_eq = None
    b_eq = None

    try:
        model = BranchAndBound(c, A_ub, b_ub, A_eq, b_eq)
        sol, val = model.solve()
        # model.print_tree()
        print("Optimal solution:", sol)
        print("Optimal value:", val)
    except ValueError as e:
        print("Error:", e)

def _test3():
    c = np.array([4, 7, 3])
    A_ub = np.array([[1, 1, 1],
                    [2, 5, 2]])
    b_ub = np.array([5.5, 10.5])
    A_eq = None
    b_eq = None

    try:
        model = BranchAndBound(c, A_ub, b_ub, A_eq, b_eq)
        sol, val = model.solve()
        # model.print_tree()
        print("Optimal solution:", sol)
        print("Optimal value:", val)
    except ValueError as e:
        print("Error:", e)

def _test4():
    c = np.array([1, 1])
    A_ub = np.array([[ 1, 1],
                    [-1, 0]])
    b_ub = np.array([ 1, -2])
    A_eq = None
    b_eq = None

    try:
        model = BranchAndBound(c, A_ub, b_ub, A_eq, b_eq)
        sol, val = model.solve()
        # model.print_tree()
        print("Optimal solution:", sol)
        print("Optimal value:", val)
    except ValueError as e:
        print("Error:", e)

def _test5():
    c = np.array([3, 2])
    A_ub = np.array([[1, 0]])   # x1 <= 4
    b_ub = np.array([4])
    A_eq = np.array([[1, 1]])   # x1 + x2 = 5
    b_eq = np.array([5])

    try:
        model = BranchAndBound(c, A_ub, b_ub, A_eq, b_eq)
        sol, val = model.solve()
        # model.print_tree()
        print("Optimal solution:", sol)
        print("Optimal value:", val)
    except ValueError as e:
        print("Error:", e)

def _test6():
    c = np.array([1, 1])
    A_ub = None
    b_ub = None
    A_eq = np.array([[1, 1]])   # x1 + x2 = 3
    b_eq = np.array([3])

    try:
        model = BranchAndBound(c, A_ub, b_ub, A_eq, b_eq)
        sol, val = model.solve()
        # model.print_tree()
        print("Optimal solution:", sol)
        print("Optimal value:", val)
    except ValueError as e:
        print("Error:", e)

def _test7():
    c = np.array([2, 1])
    A_ub = None
    b_ub = None
    A_eq = np.array([[1, 1]])   # x1 + x2 = 1.5
    b_eq = np.array([1.5])

    try:
        model = BranchAndBound(c, A_ub, b_ub, A_eq, b_eq)
        sol, val = model.solve()
        # model.print_tree()
        print("Optimal solution:", sol)
        print("Optimal value:", val)
    except ValueError as e:
        print("Error:", e)

def _test8():
    c = np.array([6, 10, 12])
    A_ub = np.array([
        [1, 2, 3],   # 重量<=5
        [1, 0, 0],   # x1 <= 1
        [0, 1, 0],   # x2 <= 1
        [0, 0, 1],   # x3 <= 1
    ])
    b_ub = np.array([5.5, 1.5, 1.5, 1.5])
    A_eq = None
    b_eq = None

    try:
        model = BranchAndBound(c, A_ub, b_ub, A_eq, b_eq)
        sol, val = model.solve()
        model.print_tree()
        print("Optimal solution:", sol)
        print("Optimal value:", val)
    except ValueError as e:
        print("Error:", e)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    _test1()
    _test2()
    _test3()
    _test4()
    _test5()
    _test6()
    _test7()
    _test8()