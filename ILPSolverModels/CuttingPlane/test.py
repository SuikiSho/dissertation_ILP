import numpy as np
# ILPSolverModels/CuttingPlane/test.py
from .gomory import GomoryCuttingPlane
from .simplex import Simplex, TwoPhaseSimplex, DualSimplex


def _test1():
    """
    Maximize: x1 + x2
    Subject to: x1 + x2 <= 4
    x1, x2 are integers
    """
    c = np.array([1, 1])
    A_ub = np.array([[1, 1]])
    b_ub = np.array([4])

    gomory = GomoryCuttingPlane(c=c, A_ub=A_ub, b_ub=b_ub)
    solution, value = gomory.solve()

    print("Optimal solution:", solution)
    print("Optimal value:", value)
    print("Final tableau:\n", gomory.tableau)

def _test2():
    """
    Maximize: x1 + x2
    Subject to: 2x1 + x2 <= 4.5
    x1, x2 are integers
    """ 
    c = np.array([1, 1])
    A_ub = np.array([[2, 1]])
    b_ub = np.array([4.5])

    gomory = GomoryCuttingPlane(c, A_ub, b_ub)
    solution, value = gomory.solve()

    print("Optimal solution:", solution)
    print("Optimal value:", value)
    print("Final tableau:\n", gomory.tableau)

def _test3():
    """
    Maximize: 3x1 + x2
    Subject to: 5x1 + 2x2 <= 10.5
    x1, x2 are integers
    """
    c = np.array([3, 1])
    A_ub = np.array([[5, 2]])
    b_ub = np.array([10.5])

    gomory = GomoryCuttingPlane(c, A_ub, b_ub)
    solution, value = gomory.solve()

    print("Optimal solution:", solution)
    print("Optimal value:", value)
    print("Final tableau:\n", gomory.tableau)
    
def _test4():
    """
    Minimize: x1 + x2
    Subject to: x1 + x2 = 0.5
    x1, x2 are integers
    """
    c = np.array([-1, -1])
    A_ub = None
    b_ub = None

    A_eq = np.array([[1, 1]])
    b_eq = np.array([0.5])

    try:
        gomory = GomoryCuttingPlane(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        solution, value = gomory.solve()

        print("Optimal solution:", solution)
        print("Optimal value:", value)
        print("Final tableau:\n", gomory.tableau)
    except Exception as e:
        print("Error:", e)

def _test5():
    """
    Minimize: x1 + 2x2
    Subject to: x1 + x2 = 3
                x1 <= 2.5
    x1, x2 are integers
    """
    c = np.array([-1, -2])
    A_ub = np.array([[1, 0]])
    b_ub = np.array([2.5])

    A_eq = np.array([[1, 1]])
    b_eq = np.array([3])

    gomory = GomoryCuttingPlane(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    solution, value = gomory.solve()

    print("Optimal solution:", solution)
    print("Optimal value:", value)
    print("Final tableau:\n", gomory.tableau)
    
def _test6():
    c = np.array([1, 1, 1, 1, 1], dtype=float)
    A_ub = np.array([
        [0, 0, 0, 1, 1],   # x4 + x5 ≤ 1.5
        [0, 0, 1, 1, 0],   # x3 + x4 ≤ 1.5
        [0, 1, 1, 0, 0],   # x2 + x3 ≤ 1.5
        [1, 1, 0, 0, 0],   # x1 + x2 ≤ 1.5
    ], dtype=float)
    b_ub = np.array([1.5, 1.5, 1.5, 1.5], dtype=float)
    A_eq = None
    b_eq = None

    gomory = GomoryCuttingPlane(c, A_ub, b_ub, A_eq, b_eq)
    opt_sol, opt_val = gomory.solve()
    print("Optimal Solution:", opt_sol)
    print("Optimal Value:", opt_val)
    print("Finial tableau:", gomory.tableau)

def _test7():
    c = np.array([3., 3., 5., 9., 3.])
    A_ub = np.array([
        [6., 6., 4., 1., 2.],
        [4., 1., 9., 2., 9.],
        [8., 4., 4., 9., 7.]
    ])
    b_ub = np.array([9.5, 12.5, 16.])
    A_eq = None
    b_eq = None

    gomory = GomoryCuttingPlane(c, A_ub, b_ub, A_eq, b_eq)
    opt_sol, opt_val = gomory.solve()
    print("Optimal Solution:", opt_sol)
    print("Optimal Value:", opt_val)
    print("Finial tableau:", gomory.tableau)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)  # Set print options for better readability

    _test1()
    _test2()
    _test3()
    _test4()
    _test5()
    _test6()
    # _test7()