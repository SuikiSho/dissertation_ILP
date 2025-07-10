import numpy as np
from simplex import Simplex
from simplex import TwoPhaseSimplex
from simplex import DualSimplex
from scipy.optimize import linprog

# Test for the Simplex method
def _test1():
    # Bounded solution
    c = np.array([3, 2])
    A_ub = np.array([[1, 1], [2, 1], [1, 0]])
    b_ub = np.array([4, 6, 2])

    simplex = Simplex(c, A_ub, b_ub)
    solution, value = simplex.solve()
    print("Optimal solution:", solution)
    print("Optimal value:", value)
    print("Final tableau:\n", simplex.tableau)

def _test2():
    # Unbounded solution
    c = np.array([1, 1])
    A_ub = np.array([[-1, 1]])
    b_ub = np.array([1])

    simplex = Simplex(c, A_ub, b_ub)
    try:
        solution, value = simplex.solve()
        print("Optimal solution:", solution)
        print("Optimal value:", value)
        print("Final tableau:\n", simplex.tableau)
    except Exception as e:
        print("Error:", e)

def _test3():
    # Infeasible solution
    c = np.array([1, 1])
    A_ub = np.array([[1, 1]])
    b_ub = np.array([-1]) 

    simplex = Simplex(c, A_ub, b_ub)
    try:
        solution, value = simplex.solve()
        print("Optimal solution:", solution)
        print("Optimal value:", value)
        print("Final tableau:\n", simplex.tableau)
    except Exception as e:
        print("Error:", e)

# Test for the Two-Phase Simplex method
def _test4():
    c = np.array([3, 2])
    A_ub = np.array([[1, 1], [2, 1], [1, 0]])
    b_ub = np.array([4, 6, 2])
    A_eq = np.array([[1, 2]])
    b_eq = np.array([5])

    two_phase_simplex = TwoPhaseSimplex(c, A_ub, b_ub, A_eq, b_eq)
    solution, value = two_phase_simplex.solve()
    print("Two-phase optimal solution:", solution)
    print("Two-phase optimal value:", value)
    print("Final tableau:\n", two_phase_simplex.tableau)

def _test5():
    c = np.array([2, 3])
    A_ub = np.array([[1, 1], [-1, 0]])
    b_ub = np.array([5, -1])
    A_eq = np.array([[2, 1]])
    b_eq = np.array([6])

    two_phase_simplex = TwoPhaseSimplex(c, A_ub, b_ub, A_eq, b_eq)
    solution, value = two_phase_simplex.solve()
    print("Two-phase optimal solution:", solution)
    print("Two-phase optimal value:", value)
    print("Final tableau:\n", two_phase_simplex.tableau)

def _test6():
    c = np.array([2, 3])
    A_ub = np.array([[1, 1]])   # x + y <= 2
    b_ub = np.array([2])
    A_eq = np.array([[2, 1]])   # 2x + y = 6
    b_eq = np.array([6])

    two_phase_simplex = TwoPhaseSimplex(c, A_ub, b_ub, A_eq, b_eq)
    try:
        solution, value = two_phase_simplex.solve()
        print("Two-phase optimal solution:", solution)
        print("Two-phase optimal value:", value)
        print("Final tableau:\n", two_phase_simplex.tableau)
    except Exception as e:
        print("Error:", e)

# Test for the Dual Simplex method
def _test7():
    tableau = np.array([
        [3, 2, 0, 0, 0, 0],       # c_j - z_j
        [2, 1, 1, 0, 0, 4],       # 2x + y + s1 = 4
        [1, 0, 0, 1, 0, 2],       # x + s2 = 2
        [-2, -1, 0, 0, 1, -1]     # -2x - y + a1 = -1 (不满足 RHS)
    ], dtype=float)
    basic_index = np.array([2, 3, 4])

    dual_simplex = DualSimplex(None, None, None, tableau=tableau, basic_index=basic_index)
    solution, value = dual_simplex.solve()
    print("Dual simplex optimal solution:", solution)
    print("Dual simplex optimal value:", value) 
    print("Final tableau:\n", dual_simplex.tableau)

def _test8():
    """
    Maximiz: c = x2
    Subject to: x1 >= 0
                -2x1 + x2 <= 2
                x1 + 2x2 <= -1
                x1, x2 >= 0
    This is an example of a dual simplex problem with a negative RHS and a ratio of 0.
    """
    tableau = np.array([
        [0, 1, 0, 0, 0, 0],
        [-1, 0, 1, 0, 0, 0],      # s1 = 0
        [-2, 1, 0, 1, 0, 2],
        [1, 2, 0, 0, 1, -1]       
    ], dtype=float)
    basic_index = np.array([2, 3, 4])

    dual_simplex = DualSimplex(None, None, None, tableau=tableau, basic_index=basic_index)
    try:
        solution, value = dual_simplex.solve()
        print("Dual simplex optimal solution:", solution)
        print("Dual simplex optimal value:", value)
        print("Final tableau:\n", dual_simplex.tableau)
    except Exception as e:
        print("Error:", e)

def _test9():
    c = np.array([3, 2])
    A_ub = np.array([
        [-1, -1],
        [-2, -1],
        [-1, 0]
    ])
    b_ub = np.array([-4, -6, -2])

    try:
        dual_simplex = DualSimplex(c, A_ub, b_ub)
        solution, value = dual_simplex.solve()
        print("Dual simplex optimal solution:", solution)
        print("Dual simplex optimal value:", value)
        print("Final tableau:\n", dual_simplex.tableau)
    except Exception as e:
        print("Error:", e)

def _test10():
    """
    Maximize: c = x1 + x2
    Subject to: x1 - x2 >= 0
                x1 - x2 <= 0
                x1 + x2 <= 4
                x1, x2 >= 0
    """
    c = np.array([1, 1])
    A_ub = np.array([
        [-1, 1],
        [1, -1],
        [1, 1]
    ])
    b_ub = np.array([0, 0, 4])


    try:
        dual_simplex = DualSimplex(c, A_ub, b_ub)
        solution, value = dual_simplex.solve()
        print("Dual simplex optimal solution:", solution)
        print("Dual simplex optimal value:", value)
        print("Final tableau:\n", dual_simplex.tableau)
    except Exception as e:
        print("Error:", e)

def _test11():
    """
    Maximize: c = x1 + x2
    Subject to: x1 + x2 >= 5
                x1, x2 >= 0
    """
    c = np.array([1, 1])
    A_ub = np.array([
        [-1, -1]
    ])
    b_ub = np.array([-5])

    try:
        dual_simplex = DualSimplex(c, A_ub, b_ub)
        solution, value = dual_simplex.solve()
        print("Dual simplex optimal solution:", solution)
        print("Dual simplex optimal value:", value)
        print("Final tableau:\n", dual_simplex.tableau)
    except Exception as e:
        print("Error:", e)

def _test12():
    """
    Maximize: c = 0
    Subject to: x1 - x2 <= -1
                -x1 + x2 <= 3
                x1, x2 >= 0
    """
    c = np.array([0, 0])
    A_ub = np.array([
        [1, -1],
        [-1, 1]
    ])
    b_ub = np.array([-1, 3])

    try:
        dual_simplex = DualSimplex(c, A_ub, b_ub)
        solution, value = dual_simplex.solve()
        print("Dual simplex optimal solution:", solution)
        print("Dual simplex optimal value:", value)
        print("Final tableau:\n", dual_simplex.tableau)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    # print("Running Simplex tests...")
    # _test1()
    # _test2()
    # _test3()

    # print("\nRunning Two-Phase Simplex tests...")
    # _test4()
    # _test5()
    # _test6()

    print("\nRunning Dual Simplex test...")
    _test7()
    _test8()
    _test9()
    _test10()
    _test11()
    _test12()