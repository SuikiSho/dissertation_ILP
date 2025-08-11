import numpy as np
from gomory import GomoryCuttingPlane

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
    


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)  # Set print options for better readability

    _test1()
    _test2()
    _test3()
    _test4()
    _test5()