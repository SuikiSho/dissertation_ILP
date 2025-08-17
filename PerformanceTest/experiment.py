import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import time

from ILPSolverModels.BranchAndBound import BranchAndBound as BnB
from ILPSolverModels.CuttingPlane import GomoryCuttingPlane as GCP
from ILPSolverModels.LiftAndProject import SheraliAdams as SA



def _test_bnb():
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

    bnb = BnB(c, A_ub, b_ub, A_eq, b_eq)
    opt_sol, opt_val = bnb.solve()
    # bnb.print_tree()
    print("Optimal Solution:", opt_sol)
    print("Optimal Value:", opt_val)

def _test_gomory():
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

    gomory = GCP(c, A_ub, b_ub, A_eq, b_eq)
    opt_sol, opt_val = gomory.solve()
    print("Optimal Solution:", opt_sol)
    print("Optimal Value:", opt_val)
    print("Finial tableau:", gomory.tableau)

def _test_sa():
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

    sa = SA(c, A_ub, b_ub, A_eq, b_eq)
    sa.print_status()
    opt_sol, opt_val = sa.solve()
    print("Optimal Solution:", opt_sol)
    print("Optimal Value:", opt_val)

if __name__ == "__main__":
    # _test_bnb()
    # _test_gomory()
    _test_sa()