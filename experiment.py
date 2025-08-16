import numpy as np
import pandas as pd
import time

from ILPSolverModels.BranchAndBound import BranchAndBound
from ILPSolverModels.CuttingPlane import GomoryCuttingPlane
from ILPSolverModels.LiftAndProject import SheraliAdams


import numpy as np

cases = [
    {
        'n_vars': 1,
        'c': np.array([1.]),
        'A_ub': np.array([[ 1.],
                          [-1.]]),
        'b_ub': np.array([0.5, 0.])
    },
    {
        'n_vars': 2,
        'c': np.array([1., 1.]),
        'A_ub': np.array([[ 1.,  1.],
                          [-1.,  0.],
                          [ 0., -1.]]),
        'b_ub': np.array([1.5, 0., 0.])
    },
    {
        'n_vars': 3,
        'c': np.array([1., 1., 1.]),
        'A_ub': np.array([[ 1.,  1.,  1.],
                          [-1.,  0.,  0.],
                          [ 0., -1.,  0.],
                          [ 0.,  0., -1.]]),
        'b_ub': np.array([2.5, 0., 0., 0.])
    },
    {
        'n_vars': 4,
        'c': np.array([1., 1., 1., 1.]),
        'A_ub': np.array([[ 1.,  1.,  1.,  1.],
                          [-1.,  0.,  0.,  0.],
                          [ 0., -1.,  0.,  0.],
                          [ 0.,  0., -1.,  0.],
                          [ 0.,  0.,  0., -1.]]),
        'b_ub': np.array([3.5, 0., 0., 0., 0.])
    },
    {
        'n_vars': 5,
        'c': np.array([1., 1., 1., 1., 1.]),
        'A_ub': np.array([[ 1.,  1.,  1.,  1.,  1.],
                          [-1.,  0.,  0.,  0.,  0.],
                          [ 0., -1.,  0.,  0.,  0.],
                          [ 0.,  0., -1.,  0.,  0.],
                          [ 0.,  0.,  0., -1.,  0.],
                          [ 0.,  0.,  0.,  0., -1.]]),
        'b_ub': np.array([4.5, 0., 0., 0., 0., 0.])
    },
    {
        'n_vars': 6,
        'c': np.array([1., 1., 1., 1., 1., 1.]),
        'A_ub': np.array([[ 1.,  1.,  1.,  1.,  1.,  1.],
                          [-1.,  0.,  0.,  0.,  0.,  0.],
                          [ 0., -1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0., -1.,  0.,  0.,  0.],
                          [ 0.,  0.,  0., -1.,  0.,  0.],
                          [ 0.,  0.,  0.,  0., -1.,  0.],
                          [ 0.,  0.,  0.,  0.,  0., -1.]]),
        'b_ub': np.array([5.5, 0., 0., 0., 0., 0., 0.])
    },
    {
        'n_vars': 7,
        'c': np.array([1., 1., 1., 1., 1., 1., 1.]),
        'A_ub': np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
                          [-1.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0., -1.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0., -1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0., -1.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0., -1.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0., -1.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0., -1.]]),
        'b_ub': np.array([6.5, 0., 0., 0., 0., 0., 0., 0.])
    },
    {
        'n_vars': 8,
        'c': np.array([1., 1., 1., 1., 1., 1., 1., 1.]),
        'A_ub': np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                          [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.]]),
        'b_ub': np.array([7.5, 0., 0., 0., 0., 0., 0., 0., 0.])
    },
    {
        'n_vars': 9,
        'c': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]),
        'A_ub': np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                          [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.]]),
        'b_ub': np.array([8.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    },
    {
        'n_vars': 10,
        'c': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
        'A_ub': np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                          [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.]]),
        'b_ub': np.array([9.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    }
]

# Run timing test on the provided 'cases'
results = []
for case in cases:
    n_vars = case['n_vars']
    c, A_ub, b_ub = case['c'], case['A_ub'], case['b_ub']
    bounds = [(0, None)] * n_vars

    # Branch and Bound
    start = time.time()
    bnb = BranchAndBound(c, A_ub, b_ub)
    elapsed = time.time() - start
    results.append({'Algorithm': 'Branch and Bound', 'Size': n_vars, 'Time': elapsed})

    # Cutting Planes
    start = time.time()
    gomory = GomoryCuttingPlane(c, A_ub, b_ub)
    opt_sol, opt_val = gomory.solve()
    elapsed = time.time() - start
    results.append({'Algorithm': 'Cutting Planes', 'Size': n_vars, 'Time': elapsed})

    # Sherali-Adams
    start = time.time()
    sa = SheraliAdams(c, A_ub, b_ub)
    opt_sol, opt_val = sa.solve()
    elapsed = time.time() - start
    results.append({'Algorithm': 'Sherali-Adams', 'Size': n_vars, 'Time': elapsed})

df_results = pd.DataFrame(results)
import matplotlib.pyplot as plt

# Plotting the timing results
plt.figure(figsize=(10, 6))
for algo in df_results['Algorithm'].unique():
    subset = df_results[df_results['Algorithm'] == algo]
    plt.plot(subset['Size'], subset['Time'], label=algo, marker='o')

plt.xlabel('Problem Size (Number of Variables)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time of ILP Solvers on Fractional Cases')
plt.legend()
plt.grid(True)
plt.show()