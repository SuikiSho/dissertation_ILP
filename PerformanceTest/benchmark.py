import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# PerformanceTest/benchmark_min.py
import time
import numpy as np
import pandas as pd

# ---- setup input/output folders ----
BASE_DIR = os.path.dirname(__file__)
NPZ_DIR = os.path.join(BASE_DIR, "cases")
CSV_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(NPZ_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# ---- choose input/output and solvers here ----
NPZ_FILE = os.path.join(NPZ_DIR, "c_n2_20.npz")
OUT_CSV  = os.path.join(CSV_DIR, "c_n2_20.csv")
SOLVERS_TO_RUN = ["BranchAndBound", "GomoryCuttingPlane", "SheraliAdams"]   # "BranchAndBound", "GomoryCuttingPlane", "SheraliAdams"

# ---- import solver classes ----
from ILPSolverModels.BranchAndBound.branch_and_bound import BranchAndBound
from ILPSolverModels.CuttingPlane.gomory import GomoryCuttingPlane
from ILPSolverModels.LiftAndProject.sherali_adams import SheraliAdams

SOLVER_CLASSES = {
    "BranchAndBound": BranchAndBound,
    "GomoryCuttingPlane": GomoryCuttingPlane,
    "SheraliAdams": SheraliAdams,
}

def run_solver(SolverCls, c, A, b):
    try:
        solver = SolverCls(c=c, A_ub=A, b_ub=b, A_eq=None, b_eq=None)
        sol, val = solver.solve()
        return (val, sol)
    except Exception as e:
        return f"fail: {type(e).__name__}: {e}"

def main():
    data = np.load(NPZ_FILE, allow_pickle=True)
    rows = []

    for key in sorted(data.files):
        entry = data[key]
        if isinstance(entry, np.ndarray) and entry.dtype == object and entry.shape == (3,):
            c, A_ub, b_ub = entry[0], entry[1], entry[2]
        elif isinstance(entry, (tuple, list)) and len(entry) == 3:
            c, A_ub, b_ub = entry
        else:
            rows.append({"key": key, "solver": None, "time_sec": None, "obj_val": "malformed"})
            continue

        for name in SOLVERS_TO_RUN:
            print(f"Running {name} on {key}...")
            t0 = time.perf_counter()
            obj = run_solver(SOLVER_CLASSES[name], c, A_ub, b_ub)
            elapsed = time.perf_counter() - t0
            rows.append({"key": key, "solver": name, "time_sec": elapsed, "obj_val": obj})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[OK] saved -> {OUT_CSV}")

if __name__ == "__main__":
    main()
