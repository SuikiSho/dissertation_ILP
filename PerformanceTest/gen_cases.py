import os
OUT_DIR = os.path.join(os.path.dirname(__file__), "cases")
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np

# =========================
# Config
# =========================
MODE = "vary_n"                     # "vary_n" or "vary_m"
FAMILIES = ["cover"]             # "knapsack", "packing", "cover"
SEED_BASE = 2025
OUT_FILE = os.path.join(OUT_DIR, "c_n2_20(100).npz")
N_PER_SCALE = 100                    # number of cases per scale

# vary_n params
N_START = 2
N_END = 20

# vary_m params
N_FIXED = 5
M_START = 1
M_END = 10

# --- Feasibility control ---
FEAS_MODE = "witness"               # "witness" or "zero"
#   - "zero": All constraints are ≤, A >= 0, b >= 0, so x=0 is feasible
#   - "witness": sample a witness x*, then set b := A x* + margin, so x* is feasible
WITNESS_DENSITY = 0.4               # probability for each variable to be 1 in witness x*
WITNESS_MARGIN  = 1.0               # extra slack added to b
# knapsack specific
KNAPSACK_CAP_RATIO = 0.5            # only used if FEAS_MODE="zero"
KNAPSACK_INTEGER_CAP = True         # floor/ceil capacity to integer

# =========================
# Helpers
# =========================
def make_feasible_b(A: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_star, b) that makes A x <= b feasible by construction."""
    n = A.shape[1]
    x_star = (rng.random(n) < WITNESS_DENSITY).astype(float)
    b = A @ x_star + WITNESS_MARGIN
    return x_star, b.astype(float)

def ensure_column_coverage(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make sure every column has at least one nonzero (prevents unbounded variables)."""
    m, n = A.shape
    for j in range(n):
        if np.allclose(A[:, j], 0.0):
            i = rng.integers(0, m)
            A[i, j] = 1.0
    return A

def add_box_01(A: np.ndarray, b: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Append box 0 <= x <= 1 as ≤-form rows:  x<=1  (I),  -x<=0  (-I)."""
    A2 = np.vstack([A,  np.eye(n), -np.eye(n)])
    b2 = np.concatenate([b, np.ones(n), np.zeros(n)])
    return A2, b2

# =========================
# Generators (≤-form)
# =========================
def gen_knapsack(n: int, m: int = 1, seed: int | None = None):
    rng = np.random.default_rng(seed)
    c = rng.integers(1, 10, size=n).astype(float)
    A = rng.integers(1, 10, size=(m, n)).astype(float)

    if FEAS_MODE == "witness":
        _, b = make_feasible_b(A, rng)
        if KNAPSACK_INTEGER_CAP:
            b = np.ceil(b)
    else:
        rowsum = np.sum(A, axis=1).astype(float)
        b = (KNAPSACK_CAP_RATIO * rowsum).astype(float)
        if KNAPSACK_INTEGER_CAP:
            b = np.floor(b)

    # --- ensure boundedness ---
    A, b = add_box_01(A, b, n)
    return c, A, b


def gen_packing(n: int, m: int | None = None, seed: int | None = None):
    rng = np.random.default_rng(seed)
    if m is None:
        m = max(1, n // 2)
    c = rng.integers(1, 5, size=n).astype(float)
    A = rng.integers(0, 3, size=(m, n)).astype(float)

    # fix zero columns BEFORE computing b
    ensure_column_coverage(A, rng)

    if FEAS_MODE == "witness":
        _, b = make_feasible_b(A, rng); b = np.ceil(b)
    else:
        b = rng.integers(2, 6, size=m).astype(float)

    # boundedness
    A, b = add_box_01(A, b, n)
    return c, A, b


def gen_cover(n: int, m: int | None = None, seed: int | None = None):
    rng = np.random.default_rng(seed)
    if m is None:
        m = max(1, int(0.6 * n))
    c = rng.integers(1, 5, size=n).astype(float)

    # ensure each row has at least one 1
    A = np.zeros((m, n), dtype=float)
    for i in range(m):
        k = rng.integers(1, max(2, n // 3))
        cols = rng.choice(n, size=k, replace=False)
        A[i, cols] = 1.0

    # fix zero columns BEFORE computing b
    ensure_column_coverage(A, rng)

    if FEAS_MODE == "witness":
        _, b = make_feasible_b(A, rng); b = np.ceil(b)
    else:
        b = np.ones(m, dtype=float)

    # boundedness
    A, b = add_box_01(A, b, n)
    return c, A, b


FAMILY_FUNCS = {
    "knapsack": gen_knapsack,
    "packing":  gen_packing,
    "cover":    gen_cover,
}

# =========================
# Save / Load
# =========================
def save_cases(filename: str, cases: dict):
    payload = {}
    for name, (c, A_ub, b_ub) in cases.items():
        arr = np.empty(3, dtype=object)
        arr[0], arr[1], arr[2] = c, A_ub, b_ub
        payload[name] = arr
    np.savez_compressed(filename, **payload)
    print(f"[OK] Saved {len(cases)} cases to {filename}")

def load_cases(filename: str) -> dict:
    data = np.load(filename, allow_pickle=True)
    return {name: tuple(data[name]) for name in data.files}

# =========================
# Case generation modes
# =========================
def generate_vary_n(n_start: int, n_end: int, families: list[str], seed_base: int):
    cases = {}
    for i, n in enumerate(range(n_start, n_end + 1)):
        for fam in families:
            gen = FAMILY_FUNCS[fam]
            for rep in range(N_PER_SCALE):
                key = f"{fam}_n{n}_seed{rep:02d}"
                cases[key] = gen(n=n, seed=seed_base + 1000 * i + rep)
    return cases

def generate_vary_m(n_fixed: int, m_start: int, m_end: int, families: list[str], seed_base: int):
    cases = {}
    for j, m in enumerate(range(m_start, m_end + 1)):
        for fam in families:
            gen = FAMILY_FUNCS[fam]
            for rep in range(N_PER_SCALE):
                key = f"{fam}_n{n_fixed}_m{m}_seed{rep:02d}"
                cases[key] = gen(n=n_fixed, m=m, seed=seed_base + 2000 * j + rep)
    return cases

# =========================
# Main
# =========================
def main():
    fams = [f.lower() for f in FAMILIES]
    for f in fams:
        if f not in FAMILY_FUNCS:
            raise ValueError(f"Unknown family: {f}")

    if MODE == "vary_n":
        cases = generate_vary_n(N_START, N_END, fams, SEED_BASE)
    elif MODE == "vary_m":
        cases = generate_vary_m(N_FIXED, M_START, M_END, fams, SEED_BASE)
    else:
        raise ValueError("MODE must be 'vary_n' or 'vary_m'")

    save_cases(OUT_FILE, cases)

if __name__ == "__main__":
    main()
