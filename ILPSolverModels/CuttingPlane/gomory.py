import numpy as np
from .simplex import Simplex, TwoPhaseSimplex, DualSimplex

TOL = 1e-6  # floating tolerance

def _frac(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)

def _efficacy(a: np.ndarray, viol: float, eps: float = 1e-12) -> float:
    return float(viol) / (np.linalg.norm(a) + eps)

def _cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(a, b)) / ((np.linalg.norm(a)+eps)*(np.linalg.norm(b)+eps))


class GomoryCuttingPlane(Simplex):
    def __init__(self, c, A_ub, b_ub, A_eq=None, b_eq=None):
        super().__init__(c=c, A_ub=A_ub, b_ub=b_ub)
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.tableau = None
        self.basic_index = None

        # ---- cut management (NEW) ----
        self.cut_pool = []         # store (a, rhs) to avoid near-duplicates
        self._last_obj = None
        self._stagn_rounds = 0

    # ------ helpers ------
    def _is_integer(self):
        """Check integer-ness of basic original variables."""
        rhs = self.tableau[1:, -1]                     # basic values per row
        idx = self.basic_index                         # index of basic var in each row
        mask = idx < len(self.c)                       # only original vars (not slacks)
        if not np.any(mask):
            return True
        vals = rhs[mask]
        return np.all(np.isclose(vals, np.round(vals), atol=TOL))

    def _choose_fractional_rows(self, tol: float = 1e-8):
        """Return candidate row indices (in tableau) sorted by fractional RHS desc."""
        rhs = self.tableau[1:, -1]
        mask = (self.basic_index < len(self.c)) & (~np.isclose(rhs, np.round(rhs), atol=tol))
        if not np.any(mask):
            return []
        idxs = np.where(mask)[0]               # row indices in 1..m (0-based on the 1: slice)
        fracs = _frac(rhs[idxs])
        order = np.argsort(-fracs)             # descending by fractional part
        # convert to tableau row indices (+1 for skipping objective row)
        return [int(idxs[k] + 1) for k in order]

    def _is_near_duplicate(self, a_new: np.ndarray, rhs_new: float,
                        cos_th: float = 0.995, rhs_tol: float = 1e-7) -> bool:
        """
        Check near-duplicate cuts against the pool.
        The tableau grows by adding slack columns over rounds, so previously stored
        cuts may have shorter length. We align by zero-padding the shorter one.
        """
        for a_old, rhs_old in self.cut_pool:
            # --- align lengths by zero-padding the shorter vector ---
            if a_new.shape[0] != a_old.shape[0]:
                n = max(a_new.shape[0], a_old.shape[0])
                a1 = np.zeros(n, dtype=float)
                a2 = np.zeros(n, dtype=float)
                a1[:a_new.shape[0]] = a_new
                a2[:a_old.shape[0]] = a_old
            else:
                a1, a2 = a_new, a_old

            # skip degenerate vectors
            n1 = np.linalg.norm(a1)
            n2 = np.linalg.norm(a2)
            if n1 < 1e-12 or n2 < 1e-12:
                continue

            cos = float(np.dot(a1, a2) / (n1 * n2))
            if abs(cos) > cos_th and abs(rhs_new - rhs_old) < rhs_tol:
                return True
        return False
    
    def _append_cut_row(self, cut_coeffs: np.ndarray, cut_rhs: float):
        """Append the cut to tableau and also to (A_ub, b_ub) over original variables."""
        # append to tableau (add a new slack column)
        T = np.zeros((self.tableau.shape[0] + 1, self.tableau.shape[1] + 1))
        T[:-1, :-2] = self.tableau[:, :-1]
        T[:-1,  -1] = self.tableau[:, -1]
        T[-1,  :-2] = cut_coeffs
        T[-1,  -2]  = 1.0
        T[-1,  -1]  = cut_rhs
        self.tableau = T
        self.basic_index = np.append(self.basic_index, self.tableau.shape[1] - 2)

        # also update the model in original variable space for fallback re-solve
        coeffs_x = cut_coeffs[:len(self.c)]          # drop slack parts
        self.A_ub = np.vstack([self.A_ub, coeffs_x])
        self.b_ub = np.append(self.b_ub, cut_rhs)

        # record for duplicate detection
        self.cut_pool.append((cut_coeffs.copy(), float(cut_rhs)))

    # ------ LP relax / dual repair ------
    def solve_lp_relaxation(self):
        lp = TwoPhaseSimplex(c=self.c, A_ub=self.A_ub, b_ub=self.b_ub,
                             A_eq=self.A_eq, b_eq=self.b_eq)
        lp.solve()
        self.tableau = lp.tableau
        self.basic_index = lp.basic_index

    def solve_with_dual_simplex(self):
        try:
            dual = DualSimplex(c=None, A_ub=None, b_ub=None,
                            tableau=self.tableau, basic_index=self.basic_index)
            dual.solve()
            self.tableau = dual.tableau
            self.basic_index = dual.basic_index
        except Exception:
            # fallback: re-solve from scratch on expanded (A_ub, b_ub)
            lp = TwoPhaseSimplex(c=self.c, A_ub=self.A_ub, b_ub=self.b_ub,
                                A_eq=self.A_eq, b_eq=self.b_eq)
            lp.solve()
            self.tableau = lp.tableau
            self.basic_index = lp.basic_index

    # ------ cut construction with filters ------
    def _try_add_cut_from_row(self, cut_row_idx: int,
                            cut_tol: float = 1e-7,
                            eff_min: float = 1e-3) -> bool:
        """
        Build a GMI cut from tableau row `cut_row_idx`. If the GMI degenerates,
        fall back to a CG rounding cut. Ensure dual-pivotability before appending.
        Returns True if a cut is appended.
        """
        row = self.tableau[cut_row_idx, :].copy()
        n_cols = self.tableau.shape[1] - 1              # exclude RHS
        basic_col = int(self.basic_index[cut_row_idx - 1])

        # Only meaningful if the basic var is an original integer var
        if basic_col >= len(self.c):
            return False

        # canonical coefficients for nonbasic variables:
        # x_B = b – sum_j a_nb[j] * x_j  =>  a_nb[j] = - row[j]
        a_nb = -row[:-1].copy()
        a_nb[basic_col] = 0.0
        b = float(row[-1])
        f = b - np.floor(b)

        # need genuinely fractional RHS
        if not (1e-12 < f < 1.0 - 1e-12):
            return False

        # ---- Try GMI first ----
        denom_pos, denom_neg = (1.0 - f), f
        if denom_pos <= 1e-12 or denom_neg <= 1e-12:
            return False

        alpha = np.zeros(n_cols, dtype=float)
        for j in range(n_cols):
            if j == basic_col:
                continue
            a = float(a_nb[j])
            if a >= 0.0:
                a_frac = a - np.floor(a)
                alpha[j] = a_frac / denom_pos
            else:
                a_frac = a - np.floor(a)          # in [0,1)
                one_minus_frac = 1.0 - a_frac     # = ceil(a) - a
                alpha[j] = one_minus_frac / denom_neg

        # ≤-form for our solver: (-alpha) x ≤ -1
        cut_coeffs = -alpha
        cut_rhs = -1.0

        # ---- Degeneracy: alpha ~ 0  -> fall back to CG rounding ----
        if np.linalg.norm(alpha) < 1e-12:
            cut_coeffs = np.zeros(n_cols, dtype=float)
            cut_coeffs[basic_col] = 1.0
            for j in range(n_cols):
                if j == basic_col:
                    continue
                # use the raw tableau row to align with current columns
                cut_coeffs[j] = np.floor(row[j] + 1e-12)
            cut_rhs = np.floor(b + 1e-12)

        # ---- Dual-pivotability check ----
        # If RHS is negative, dual simplex needs at least one negative coeff
        # (excluding the new slack we will add) to enter.
        if (cut_rhs < 0.0) and not np.any(cut_coeffs < -1e-12):
            return False

        # violation / efficacy filters
        viol = 1.0 if cut_rhs == -1.0 else (b - np.floor(b))
        eff = _efficacy(cut_coeffs, viol)
        if not (viol > cut_tol and eff > eff_min):
            return False

        # near-duplicate filter (handles different widths by zero-padding)
        if self._is_near_duplicate(cut_coeffs, float(cut_rhs)):
            return False

        # ---- Append to tableau & model (original variable space) ----
        self._append_cut_row(cut_coeffs, cut_rhs)
        return True


    # ------ main loop with anti-storm guards ------
    def solve(self,
              max_rounds: int = 50,
              top_k: int = 2,
              cut_tol: float = 1e-8,
              eff_min: float = 1e-3,
              int_tol: float = 1e-6,
              obj_improve_tol: float = 1e-8,
              stagn_limit: int = 3):
        """
        Anti-storm strategy:
        - each round, pick at most `top_k` best fractional rows to try cuts
        - filter by violation & efficacy
        - drop near-duplicate cuts
        - stop if no cut added in a round, or objective stagnates `stagn_limit` rounds
        """
        self.solve_lp_relaxation()

        for r in range(1, max_rounds + 1):
            # integrality check
            if self._is_integer():
                break

            # candidates by largest fractional RHS
            cand_rows = self._choose_fractional_rows()
            added = 0
            for ridx in cand_rows[:top_k]:
                if self._try_add_cut_from_row(ridx, cut_tol=cut_tol, eff_min=eff_min):
                    added += 1

            # nothing helpful this round -> stop to avoid storm
            if added == 0:
                break

            # dual simplex repair
            prev_obj = float(self.tableau[0, -1])
            self.solve_with_dual_simplex()
            curr_obj = float(self.tableau[0, -1])

            # stagnation monitor
            if self._last_obj is None:
                self._last_obj = curr_obj
            else:
                if abs(curr_obj - self._last_obj) < obj_improve_tol:
                    self._stagn_rounds += 1
                else:
                    self._stagn_rounds = 0
                self._last_obj = curr_obj

            if self._stagn_rounds >= stagn_limit:
                break

        # extract solution (same format as your original)
        rhs = np.round(self.tableau[1:, -1])
        x = np.zeros(len(self.c))
        for i, idx in enumerate(self.basic_index):
            if idx < len(self.c):
                x[idx] = rhs[i]
        optimal_value = float(self.c @ x)

        return x, optimal_value

    

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)  # Set print options for better readability
    """
    Gomory Cutting Plane Example Usage
    Example usage
    Maximize:   40x1 + 50x2
    Subject to: 2x1 + 3x2 <= 12
                3x1 + 1x2 <= 9
                x1, x2 are integers
    This example demonstrates how to use the Gomory Cutting Plane method to solve an integer linear programming problem.
    The solution should be (0, 4) with an optimal value of 200.
    """
    c = np.array([40, 50])
    A_ub = np.array([[2, 3], [3, 1]])
    b_ub = np.array([12, 9])
    
    gomory = GomoryCuttingPlane(c=c, A_ub=A_ub, b_ub=b_ub)
    solution, value = gomory.solve()
    
    print("Optimal solution:", solution)
    print("Optimal value:", value)
    print("Final tableau:\n", gomory.tableau)


