import numpy as np
import sympy as sp
from scipy.optimize import linprog

TOL = 1e-10


class LiftAndProject:
    """
    Implements a Lift-and-Project (BCC/SA) algorithm using numpy, scipy, and sympy.
    Constraints are stored as Sympy inequalities: lhs >= 0.
    """
    def __init__(self, c, A_ub, b_ub, A_eq=None, b_eq=None, maxlayers=None):
        self.c = c          # default as maximize
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.maxlayers = maxlayers

        self.x = sp.symbols(' '.join(f"x_{i}" for i in range(len(self.c))))
        self.constraints = self._arr_to_expr(self.A_ub, self.b_ub)
        # self.current_rlp_all_vars = None
        # self.current_rlp_sympy_lhs = None
        self.cur_res = self._rlp(self.constraints, None)

    def _arr_to_expr(self, A_ub, b_ub):
        # Convert the matrix and vector to expressions.
        m, n = A_ub.shape
        constraints = []

        for i in range(m):
            expr = float(b_ub[i])
            for j in range(n):
                coef = float(A_ub[i, j])
                expr -= coef * self.x[j] if abs(coef) > TOL else 0
            constraints.append(expr >= 0)
        return constraints
        
    def _rlp(self, constraints, ys=None):
        # Convert the constraints to a matrix 
        A_ub = []
        b_ub = []
        for cons in constraints:
            expr = sp.expand(cons.lhs)
            coeffs_dict = expr.as_coefficients_dict()
            coes_x = np.array([float(coeffs_dict.get(xj, 0)) for xj in self.x])
            coes_y = np.array([float(coeffs_dict.get(y_v, 0)) for y_v in ys]) if ys is not None else np.array([])
            b_val = float(coeffs_dict.get(sp.S.One, 0)) # 提取常数项
            A_ub.append(np.hstack([-coes_x, -coes_y]))
            b_ub.append(b_val)
        A_ub = np.array(A_ub, dtype=float)
        b_ub = np.array(b_ub, dtype=float)
        c = np.zeros(A_ub.shape[1])
        c[:len(self.c)] = -self.c

        # # Store the order of all variables and the sympy lhs.
        # self.current_rlp_all_vars = list(self.x)
        # if ys is not None:
        #     self.current_rlp_all_vars.extend(ys)
        # self.current_rlp_sympy_lhs = [sp.expand(cons.lhs) for cons in constraints]

        # Solve the RLP problem for a given layer.
        res = linprog(
            c=c, 
            A_ub=A_ub, b_ub=b_ub, 
            # A_eq=self.A_eq, b_eq=self.b_eq, 
            bounds=[(0,1)]*A_ub.shape[1], 
            method="highs")
        return res

    def lift(self, constraints, i):
        xi = self.x[i]
        new_ys = []
        new_ineqs = []

        # For each inequality f(x,y)>=0, multiply by xi and (1-xi)
        for cons in constraints:
            f = sp.simplify(cons.lhs)
            f1 = sp.expand(xi * f)
            f2 = sp.expand((1 - xi) * f)

            # Substitute xi*xj -> y_i_j
            for j, xj in enumerate(self.x):
                if j == i:
                    f1 = f1.xreplace({xi*xi: xi})
                    f2 = f2.xreplace({xi*xi: xi}) 
                yij = sp.symbols(f'y_{i}_{j}')

                if f1.has(xi * xj) or f1.has(xj * xi) or \
                   f2.has(xi * xj) or f2.has(xj * xi):
                    if yij not in new_ys: new_ys.append(yij)
                    f1 = f1.xreplace({xi*xj: yij, xj*xi: yij})
                    f2 = f2.xreplace({xi*xj: yij, xj*xi: yij})
            new_ineqs.extend([f1 >= 0, f2 >= 0])

        # Sort new_ys to ensure consistent ordering
        new_ys.sort(key=lambda s: s.name)

        # Add McCormick for new y
        for yij in new_ys:
            _, _, j = yij.name.split('_')
            xj = self.x[int(j)]
            new_ineqs.extend([
                xi - yij >= 0,              # yij <= xi
                xj - yij >= 0,              # yij <= xj
                yij - xi - xj + 1 >= 0,     # yij >= xi + xj - 1
                yij >= 0                    # yij >= 0
            ])
        
        return new_ineqs, new_ys

    def project(self):
        pass

    def solve(self):
        pass        



if __name__ == "__main__":
