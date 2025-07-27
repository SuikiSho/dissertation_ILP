import numpy as np
from scipy.optimize import linprog

TOL = 1e-10


class Ys:
    """
    
    """
    def __init__(self, index_set):
        self.index_set = tuple(sorted(index_set))

    def __repr__(self):
        if not self.index_set:
            return "y_∅"
        return "y_{" + ",".join(map(str, self.index_set)) + "}"

    def __eq__(self, other):
        return self.index_set == other.index_set

    def __lt__(self, other):
        # Compare set lengths first, then dictionary order
        if len(self.index_set) != len(other.index_set):
            return len(self.index_set) < len(other.index_set)
        return self.index_set < other.index_set

    def __hash__(self):
        return hash(self.index_set)

class SheraliAdams:
    """
    
    """
    def __init__(self, c, A_ub, b_ub, A_eq=None, b_eq=None):
        self.c = c                  # objective: maximize c^T x
        self.num_origin = len(c)    # number of original variables

        self.ys = [Ys(())] + [Ys((i, )) for i in range(1, self.num_origin+1)]
        self.constraints_ub = []    # List of (coef_dict, rhs)
        self.constraints_eq = []
        self._matrix_to_Constraints(A_ub, b_ub, A_eq, b_eq)

    def _is_integral(self, x):
        return np.all(np.isclose(x, np.round(x), atol=TOL))

    def _matrix_to_Constraints(self, A_ub, b_ub, A_eq, b_eq):
        # Convert matrix form constraints into dict form: (coef_dict, rhs)

        # Inequalities (A_ub x <= b_ub → b - A x >= 0)
        for i, row in enumerate(A_ub):
            coef_dict ={}
            coef_dict[self.ys[0]] = b_ub[i]
            for j, coef in enumerate(row):
                if abs(coef) > TOL:
                    coef_dict[self.ys[j+1]] = -coef
            self.constraints_ub.append((coef_dict, 0))  # b - A x >= 0

        # Equalities (A_eq x = b_eq → b - A x = 0)
        if A_eq is None and b_eq is None:
            return
        for i, row in enumerate(A_eq):
            coef_dict ={}
            coef_dict[self.ys[0]] = b_eq[i]
            for j, coef in enumerate(row):
                if abs(coef) > TOL:
                    coef_dict[self.ys[j+1]] = -coef
            self.constraints_eq.append((coef_dict, 0))  # b - A x = 0

    def _lift_single(self, constraints, i):
        new_constraints = []
        new_vars = set()

        for coef_dict, rhs in constraints:
            # times x_i
            lifted = {}
            for var, coef in coef_dict.items():
                S = set(var.index_set)
                new_S = tuple(sorted(S | {i}))
                new_y = Ys(new_S)
                lifted[new_y] = lifted.get(new_y, 0) + coef
                if new_y not in self.ys:
                    new_vars.add(new_y)
            lifted = {k: v for k, v in lifted.items() if abs(v) > TOL}  # remove items with zero coef
            new_constraints.append((lifted, rhs))

            # times 1 - x_i
            lifted = {}
            for var, coef in coef_dict.items():
                # remain oringinal (times 1)
                lifted[var] = lifted.get(var, 0) + coef 
                # minus multiplier items
                S = set(var.index_set)
                new_S = tuple(sorted(S | {i}))
                new_y = Ys(new_S)
                lifted[new_y] = lifted.get(new_y, 0) - coef
                if new_y not in self.ys:
                    new_vars.add(new_y)
            lifted = {k: v for k, v in lifted.items() if abs(v) > TOL}  # remove items with zero coef
            new_constraints.append((lifted, rhs))

        return new_constraints, new_vars

    def _add_envelope(self, new_vars, i):
        # Add envelope constraints
        enve_constraints = []
        y_i = self.ys[i]
        y_0 = self.ys[0]

        for y_S in new_vars:
            S = y_S.index_set
            if not S or len(S) == 1:
                # y_∅ and y_i don't need envelope
                continue
            y_Si = Ys(tuple(sorted(set(S) - {i})))
            
            constraints = [
                ({y_S: -1, y_i: 1}, 0),                     # y_S <= y_i
                ({y_S: -1, y_Si: 1}, 0),                    # y_S <= y_(S - {i})
                ({y_S: 1, y_Si: -1, y_i: -1, y_0: 1}, 0),   # y_S >= y_i + y_(S - {i}) - 1
                ({y_S: 1}, 0),                              # y_S >= 0
            ]
            enve_constraints.extend(constraints)

        return enve_constraints

    def _format_term(self, v, y):
        if v > 0:
            return f'+ {v:g}·{y}' if abs(v - 1) > TOL else f'+ {y}'
        else:
            return f'- {abs(v):g}·{y}' if abs(v + 1) > TOL else f'- {y}'

    def print_status(self):
        print("\n=== Inequality Constraints (≥) ===")
        for coef, rhs in self.constraints_ub:
            expr = ' '.join(self._format_term(v, y) for y, v in sorted(coef.items()))
            print(f'{expr} >= {rhs}')

        print("\n=== Equality Constraints (=) ===")
        for coef, rhs in self.constraints_eq:
            expr = ' '.join(self._format_term(v, y) for y, v in sorted(coef.items()))
            print(f'{expr} >= {rhs}')

        print("\n=== Variables ===")
        print(" ".join(str(y) for y in self.ys))
        print()

    def solve_rlp(self):
        # Solve the current relaxed LP using scipy.optimize.linprog.
        n = len(self.ys)                # total variables (including y_empty)
        c = np.zeros(n - 1)
        c[:self.num_origin] = -self.c   # maximize → minimize
        ys_idx = {var: i for i, var in enumerate(self.ys)}

        A_ub, b_ub = [], []
        A_eq, b_eq = [], []

        for coef_dict, rhs in self.constraints_ub:
            row = np.zeros(n)
            for var, coef in coef_dict.items():
                idx = ys_idx[var]
                row[idx] = coef
            A_ub.append(-row[1:])  # remove y_∅
            b_ub.append(row[0])

        for coef_dict, rhs in self.constraints_eq:
            row = np.zeros(n)
            for var, coef in coef_dict.items():
                idx = ys_idx[var]
                row[idx] = coef
            A_eq.append(-row[1:])
            b_eq.append(row[0])

        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None

        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0,1), method='highs')
        if not res.success:
            print("LP solve failed:", res.message)
            return None, None

        return res.x, -res.fun

    def lift(self, i):
        # Generate the new variables and constraints by multiplying x_i and 1 - x_i
        new_ub, new_vars_ub = self._lift_single(self.constraints_ub, i)
        new_eq, new_vars_eq = self._lift_single(self.constraints_eq, i)

        self.constraints_ub.extend(new_ub)
        self.constraints_eq.extend(new_eq)

        # Add envelope constraints
        new_vars = list(new_vars_ub | new_vars_eq)
        self.ys = sorted(self.ys + new_vars)

        enve_constraints = self._add_envelope(new_vars, i)
        self.constraints_ub.extend(enve_constraints) 

    def solve(self):
        size  = self.num_origin
        max_l = size
        level = 0

        while True:
            sol, obj = self.solve_rlp()

            # for debugging
            print(f"Current level: {level}")
            print(f"Current LP solution: {sol}")
            print(f"Current LP objective: {obj}")

            if sol is None:
                print("LP infeasible or error. Terminating.")
                return None, None
            if self._is_integral(sol):
                print(f"Integral solution found at level {level}.")
                return sol[:size], obj
            if level >= max_l:
                print(f"Reached max lifting level {level}.")
                return sol[:size], obj
            level += 1
            self.lift(level)



def _test_lift():
    A_ub = np.array([[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
                    [-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [-1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,-1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,1,-1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,1,0,-1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [-1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,-1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,-1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,1,-1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [-1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,-1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,-1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                    [1,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,1,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,1,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,-1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,-1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,-1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,-1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,-1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,-1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,-1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,1,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,-1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,-1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,-1,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,-1,1,0,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,1,0,0,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,-1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,1,0,0,0,-1,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,1,0,0,0,-1,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,-1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,-1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],
                    [-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,0,0],
                    [0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0],
                    [0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0],
                    [0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                    [0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,0],
                    [0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0],
                    [0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0],
                    [0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0],
                    [0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,1,0,0],
                    [0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,1,-1,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,1,1,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,-1,-1],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,-1,1],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,1,-1],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,1,1]
                    ])
    b_ub = np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,0,0,1,-1,0,0,1,-1,0,0,1,-1,0,0,1])
    c = np.array([1]*24)
    model = SheraliAdams(c, A_ub, b_ub)

    print("Initial constraints and ys:")
    model.print_status()

    print("Lift to SA-1:")
    model.lift(1)
    model.print_status()

    print("Lift to SA-2:")
    model.lift(2)
    model.print_status()

def _test_rlp():
    c = np.array([1, 1])
    A_ub = np.array([[1,1], [-1, -1]])
    b_ub = np.array([1, -1])
    model = SheraliAdams(c, A_ub, b_ub)

    print("Initial constraints and ys:")
    model.print_status()

    sol, obj = model.solve_rlp()
    print("Initial rlp:")
    print(f"Optimal solution: {sol}; Optimal objective: {obj}")

    print("Lift to SA-1:")
    model.lift(1)
    model.print_status()

    sol, obj = model.solve_rlp()
    print("RLP SA-1:")
    print(f"Optimal solution: {sol}; Optimal objective: {obj}")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    # _test_lift()
    # _test_rlp()