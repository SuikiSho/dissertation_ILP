# Same as simplex.py in SimplexMethod

import numpy as np

class Simplex:
    """
    Simplex method for solving linear programming problems.
    This class implements the simplex algorithm to find the optimal solution
    for a given linear programming problem defined by the objective function
    coefficients, inequality constraints, and their right-hand side values.
    The problem is defined as:
        Maximize: c^T * x
        Subject to: A_ub * x <= b_ub
                    x >= 0
    where c is the coefficient vector for the objective function,
    A_ub is the matrix of coefficients for the inequality constraints,
    and b_ub is the right-hand side vector for the inequality constraints.
    The solution will return the optimal values of the variables and the optimal value of the objective function
    """
    def __init__(self, c, A_ub, b_ub, tableau=None, basic_index=None):
        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.tableau = tableau
        self.basic_index = basic_index

    def _construct_tableau(self):
        # If tableau is already provided, skip construction
        if self.tableau is not None:
            return

        # Create tableau with slack variables
        num_vars = len(self.c)
        num_constraints = self.A_ub.shape[0]
        tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
        tableau[0, 0:num_vars] = -self.c        # Objective function coefficients
        tableau[1:, 0:num_vars] = self.A_ub     # Coefficients of constraints
        tableau[1:, num_vars:num_vars + num_constraints] = np.eye(num_constraints)  # Slack variables
        tableau[1:, -1] = self.b_ub                  # Right-hand side values
        self.tableau = tableau
        self.basic_index = np.arange(num_vars, num_vars + num_constraints)

    def _is_optimal(self):
        # Check if the tableau is optimal (no negative coefficients in the first row)
        return np.all(self.tableau[0, :-1] >= 0)
    
    def _get_pivot(self):
        # Find pivot column (most negative in the first row)
        pivot_col = np.argmin(self.tableau[0, :-1])
        if self.tableau[0, pivot_col] >= 0:
            return False
        col_value = np.where(self.tableau[1:, pivot_col] == 0, -1, self.tableau[1:, pivot_col])  # Avoid division by zero

        # Find pivot row (minimum ratio test)
        ratios = self.tableau[1:, -1] / col_value
        ratios[ratios <= 0] = np.inf                # Ignore non-positive ratios
        pivot_row = np.argmin(ratios) + 1
        if np.isinf(ratios[pivot_row - 1]):
            raise ValueError("Problem is unbounded.")
        
        # Update basic variable index
        self.basic_index[pivot_row - 1] = pivot_col

        return pivot_row, pivot_col
    
    def _pivot(self, pivot_row, pivot_col):
        # Perform pivot operation
        pivot_value = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_value   # Normalize pivot row
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                self.tableau[i, :] -= self.tableau[pivot_row, :] * self.tableau[i, pivot_col]

    def solve(self):
        # Construct the initial tableau
        self._construct_tableau()

        # Check for negative right-hand side values
        if np.any(np.array(self.tableau[1:, -1]) < 0):
            raise ValueError("Right-hand side values must be non-negative.")

        # Perform the simplex algorithm
        while not self._is_optimal():
            pivot_row, pivot_col = self._get_pivot()
            self._pivot(pivot_row, pivot_col)

        # Extract the optimal solution and value
        rhs = self.tableau[1:, -1]
        optimal_value = self.tableau[0, -1]
        optimal_solution = np.zeros(self.tableau.shape[1] - 1)
        optimal_solution[self.basic_index] = rhs

        return optimal_solution, optimal_value
    

class TwoPhaseSimplex(Simplex):
    """
    Two-phase simplex method for solving linear programming problems.
    This method is used when the initial basic feasible solution is not obvious.
    It first solves an auxiliary problem to find a feasible solution,
    and then uses that solution to solve the original problem.
    The problem is defined as:
        Maximize: c^T * x
        Subject to: A_ub * x <= b_ub
                    A_eq * x = b_eq (default is None)
                    x >= 0
    where c is the coefficient vector for the objective function,
    A_ub is the matrix of coefficients for the inequality constraints,
    b_ub is the right-hand side vector for the inequality constraints,
    A_eq is the matrix of coefficients for the equality constraints (optional),
    and b_eq is the right-hand side vector for the equality constraints (optional).
    The solution will return the optimal values of the variables and the optimal value of the objective function
    if the problem is feasible.
    """
    def __init__(self, c, A_ub, b_ub, A_eq=None, b_eq=None, tableau=None, basic_index=None):
        super().__init__(c, A_ub, b_ub, tableau, basic_index)
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.num_vars = len(c)
        self.art_index = None

    def _construct_tableau(self):
        # If there are no equality constraints, use the parent class method
        if self.A_eq is None:
            super()._construct_tableau()
            return
                
        # If any equality constraint has a negative right-hand side, multiply the row by -1
        for i in range(len(self.b_eq)):
            if self.b_eq[i] < 0:
                self.A_eq[i, :] *= -1
                self.b_eq[i] *= -1

        # Number of variables and constraints
        num_vars = self.num_vars
        num_ub = self.A_ub.shape[0] if self.A_ub is not None else 0
        num_eq = self.A_eq.shape[0]
        num_constraints = num_ub + num_eq

        # Create tableau with slack and artificial variables
        A = np.vstack((self.A_ub, self.A_eq)) if self.A_ub is not None else self.A_eq
        b = np.hstack((self.b_ub, self.b_eq)) if self.b_ub is not None else self.b_eq
        tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
        tableau[1:, 0:num_vars] = A                                                 # Coefficients of constraints
        tableau[1:, -1] = b                                                         # Right-hand side values    
        tableau[1:, num_vars:num_vars + num_constraints] = np.eye(num_constraints)  # Slack variables and artificial variables
        tableau[0, :] = -np.sum(tableau[num_ub + 1:, :], axis=0)                    # Objective function coefficients
        self.basic_index = np.arange(num_vars, num_vars + num_constraints)
        self.art_index = np.arange(num_vars + num_ub, num_vars + num_constraints)

        for i in range(num_ub):
            if tableau[i + 1, -1] < 0:
                tableau[i + 1, :] *= -1
                tableau[0, :] -= tableau[i + 1, :] 
                new_col = np.eye(num_constraints + 1)[i + 1]
                tableau = np.insert(tableau, -1, new_col, axis=1)                           # Add a new column for the artificial variable
                self.basic_index[i] = num_vars + num_constraints + i                        # Update basic index
                self.art_index = np.append(self.art_index, num_vars + num_constraints + i)  # Add artificial variable

        tableau[0, num_vars + num_ub:-1] = 0    # Set coefficients of artificial variables to zero
        self.tableau = tableau

    # Construct the tableau for phase 2
    def _construct_phase2_tableau(self):
        # Remove artificial variables from the tableau
        A = self.tableau[1:, 0:self.num_vars + self.A_ub.shape[0]] if self.A_ub is not None else self.tableau[1:, 0:self.num_vars]
        b = self.tableau[1:, -1]
        c = self.c

        for i, idx in enumerate(self.basic_index):
            if idx in self.art_index:
                A = np.delete(A, i, axis=0)
                b = np.delete(b, i)
                self.basic_index = np.delete(self.basic_index, i)

        # Create the tableau for phase 2
        tableau = np.zeros((A.shape[0] + 1, A.shape[1] + 1))
        tableau[0, 0:self.num_vars] = -c
        tableau[1:, 0:A.shape[1]] = A
        tableau[1:, -1] = b 
        self.tableau = tableau

        # Ensure basic columns are correct
        for i, idx in enumerate(self.basic_index):
            if self.tableau[0, idx] != 0:
                self._pivot(i + 1, idx)

    def solve(self):
        if self.A_eq is None:
            return super().solve()
        
        # Phase 1: Solve the auxiliary problem
        self._construct_tableau()

        while not self._is_optimal():
            pivot_row, pivot_col = self._get_pivot()
            self._pivot(pivot_row, pivot_col)

        # Check if the problem is feasible
        if self.tableau[0, -1] != 0:
            raise ValueError("The problem is infeasible.")
        
        # Remove artificial variables and construct the tableau for phase 2
        self._construct_phase2_tableau()

        # Phase 2: Solve the original problem
        while not self._is_optimal():
            pivot_row, pivot_col = self._get_pivot()
            self._pivot(pivot_row, pivot_col)

        # Extract the optimal solution and value
        rhs = self.tableau[1:, -1]
        optimal_value = self.tableau[0, -1]
        optimal_solution = np.zeros(self.tableau.shape[1] - 1)  
        optimal_solution[self.basic_index] = rhs

        return optimal_solution, optimal_value


class DualSimplex(Simplex):
    """
    Dual Simplex Algorithm for solving linear programming problems.
    This class implements the dual simplex algorithm to find the optimal solution
    for a given linear programming problem defined by the tableau and basic variable indices.
    """
    def __init__(self, c, A_ub, b_ub, tableau=None, basic_index=None):
        super().__init__(c, A_ub, b_ub, tableau, basic_index)

    def _is_feasible(self):
        # Check if all basic variables are non-negative
        return np.all(self.tableau[1:, -1] >= 0)
    
    def _ratio_test(self, nume, deno):
        # Perform ratio test for dual simplex
        ratios = np.full_like(nume, np.inf, dtype=float)
        mask = deno < 0
        ratios[mask] = -nume[mask] / deno[mask]
        ratios[ratios < 0] = np.inf  # Ignore non-positive ratios
        return ratios

    def _get_pivot(self):
        # Select pivot row
        rhs = self.tableau[1:, -1]
        pivot_row = np.argmin(rhs) + 1
        if np.all(self.tableau[pivot_row, :-1] >= 0):
            raise ValueError("Problem is infeasible.")
        
        # Select pivot column
        nume = self.tableau[0, :-1]                 # C_j - Z_j
        deno = self.tableau[pivot_row, :-1]         # Leaving basic variable row
        ratios = self._ratio_test(nume, deno)
        if np.all(ratios == np.inf):
            raise ValueError("Problem is unbounded.")
        pivot_col = np.argmin(ratios)

        # Update basic variable index
        self.basic_index[pivot_row - 1] = pivot_col

        return pivot_row, pivot_col
    
    def solve(self):
        # Construct the initial tableau
        self._construct_tableau()

        # Perform the dual simplex algorithm
        while not self._is_feasible():
            pivot_row, pivot_col = self._get_pivot()
            self._pivot(pivot_row, pivot_col)

        # Check for optimality
        while not self._is_optimal():
            pivot_row, pivot_col = super()._get_pivot()
            self._pivot(pivot_row, pivot_col)
        
        # Extract the optimal solution and value
        rhs = self.tableau[1:, -1]
        optimal_value = self.tableau[0, -1]
        optimal_solution = np.zeros(self.tableau.shape[1] - 1)
        optimal_solution[self.basic_index] = rhs

        return optimal_solution, optimal_value


if __name__ == "__main__":
    # Set numpy print options for better readability
    np.set_printoptions(precision=3, suppress=True)

    # Example usage
    c = np.array([3, 2])
    A_ub = np.array([[1, 1], [2, 1], [1, 0]])
    b_ub = np.array([4, 6, 2])

    simplex = Simplex(c, A_ub, b_ub)
    solution, value = simplex.solve()
    print("Optimal solution:", solution)
    print("Optimal value:", value)
    print("Final tableau:\n", simplex.tableau)
    
    # Two-phase example
    A_eq = np.array([[1, 2]])
    b_eq = np.array([5])
    
    two_phase_simplex = TwoPhaseSimplex(c, A_ub, b_ub, A_eq, b_eq)
    solution, value = two_phase_simplex.solve()
    print("Two-phase optimal solution:", solution)
    print("Two-phase optimal value:", value)
    print("Two-phase final tableau:\n", two_phase_simplex.tableau)

    # Dual simplex example
    tableau = np.array([[1, 2, 0, 0, 0, 0],
                        [2, 1, 1, 0, 0, 4],
                        [1, 0, 0, 1, 0, 2],
                        [-3, -2, 0, 0, 1, -1]], dtype=float)
    basic_index = np.array([2, 3, 4])

    dual_simplex = DualSimplex(None, None, None, tableau=tableau, basic_index=basic_index)
    solution, value = dual_simplex.solve()
    print("Dual simplex optimal solution:", solution)
    print("Dual simplex optimal value:", value)
    print("Dual simplex final tableau:\n", dual_simplex.tableau)