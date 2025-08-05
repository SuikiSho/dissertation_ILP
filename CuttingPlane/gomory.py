import numpy as np
from .simplex import Simplex, TwoPhaseSimplex, DualSimplex

TOL = 1e-10 # Tolerance for floating-point comparisons

class GomoryCuttingPlane(Simplex):
    def __init__(self, c, A_ub, b_ub, A_eq=None, b_eq=None):
        """
        Initialize the Gomory Cutting Plane method.

        Parameters:
        c (np.ndarray): Coefficients of the objective function.
        A_ub (np.ndarray): Coefficients of the inequality constraints.
        b_ub (np.ndarray): Right-hand side values of the inequality constraints.
        A_eq (np.ndarray, optional): Coefficients of the equality constraints.
        b_eq (np.ndarray, optional): Right-hand side values of the equality constraints.
        """
        super().__init__(c=c, A_ub=A_ub, b_ub=b_ub)
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.tableau = None
        self.basic_index = None

    def _is_integer(self):
        # Check if all basic variables are integers
        rhs = self.tableau[1:, -1]
        idx = self.basic_index
        decs = rhs[idx < len(self.c)]
        return np.all(np.isclose(decs, np.round(decs), atol=TOL))

    def solve_lp_relaxation(self):
        # Solve the linear programming relaxation of the integer program.
        lp = TwoPhaseSimplex(c=self.c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq, b_eq=self.b_eq)
        lp.solve()
        self.tableau = lp.tableau
        self.basic_index = lp.basic_index

    def solve_with_dual_simplex(self):
        # Use the dual simplex method make tableau feasible.
        dual = DualSimplex(None, None, None, self.tableau, self.basic_index)
        dual.solve()
        self.tableau = dual.tableau
        self.basic_index = dual.basic_index

    def frac_part(self, x):
        # Calculate the fractional part of x
        return x - np.floor(x)

    def add_cutting_plane(self):
        # add a Gomory cutting plane to the tableau
        rhs = self.tableau[1:, -1]
        for i, b in enumerate(rhs):
            if not np.isclose(b, np.round(b), atol=TOL):
                cut_row_idx = i + 1  # +1 because the first row is the objective function
                break 
            
        # Extract the coefficients of the cutting plane
        row = self.tableau[cut_row_idx, :]
        cut_coeffs = np.zeros(len(row) - 1)
        
        # C-G cutting plane
        if np.all(np.isclose(row[:-1], np.round(row[:-1]), atol=TOL)):
            cut_coeffs[:len(self.c)] = row[:len(self.c)]
            cut_rhs = np.floor(row[-1])
        else:
            cut_coeffs = -self.frac_part(row[:-1])
            cut_rhs = -self.frac_part(row[-1])

        # Create a new tableau
        tableau = np.zeros((self.tableau.shape[0] + 1, self.tableau.shape[1] + 1))
        tableau[:-1, :-2] = self.tableau[:, :-1]
        tableau[:-1, -1] = self.tableau[:, -1]
        tableau[-1, :-2] = cut_coeffs
        tableau[-1, -1] = cut_rhs
        tableau[-1, -2] = 1

        # recover basic matrix
        basic_rec = Simplex(None, None, None, tableau=tableau, basic_index=self.basic_index)
        basic_rec._pivot(cut_row_idx, self.basic_index[cut_row_idx - 1])
        tableau = basic_rec.tableau

        self.tableau = tableau
        self.basic_index = np.append(self.basic_index, len(row) - 1)  # Add the new basic variable index

    def solve(self):
        # Step 1: Solve the LP relaxation
        self.solve_lp_relaxation()

        ### log
        # print("Initial tableau:\n", self.tableau)
        # print("Basic variables:", self.basic_index)

        # Step 2: Make solution integer
        while not self._is_integer():
            # Step 2a: Add a Gomory cutting plane
            self.add_cutting_plane()

            ### log
            # print("Tableau after adding cutting plane:\n", self.tableau)
            # print("Basic variables:", self.basic_index)

            # Step 2b: Make the tableau feasible using dual simplex
            self.solve_with_dual_simplex()

            ### log
            # print("Tableau after dual simplex:\n", self.tableau)
            # print("Basic variables:", self.basic_index)

        # Step 3: Extract the solution
        rhs = self.tableau[1:, -1]
        rhs = np.round(rhs)
        optimal_value = np.round(self.tableau[0, -1])
        optimal_solution = np.zeros(len(self.c))
        for i, idx in enumerate(self.basic_index):
            if idx < len(self.c):
                optimal_solution[idx] = rhs[i]

        return optimal_value, optimal_solution
    

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
    value, solution = gomory.solve()
    
    print("Optimal solution:", solution)
    print("Optimal value:", value)
    print("Final tableau:\n", gomory.tableau)


