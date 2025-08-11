import numpy as np
from scipy.optimize import linprog

_epsilon = 1e-8

class BnBTreeNode:
    """

    """
    def __init__(self, parent=None, x=None, fun=None):
        self.parent = parent
        self.x = x
        self.fun = fun

        self.left = None
        self.right = None
        self.pruned = False

    def __repr__(self):
        return f"BnBTreeNode(x={self.x}, fun={self.fun}, pruned={self.pruned})"
    
    def print_tree(self, level=0):
        indent = "  " * level
        print(f"{indent}Node: {self}")
        if self.left:
            print(f"{indent}Left:")
            self.left.print_tree(level + 1)
        if self.right:
            print(f"{indent}Right:")
            self.right.print_tree(level + 1)
        if self.pruned:
            print(f"{indent}Pruned")


class BranchAndBound:
    """
    
    """
    def __init__(self, c, A_ub, b_ub, A_eq=None, b_eq=None):
        self.c = c              # default: maximize c^T x
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq

        self.BnBTree = None
        self.best_val = float('-inf')
        self.best_sol = None

    @staticmethod
    def _is_int(x):
        """
        Check if all elements in x are integers.
        """
        return np.all(np.isclose(x, np.round(x), rtol=0, atol=_epsilon))
    
    def _solve_rlp(self, A_ub, b_ub):
        """
        Solve the linear relaxation of the integer programming problem.
        """
        res = linprog(-self.c, A_ub=A_ub, b_ub=b_ub, A_eq=self.A_eq, b_eq=self.b_eq, bounds=(0, None), method='highs')
        return res
    
    def _append_ub_row(self, A_ub, b_ub, idx, coeff, rhs):
        # 
        row = np.zeros((1, len(self.c)))
        row[0, idx] = coeff
        if A_ub is None:
            A_ub_new = row
            b_ub_new = np.array([rhs], dtype=float)
        else:
            A_ub_new = np.vstack([A_ub, row])
            b_ub_new = np.append(b_ub, rhs)
        return A_ub_new, b_ub_new
    
    def solve(self):
        """
        Solve the integer programming problem using branch and bound.
        """
        res = self._solve_rlp(self.A_ub, self.b_ub)
        if res.status == 2:     # Infeasible
            raise ValueError("The problem is infeasible.")
        elif res.status == 3:   # Unbounded
            raise ValueError("The problem is unbounded.")
        
        self.BnBTree = BnBTreeNode(parent=None, x=res.x, fun=-res.fun)        
        stack = [(self.BnBTree, self.A_ub, self.b_ub)]

        while stack:
            node, A_ub, b_ub = stack.pop()

            if node.fun <= self.best_val:
                node.pruned = True
                continue

            if self._is_int(node.x):
                self.best_sol = np.round(node.x).astype(float)
                self.best_val = float(self.c @ self.best_sol)
            else:
                # Branching: create two new nodes
                idx = [i for i, v in enumerate(node.x) if not self._is_int(v)][0]   # Find the first non-integer variable
                
                A_ub_left, b_ub_left = self._append_ub_row(A_ub, b_ub, idx, +1.0, np.floor(node.x[idx]))
                res_left = self._solve_rlp(A_ub_left, b_ub_left)
                if res_left.success:
                    node.left = BnBTreeNode(parent=node, x=res_left.x, fun=-res_left.fun)
                    stack.append((node.left, A_ub_left, b_ub_left))

                A_ub_right, b_ub_right = self._append_ub_row(A_ub, b_ub, idx, -1.0, -np.ceil(node.x[idx]))
                res_right = self._solve_rlp(A_ub_right, b_ub_right)
                if res_right.success:
                    node.right = BnBTreeNode(parent=node, x=res_right.x, fun=-res_right.fun)
                    stack.append((node.right, A_ub_right, b_ub_right))

        # Check if a solution was found
        if self.best_sol is None:
            raise ValueError("No feasible solution found.")

        return self.best_sol, self.best_val   

    def print_tree(self):
        """
        Print the branch and bound tree.
        """
        if self.BnBTree is not None:
            self.BnBTree.print_tree()
        else:
            print("The tree is empty.")


if __name__ == "__main__":
    # Example usage
    c = np.array([3, 2])
    A_ub = np.array([[1, 1], [2, 1]])
    b_ub = np.array([4.5, 6.5])
    A_eq = None
    b_eq = None

    bnb = BranchAndBound(c, A_ub, b_ub, A_eq, b_eq)
    solution, value = bnb.solve()
    bnb.print_tree()
    print("Best solution:", solution)
    print("Best value:", value)