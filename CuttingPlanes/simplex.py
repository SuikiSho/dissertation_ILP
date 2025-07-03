import numpy as np

class SimplexSolver:
    def __init__(self, c, A, b):
        self.m, self.n = A.shape
        self.A = np.hstack((A, np.eye(self.m)))        # 加入松弛变量
        self.c = np.concatenate([c, np.zeros(self.m)]) # 扩展目标函数
        self.b = b.astype(float)
        self.tableau = self._build_tableau()
        self.basic_vars = list(range(self.n, self.n + self.m))  # 初始基变量为松弛变量

    def _build_tableau(self):
        T = np.zeros((self.m + 1, self.n + self.m + 1))
        T[:-1, :-1] = self.A
        T[:-1, -1] = self.b
        T[-1, :-1] = -self.c  # 注意：目标是最大化，需要取负号
        return T

    def _pivot(self, row, col):
        self.tableau[row] /= self.tableau[row, col]
        for r in range(len(self.tableau)):
            if r != row:
                self.tableau[r] -= self.tableau[r, col] * self.tableau[row]
        self.basic_vars[row] = col

    def solve(self):
        while True:
            obj_row = self.tableau[-1, :-1]
            if np.all(obj_row >= 0):
                break  # 最优解找到
            col = np.argmin(obj_row)
            ratios = []
            for i in range(self.m):
                if self.tableau[i, col] > 0:
                    ratios.append(self.tableau[i, -1] / self.tableau[i, col])
                else:
                    ratios.append(np.inf)
            row = np.argmin(ratios)
            if ratios[row] == np.inf:
                raise Exception("Unbounded solution")
            self._pivot(row, col)

        return self._get_solution()

    def _get_solution(self):
        x = np.zeros(self.n + self.m)
        for i, var in enumerate(self.basic_vars):
            x[var] = self.tableau[i, -1]
        obj = self.tableau[-1, -1]
        return x[:self.n], obj, self.tableau, self.basic_vars

def _test1():
    # 示例问题：
    # max 3x1 + 2x2
    # s.t. x1 + x2 <= 4
    #      x1 <= 2
    #      x2 <= 3
    #      x1, x2 >= 0

    c = np.array([3, 2])
    A = np.array([
        [1, 1],
        [1, 0],
        [0, 1]
    ])
    b = np.array([4, 2, 3])

    solver = SimplexSolver(c, A, b)
    solution, value, tableau, basic_vars = solver.solve()

    print("Optimal solution:", solution)
    print("Optimal value:", value)
    print("Tableau:\n", tableau)
    print("Basic variables:", basic_vars)



if __name__ == "__main__":
    _test1()
