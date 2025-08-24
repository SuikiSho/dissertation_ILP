# gomory.py
import numpy as np

class SimplexTableau:
    """
    单纯形表 + 对偶单纯形（用于割后修复可行性）
    约定：
      - 最后一行为目标函数行（objective row）
      - 最后一列为 RHS（右端常数）
      - 初始构造：Max 问题，目标行放置 -c（即“最负进入”规则）
      - 变量列顺序：前 n 列为原始变量 x，后续为松弛/割用 slack
    """
    def __init__(self, A, b, c, tol=1e-9, use_bland=True):
        """
        A: (m,n), b: (m,), c: (n,)
        需要 A x <= b, b >= 0
        """
        self.tol = tol
        self.use_bland = use_bland

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        c = np.array(c, dtype=float)

        # 预处理：若存在 b_i < 0，将该行乘以 -1 转为 b_i >= 0
        for i in range(len(b)):
            if b[i] < -self.tol:
                A[i, :] *= -1.0
                b[i] *= -1.0

        m, n = A.shape
        # 初始松弛变量 I_m
        I = np.eye(m)
        # tableau 大小：(m+1) x (n + m + 1)
        T = np.zeros((m + 1, n + m + 1))
        # 约束行
        T[:m, :n] = A
        T[:m, n:n + m] = I
        T[:m, -1] = b
        # 目标行（Max）：放置 -c
        T[m, :n] = -c
        T[m, -1] = 0.0

        # 初始基变量：为 m 个松弛变量
        basis = list(range(n, n + m))

        self.T = T
        self.n = n                 # 原始变量个数
        self.m = m                 # 约束行数（不含目标行）
        self.basis = basis         # 基列索引（长度 = 当前行数 self.m）

    # -------- 工具函数 --------
    def _pivot(self, row, col):
        """在 (row, col) 处执行枢轴变换"""
        T = self.T
        piv = T[row, col]
        if abs(piv) < self.tol:
            raise RuntimeError("Pivot too small.")

        # 归一化枢轴行
        T[row, :] = T[row, :] / piv
        # 消去其他行该列
        for r in range(T.shape[0]):
            if r == row: 
                continue
            if abs(T[r, col]) > self.tol:
                factor = T[r, col]
                T[r, :] -= factor * T[row, :]

        self.basis[row] = col

    def _choose_entering_primal(self):
        """
        Primal simplex：选择进入列（最小的负值；若用 Bland 则选最小索引）
        返回 col 或 None（表示已经最优）
        """
        obj = self.T[-1, :-1]
        # 目标行为 -c 与约化成本，负数代表有改进空间
        neg_positions = np.where(obj < -self.tol)[0]
        if neg_positions.size == 0:
            return None
        # Bland：选最小索引；否则选最负
        if self.use_bland:
            return int(neg_positions[0])
        else:
            return int(np.argmin(obj))  # 最负的

    def _choose_leaving_primal(self, col):
        """
        Primal ratio test：a_ij > 0 时，min b_i / a_ij
        返回 row 或 None（无界）
        """
        T = self.T
        rhs = T[:-1, -1]
        col_vals = T[:-1, col]
        mask = col_vals > self.tol
        if not np.any(mask):
            return None  # 无界
        ratios = np.full_like(rhs, np.inf, dtype=float)
        ratios[mask] = rhs[mask] / col_vals[mask]
        # Bland tie-break：最小比值中选择列索引更小的行
        row = int(np.argmin(ratios))
        return row

    def primal_simplex(self, max_iters=10_000):
        """运行 primal simplex 直到最优或无界"""
        it = 0
        while it < max_iters:
            enter = self._choose_entering_primal()
            if enter is None:
                return "optimal"  # 目标行无负项
            leave = self._choose_leaving_primal(enter)
            if leave is None:
                return "unbounded"
            self._pivot(leave, enter)
            it += 1
        return "iteration_limit"

    def _choose_leaving_dual(self):
        """
        Dual simplex：选择离开行（RHS 最负的行；Bland 则取最小索引）
        """
        rhs = self.T[:-1, -1]
        neg_rows = np.where(rhs < -self.tol)[0]
        if neg_rows.size == 0:
            return None
        # 选最负（更典型），或 Bland 取第一个
        if self.use_bland:
            return int(neg_rows[0])
        else:
            # 最负
            return int(neg_rows[np.argmin(rhs[neg_rows])])

    def _choose_entering_dual(self, row):
        """
        Dual ratio test：在 row 行中找 a_rj < 0 的列，
        令 t_j = obj_j / (-a_rj)，取 t_j 最小的列进入
        若没有 a_rj < 0，问题不可行
        """
        T = self.T
        a = T[row, :-1]
        obj = T[-1, :-1]
        mask = a < -self.tol
        if not np.any(mask):
            return None
        ratio = np.full_like(obj, np.inf, dtype=float)
        ratio[mask] = obj[mask] / (-a[mask])
        # 取最小 t_j；Bland 打破平局：索引更小
        col = int(np.argmin(ratio))
        if np.isinf(ratio[col]):
            return None
        return col

    def dual_simplex(self, max_iters=10_000):
        """
        对偶单纯形：用于在目标行已满足最优性（>=0）但 RHS 有负值时，
        修复可行性（使所有 RHS >= 0）
        """
        it = 0
        while it < max_iters:
            row = self._choose_leaving_dual()
            if row is None:
                return "feasible"  # RHS 全部非负
            col = self._choose_entering_dual(row)
            if col is None:
                return "infeasible"
            self._pivot(row, col)
            it += 1
        return "iteration_limit"

    def add_cut_row_ge(self, coeffs, rhs):
        """
        添加一个形如   coeffs * x >= rhs   的割。
        转换为：(-coeffs) * x + s_new = -rhs，并将新行插到目标行之上。
        """
        coeffs = np.array(coeffs, dtype=float)

        old_rows, old_total_cols = self.T.shape
        old_var_cols = old_total_cols - 1            # 旧的“变量列数”（不含 RHS）
        new_var_cols = old_var_cols + 1              # +1 个新 slack
        new_total_cols = new_var_cols + 1            # 再 +1 个 RHS
        new_rows = old_rows + 1                      # +1 行（新割）

        new_T = np.zeros((new_rows, new_total_cols))

        # 1) 先拷贝“约束行”（不含目标行）到新表的顶部
        #    拷贝变量列（不含 RHS）
        new_T[:old_rows-1, :old_var_cols] = self.T[:old_rows-1, :old_var_cols]
        #    拷贝 RHS
        new_T[:old_rows-1, -1] = self.T[:old_rows-1, -1]

        # 2) 把旧目标行拷到新表的最后一行（保持目标行在最后）
        new_T[-1, :old_var_cols] = self.T[-1, :old_var_cols]
        new_T[-1, -1] = self.T[-1, -1]
        # 目标行在新 slack 列系数为 0（new_T 已初始化为 0，无需额外设置）

        # 3) 写入“新割行”（位于倒数第二行）
        #    (-coeffs) 放到原有变量列；新 slack 列系数为 1；RHS = -rhs
        new_T[-2, :len(coeffs)] = -coeffs
        new_T[-2, old_var_cols] = 1.0    # 新 slack 列位置索引 = old_var_cols
        new_T[-2, -1] = -float(rhs)

        # 4) 更新内部状态
        self.T = new_T
        self.m += 1
        self.basis.append(old_var_cols)  # 新增 slack 作为基变量

    def extract_solution(self):
        """
        返回 (x, obj)
        x：只取原始变量前 n 列的值（非基为 0，基为对应 RHS）
        obj：目标函数值（单纯形表 RHS 的值）
        """
        x = np.zeros(self.n)
        for i, col in enumerate(self.basis):
            if col < self.n:  # 基变量是原始变量
                x[col] = self.T[i, -1]
        obj = self.T[-1, -1]
        return x, obj


def frac_part(arr, tol=1e-10):
    """
    计算分数部分：x - floor(x)
    并把接近 0 或 1 的小数统一裁剪到 0
    """
    f = arr - np.floor(arr)
    f[np.isclose(f, 0.0, atol=tol)] = 0.0
    f[np.isclose(f, 1.0, atol=tol)] = 0.0
    return f


class GomoryCuttingPlane:
    """
    纯整数 Gomory Cutting Plane（分数割），基于单纯形表。
    """
    def __init__(self, c, A_ub, b_ub, integer_indices=None, tol=1e-9, max_cuts=1000, verbose=False, use_bland=True):
        """
        c, A_ub, b_ub 定义 Max c^T x, s.t. A_ub x <= b_ub, x>=0
        integer_indices: 可选，指定哪些变量必须为整数（默认全部）
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A_ub, dtype=float)
        self.b = np.array(b_ub, dtype=float)
        self.tol = tol
        self.max_cuts = max_cuts
        self.verbose = verbose
        self.use_bland = use_bland

        n = self.c.size
        if integer_indices is None:
            self.integer_mask = np.ones(n, dtype=bool)
        else:
            mask = np.zeros(n, dtype=bool)
            mask[np.array(integer_indices, dtype=int)] = True
            self.integer_mask = mask

        # 构造初始单纯形表
        self.tab = SimplexTableau(self.A, self.b, self.c, tol=self.tol, use_bland=self.use_bland)

    def _print_tableau(self, title=""):
        if not self.verbose:
            return
        print("\n====", title, "====")
        print("Basis:", self.tab.basis)
        print(self.tab.T)

    def _is_integer_vector(self, x):
        if self.integer_mask.any():
            xi = x[self.integer_mask]
        else:
            xi = np.array([])
        if xi.size == 0:
            return True
        return np.all(np.isclose(xi, np.round(xi), atol=1e-7))

    def _choose_fractional_row(self):
        """
        在当前单纯形表中选择一个 RHS 含小数部分的行（用于生成分数割）。
        优先选择小数部分最大的行；若 tie，选索引更小（Bland）。
        返回 (row_index, frac_rhs) 或 (None, None) 若不存在
        """
        rhs = self.tab.T[:-1, -1]
        frac_rhs = frac_part(rhs, tol=1e-12)
        # 只选明显有分数部分的（> tol）
        idxs = np.where(frac_rhs > self.tol)[0]
        if idxs.size == 0:
            return None, None
        # 选小数部分最大的行
        j = int(idxs[np.argmax(frac_rhs[idxs])])
        return j, float(frac_rhs[j])

    def solve(self, max_total_iters=10000):
        """
        主求解流程：
          1) primal simplex -> 最优 LP 松弛解
          2) 若整数性满足 -> 结束
          3) 选分数 RHS 行，构造割： sum frac(row) * vars >= frac(b)
             -> 添加为 (-frac(row))*vars + s_new = -frac(b)
          4) 对偶单纯形修复可行性
          5) 回到 1)
        """
        # 先求 LP 松弛最优
        status = self.tab.primal_simplex(max_iters=max_total_iters)
        if status not in ("optimal",):
            return {"status": status, "x": None, "obj": None, "cuts": 0}

        cuts_added = 0
        self._print_tableau("Initial optimal LP relaxation")

        while cuts_added < self.max_cuts:
            x, obj = self.tab.extract_solution()
            if self._is_integer_vector(x):
                return {"status": "integer_optimal", "x": x, "obj": obj, "cuts": cuts_added}

            # 选择分数 RHS 的行
            row_idx, fb = self._choose_fractional_row()
            if row_idx is None:
                # 没有分数 RHS（理论上应该已整数）
                return {"status": "no_fractional_row", "x": x, "obj": obj, "cuts": cuts_added}

            # 从该行生成割：coeffs 为该行各列系数（不含 RHS），取分数部分
            row_coeffs = self.tab.T[row_idx, :-1].copy()
            # 该行的基变量列理论上为 1，其小数部分为 0；直接整体取 frac 即可
            frac_coeffs = frac_part(row_coeffs, tol=1e-12)

            # 添加为 GE 形式：sum frac_coeffs * vars >= fb
            self.tab.add_cut_row_ge(frac_coeffs, fb)
            cuts_added += 1
            self._print_tableau(f"After adding cut #{cuts_added} (raw)")

            # 用对偶单纯形修复可行性
            status = self.tab.dual_simplex(max_iters=max_total_iters)
            if status not in ("feasible",):
                return {"status": f"dual_{status}", "x": None, "obj": None, "cuts": cuts_added}
            self._print_tableau(f"After dual simplex for cut #{cuts_added}")

            # 再用 primal simplex 达成最优（通常目标行已满足最优性，此步可省；
            # 但为稳健，这里再次调用，确保处于最优 LP）
            status = self.tab.primal_simplex(max_iters=max_total_iters)
            if status not in ("optimal",):
                return {"status": status, "x": None, "obj": None, "cuts": cuts_added}
            self._print_tableau(f"After primal simplex post cut #{cuts_added}")

        # 达到割上限
        x, obj = self.tab.extract_solution()
        return {"status": "cut_limit", "x": x, "obj": obj, "cuts": cuts_added}


# ===========================
# 示例与自测
# ===========================
if __name__ == "__main__":
    # 一个简单可测的纯整数问题：
    # Max z = 3x1 + 2x2
    # s.t.  2x1 +  x2 <= 4.5
    #        x1 + 2x2 <= 4
    #        x1, x2 >= 0, 且为整数
    c = np.array([3., 3., 5., 9., 3.])
    A_ub = np.array([
        [6., 6., 4., 1., 2.],
        [4., 1., 9., 2., 9.],
        [8., 4., 4., 9., 7.]
    ])
    b_ub = np.array([9.5, 12.5, 16.])

    solver = GomoryCuttingPlane(c, A_ub, b_ub, tol=1e-9, verbose=True, use_bland=True)
    res = solver.solve()
    print("\n=== RESULT ===")
    print(res)
