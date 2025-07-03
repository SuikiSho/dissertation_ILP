import numpy as np
from simplex import SimplexSolver
from scipy.optimize import linprog


def test_case(name, c, A, b):
    print(f"\n===== {name} =====")

    # --- 自定义 SimplexSolver ---
    print("→ 自己实现的 SimplexSolver:")
    try:
        solver = SimplexSolver(np.array(c), np.array(A), np.array(b))
        x_sol, obj_val, tableau, basic_vars = solver.solve()

        print("  最优解: x =", x_sol)
        print("  最优值: z =", obj_val)

        print("  最终 Simplex 表格:")
        np.set_printoptions(precision=3, suppress=True)
        print(tableau)
        print("  基变量索引:", basic_vars)

    except Exception as e:
        print("  出错:", e)

    # --- SciPy 的 linprog ---
    print("\n→ SciPy linprog:")
    try:
        # 注意：scipy 是最小化，因此目标函数要取负
        res = linprog(
            c=-np.array(c),
            A_ub=np.array(A),
            b_ub=np.array(b),
            method='simplex',  # 使用内建的单纯形法
        )

        if res.success:
            print("  最优解: x =", res.x)
            print("  最优值: z =", -res.fun)
        else:
            print("  求解失败:", res.message)
    except Exception as e:
        print("  出错:", e)


def main():
    # 测试 1：标准例题
    c1 = [3, 2]
    A1 = [[1, 1], [1, 0], [0, 1]]
    b1 = [4, 2, 3]
    test_case("测试 1：标准最大化例题", c1, A1, b1)

    # 测试 2：更复杂的 LP
    c2 = [5, 4]
    A2 = [[-1, -1], [2, -1], [1, 2]]
    b2 = [-2, 4, 8]
    test_case("测试 2：典型 IP 割平面例题的 LP 松弛", c2, A2, b2)

    # 测试 3：多变量问题
    c3 = [4, 3, 5, 6]
    A3 = [
        [1, 2, 1, 0],
        [2, 0, 3, 1],
        [0, 3, 2, 2],
        [1, 1, 0, 1]
    ]
    b3 = [10, 15, 20, 8]
    test_case("测试 3：多变量问题", c3, A3, b3)

    # 测试 4：稀疏约束问题
    c4 = [2, 1, 3, 4, 5]
    A4 = [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 1, 1, 0, 0]
    ]
    b4 = [5, 6, 7, 8]
    test_case("测试 4：稀疏约束问题", c4, A4, b4)

    # 测试 5：大规模问题
    c5 = [1, 2, 3, 4, 5, 6, 7, 8]
    A5 = [
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ]
    b5 = [10, 12, 15, 20, 25]
    test_case("测试 5：大规模问题", c5, A5, b5)


if __name__ == "__main__":
    main()
