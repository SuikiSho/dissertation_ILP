import numpy as np

def show_case(npz_file: str, key_substr: str):
    """
    Search and print cases in npz_file whose key contains key_substr.
    Example: show_case("PerformanceTest/cases/p_n2_10.npz", "seed07")
    """
    data = np.load(npz_file, allow_pickle=True)
    matches = [k for k in data.keys() if key_substr in k]

    if not matches:
        print(f"[INFO] No key matches '{key_substr}' in {npz_file}")
        return

    for k in matches:
        c, A_ub, b_ub = data[k]
        print("="*60)
        print("Key:", k)
        print("c:", c)
        print("A_ub:\n", A_ub)
        print("b_ub:", b_ub)
    print("="*60)
    print(f"[INFO] Found {len(matches)} matches for '{key_substr}'.")

if __name__ == "__main__":
    show_case("cases/c_n2_10.npz", "seed01")
