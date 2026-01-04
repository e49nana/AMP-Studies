"""Quick demo for iterative_methods.py

Run:
    python demo_iterative_methods.py
"""

import numpy as np

from iterative_methods import jacobi, jacobi_vec, gauss_seidel, sor


def main():
    # Example system (diagonally dominant)
    A = np.array([[10.0, -1.0,  2.0,  0.0],
                  [-1.0, 11.0, -1.0,  3.0],
                  [ 2.0, -1.0, 10.0, -1.0],
                  [ 0.0,  3.0, -1.0,  8.0]])
    b = np.array([6.0, 25.0, -11.0, 15.0])
    x0 = np.zeros_like(b)

    x_j = jacobi(A, b, x0, tol=1e-10, max_iter=10_000)
    x_jv = jacobi_vec(A, b, x0, tol=1e-10, max_iter=10_000)
    x_gs = gauss_seidel(A, b, x0, tol=1e-10, max_iter=10_000)
    x_sor = sor(A, b, x0, omega=1.2, tol=1e-10, max_iter=10_000)

    x_true = np.linalg.solve(A, b)

    print("Jacobi      :", x_j)
    print("Jacobi (vec):", x_jv)
    print("Gauss-Seidel:", x_gs)
    print("SOR (w=1.2) :", x_sor)
    print("Direct solve:", x_true)

    print("\nErrors (2-norm):")
    print("Jacobi      :", np.linalg.norm(x_j - x_true))
    print("Jacobi (vec):", np.linalg.norm(x_jv - x_true))
    print("Gauss-Seidel:", np.linalg.norm(x_gs - x_true))
    print("SOR (w=1.2) :", np.linalg.norm(x_sor - x_true))


if __name__ == "__main__":
    main()
