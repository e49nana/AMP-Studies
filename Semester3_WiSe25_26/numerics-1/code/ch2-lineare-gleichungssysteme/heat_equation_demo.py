"""
heat_equation_demo.py
=====================

Application standalone : équation de la chaleur 2D (Beispiel 2.24).

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", Beispiel 2.24.

Résout -ΔT = 0 sur [0,1]² avec conditions de Dirichlet par schéma à
5 points et compare : Gauss direct, Jacobi, Gauss-Seidel, SciPy sparse.

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


def assemblage_laplacien_2d(
    n: int,
    bord_haut: float = 100.0,
    bord_bas: float = 0.0,
    bord_gauche: float = 0.0,
    bord_droit: float = 0.0,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble le Laplacien discret 5-points sur grille n×n."""
    N = n * n
    main = 4.0 * np.ones(N)
    off1 = -np.ones(N - 1)
    off1[np.arange(1, N) % n == 0] = 0.0
    offn = -np.ones(N - n)

    A = sparse.diags([main, off1, off1, offn, offn],
                     [0, -1, 1, -n, n], shape=(N, N), format="csr")

    b = np.zeros(N)
    for j in range(n):
        b[j] += bord_bas
        b[(n - 1) * n + j] += bord_haut
    for i in range(n):
        b[i * n] += bord_gauche
        b[i * n + n - 1] += bord_droit
    return A, b


def jacobi_sparse(A, b, tol=1e-6, n_max=50_000):
    """Jacobi pour matrice sparse."""
    n = len(b)
    diag = A.diagonal()
    x = np.zeros(n)
    for k in range(1, n_max + 1):
        x_new = (b - A @ x + diag * x) / diag
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, k
        x = x_new
    return x, k


def gauss_seidel_sparse(A, b, tol=1e-6, n_max=50_000):
    """Gauss-Seidel pour matrice sparse (conversion dense pour simplicité)."""
    A_dense = A.toarray()
    n = len(b)
    x = np.zeros(n)
    for k in range(1, n_max + 1):
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - A_dense[i, :i] @ x[:i] - A_dense[i, i+1:] @ x[i+1:]) / A_dense[i, i]
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x, k
    return x, k


def tracer_solution(x, n, titre="", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(x.reshape(n, n), origin="lower", cmap="hot", extent=[0, 1, 0, 1])
    ax.set_title(titre)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="T")
    return ax


if __name__ == "__main__":
    for n in [10, 30, 50]:
        A, b = assemblage_laplacien_2d(n, bord_haut=100)
        N = n * n
        print(f"\n=== Grille {n}×{n} ({N} inconnues) ===")

        t0 = time.time()
        x_sp = spsolve(A, b)
        t_sp = time.time() - t0

        t0 = time.time()
        x_j, k_j = jacobi_sparse(A, b)
        t_j = time.time() - t0

        if n <= 30:
            t0 = time.time()
            x_gs, k_gs = gauss_seidel_sparse(A, b)
            t_gs = time.time() - t0
            print(f"  Gauss-Seidel : {k_gs:>6} it., {t_gs:.3f}s")
        else:
            print(f"  Gauss-Seidel : skipped (too slow for n={n})")

        print(f"  Jacobi       : {k_j:>6} it., {t_j:.3f}s")
        print(f"  SciPy sparse : direct,    {t_sp:.4f}s")

    # Tracé final
    n = 50
    A, b = assemblage_laplacien_2d(n, bord_haut=100)
    x = spsolve(A, b)
    fig, ax = plt.subplots(figsize=(6, 5))
    tracer_solution(x, n, "Beispiel 2.24 — Température ($n=50$)", ax)
    plt.tight_layout()
    plt.savefig("heat_equation_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
