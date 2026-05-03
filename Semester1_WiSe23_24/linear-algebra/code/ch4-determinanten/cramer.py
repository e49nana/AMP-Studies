"""
cramer.py
=========

Règle de Cramer et inverse par cofacteurs.

Couvre :
    - Règle de Cramer : xᵢ = det(Aᵢ) / det(A)
    - Matrice des cofacteurs (Kofaktormatrix)
    - Inverse par la formule A⁻¹ = (1/det A) · cof(A)ᵀ
    - Pourquoi Cramer est théoriquement élégant mais pratiquement catastrophique
    - Comparaison de coût : Cramer O(n·n!) vs Gauss O(n³)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Cofacteurs et matrice adjointe
# ======================================================================

def mineur(A: np.ndarray, i: int, j: int) -> float:
    """Mineur M_ij = det de A sans la ligne i et la colonne j."""
    sous = np.delete(np.delete(A, i, axis=0), j, axis=1)
    return float(np.linalg.det(sous))


def cofacteur(A: np.ndarray, i: int, j: int) -> float:
    """Cofacteur C_ij = (-1)^{i+j} · M_ij."""
    return (-1)**(i + j) * mineur(A, i, j)


def matrice_cofacteurs(A: np.ndarray) -> np.ndarray:
    """Matrice des cofacteurs (Kofaktormatrix)."""
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = cofacteur(A, i, j)
    return C


def adjointe(A: np.ndarray) -> np.ndarray:
    """Matrice adjointe (Adjunkte) = transposée de la matrice des cofacteurs."""
    return matrice_cofacteurs(A).T


def inverse_cofacteurs(A: np.ndarray) -> np.ndarray:
    """
    A⁻¹ = (1/det A) · adj(A).

    Formule élégante mais coûteuse : O(n²·n!) cofacteurs à calculer.
    """
    d = np.linalg.det(A)
    if abs(d) < 1e-12:
        raise np.linalg.LinAlgError("det = 0, matrice singulière.")
    return adjointe(A) / d


# ======================================================================
#  2. Règle de Cramer
# ======================================================================

def cramer(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Règle de Cramer :
        xᵢ = det(Aᵢ) / det(A)
    où Aᵢ = A avec la colonne i remplacée par b.

    Coût : (n+1) déterminants → O(n·n!) avec Leibniz, O(n⁴) avec Gauss.
    En comparaison, Gauss direct coûte O(n³).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    det_A = np.linalg.det(A)

    if abs(det_A) < 1e-12:
        raise np.linalg.LinAlgError("det(A) = 0.")

    x = np.zeros(n)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / det_A

    return x


def cramer_2x2(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cramer explicite pour 2×2 :
        x₁ = (b₁a₂₂ - b₂a₁₂) / det,  x₂ = (a₁₁b₂ - a₂₁b₁) / det.
    """
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    x1 = (b[0]*A[1,1] - b[1]*A[0,1]) / det
    x2 = (A[0,0]*b[1] - A[1,0]*b[0]) / det
    return np.array([x1, x2])


# ======================================================================
#  3. Comparaison de performance
# ======================================================================

def benchmark_cramer_vs_gauss(
    tailles: tuple[int, ...] = (3, 5, 8, 10),
) -> tuple[list[float], list[float]]:
    """Compare les temps de Cramer vs np.linalg.solve."""
    rng = np.random.default_rng(42)
    t_cramer, t_gauss = [], []

    for n in tailles:
        A = rng.standard_normal((n, n))
        b = rng.standard_normal(n)

        t0 = time.perf_counter()
        for _ in range(10):
            cramer(A, b)
        t_cramer.append((time.perf_counter() - t0) / 10)

        t0 = time.perf_counter()
        for _ in range(1000):
            np.linalg.solve(A, b)
        t_gauss.append((time.perf_counter() - t0) / 1000)

    return t_cramer, t_gauss


def tracer_benchmark(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    tailles = (3, 5, 8, 10)
    t_cr, t_ga = benchmark_cramer_vs_gauss(tailles)

    ax.semilogy(tailles, t_cr, "rs-", label="Cramer", markersize=6)
    ax.semilogy(tailles, t_ga, "bo-", label="Gauss (numpy)", markersize=6)
    ax.set_xlabel("taille $n$")
    ax.set_ylabel("temps (s)")
    ax.set_title("Cramer $O(n^4)$ vs Gauss $O(n^3)$ — inutilisable pour $n > 10$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Règle de Cramer 3×3 ===")
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    x_cr = cramer(A, b)
    x_np = np.linalg.solve(A, b)
    print(f"  Cramer : x = {x_cr}")
    print(f"  NumPy  : x = {x_np}")
    print(f"  ||diff|| = {np.linalg.norm(x_cr - x_np):.2e}")

    print(f"\n=== Cramer 2×2 (formule explicite) ===")
    A2 = np.array([[3, 1], [2, 4]], dtype=float)
    b2 = np.array([9, 8], dtype=float)
    x2 = cramer_2x2(A2, b2)
    print(f"  x = {x2}")
    print(f"  det(A) = {np.linalg.det(A2):.0f}")

    print(f"\n=== Inverse par cofacteurs ===")
    A_inv = inverse_cofacteurs(A)
    print(f"  ||mine - numpy||_∞ = {np.linalg.norm(A_inv - np.linalg.inv(A), np.inf):.2e}")

    print(f"\n=== Matrice des cofacteurs ===")
    C = matrice_cofacteurs(A)
    print(f"  cof(A) =\n{np.round(C, 4)}")
    print(f"  adj(A) = cof(A)ᵀ =\n{np.round(adjointe(A), 4)}")

    print(f"\n=== Pourquoi Cramer est mauvais en pratique ===")
    from math import factorial
    for n in [3, 5, 10, 20, 50]:
        print(f"  n={n:>2} : Cramer ~{n+1} dets × O(n³) = O(n⁴), "
              f"Gauss O(n³) = {n**3//3} ops")

    tracer_benchmark()
    plt.tight_layout()
    plt.savefig("cramer_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
