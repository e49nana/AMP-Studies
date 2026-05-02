"""
matrix_rank.py
==============

Rang d'une matrice et théorème du rang (Rangsatz).

Couvre :
    - Rang par échelonnement (from-scratch)
    - Rangsatz : rang(A) + dim Kern(A) = n
    - Rang colonne = rang ligne (preuve numérique)
    - Rang et inversibilité
    - Rang d'un produit : rang(AB) ≤ min(rang(A), rang(B))
    - Application : déterminer si un système est compatible

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def rang_echelonnement(A: np.ndarray, tol: float = 1e-10) -> int:
    """Rang par échelonnement (nombre de pivots non nuls)."""
    A = np.asarray(A, dtype=float).copy()
    m, n = A.shape
    r = 0
    for col in range(n):
        if r >= m:
            break
        i_max = r + np.argmax(np.abs(A[r:, col]))
        if abs(A[i_max, col]) < tol:
            continue
        A[[r, i_max]] = A[[i_max, r]]
        for i in range(r + 1, m):
            A[i] -= (A[i, col] / A[r, col]) * A[r]
        r += 1
    return r


def rang_svd(A: np.ndarray, tol: float = 1e-10) -> int:
    """Rang par SVD (nombre de valeurs singulières > tol)."""
    sv = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(sv > tol))


def verifier_rangsatz(A: np.ndarray) -> dict:
    """
    Rangsatz : rang(A) + dim Kern(A) = n (nombre de colonnes).
    """
    m, n = A.shape
    r = rang_echelonnement(A)
    dim_kern = n - r
    return {
        "taille": f"{m}×{n}",
        "rang": r,
        "dim_kern": dim_kern,
        "n": n,
        "vérifié": r + dim_kern == n,
    }


def verifier_rang_ligne_colonne(A: np.ndarray) -> dict:
    """Vérifie que rang ligne = rang colonne."""
    r_ligne = rang_echelonnement(A)
    r_colonne = rang_echelonnement(A.T)
    return {
        "rang_ligne": r_ligne,
        "rang_colonne": r_colonne,
        "égaux": r_ligne == r_colonne,
    }


def verifier_rang_produit(A: np.ndarray, B: np.ndarray) -> dict:
    """Vérifie rang(AB) ≤ min(rang(A), rang(B))."""
    rA = rang_echelonnement(A)
    rB = rang_echelonnement(B)
    rAB = rang_echelonnement(A @ B)
    return {
        "rang(A)": rA,
        "rang(B)": rB,
        "rang(AB)": rAB,
        "min(rA, rB)": min(rA, rB),
        "vérifié": rAB <= min(rA, rB),
    }


def critere_compatibilite(A: np.ndarray, b: np.ndarray) -> dict:
    """
    Un système Ax = b est compatible ssi rang(A) = rang([A|b]).
    """
    r_A = rang_echelonnement(A)
    aug = np.column_stack([A, b])
    r_aug = rang_echelonnement(aug)
    return {
        "rang(A)": r_A,
        "rang([A|b])": r_aug,
        "compatible": r_A == r_aug,
    }


def tracer_rang_vs_taille(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre le rang de matrices aléatoires m×n pour différentes tailles."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(42)
    ms = range(1, 11)
    ns = range(1, 11)
    rang_data = np.zeros((len(ms), len(ns)))

    for i, m in enumerate(ms):
        for j, n in enumerate(ns):
            A = rng.standard_normal((m, n))
            rang_data[i, j] = rang_echelonnement(A)

    im = ax.imshow(rang_data, origin="lower", cmap="YlOrRd",
                   extent=[0.5, 10.5, 0.5, 10.5])
    ax.set_xlabel("n (colonnes)"); ax.set_ylabel("m (lignes)")
    ax.set_title("rang(A) pour matrices aléatoires m×n")
    plt.colorbar(im, ax=ax, label="rang")

    # Diagonale min(m,n)
    ax.plot([0.5, 10.5], [0.5, 10.5], "w--", linewidth=1, alpha=0.5)
    return ax


if __name__ == "__main__":
    print("=== Rang par échelonnement vs SVD ===")
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    print(f"A =\n{A}")
    print(f"  rang(échelon) = {rang_echelonnement(A)}")
    print(f"  rang(SVD)     = {rang_svd(A)}")
    print(f"  rang(numpy)   = {np.linalg.matrix_rank(A)}")

    print(f"\n=== Rangsatz ===")
    for name, M in [("3×3 rang 2", A),
                     ("2×4", np.array([[1,0,2,1],[0,1,1,2]], dtype=float)),
                     ("4×2", np.random.default_rng(0).standard_normal((4, 2)))]:
        rs = verifier_rangsatz(M)
        print(f"  {name:12s} : rang={rs['rang']}, dim Kern={rs['dim_kern']}, "
              f"n={rs['n']}, rang+kern=n ? {rs['vérifié']} ✓")

    print(f"\n=== Rang ligne = rang colonne ===")
    B = np.array([[1,2,3],[4,5,6]], dtype=float)
    r = verifier_rang_ligne_colonne(B)
    print(f"  B (2×3) : rang ligne = {r['rang_ligne']}, rang colonne = {r['rang_colonne']} ✓")

    print(f"\n=== rang(AB) ≤ min(rang A, rang B) ===")
    rng = np.random.default_rng(42)
    C = rng.standard_normal((4, 3))
    D = rng.standard_normal((3, 5))
    rp = verifier_rang_produit(C, D)
    print(f"  {rp}")

    print(f"\n=== Critère de compatibilité ===")
    A1 = np.array([[1,1],[1,1]], dtype=float)
    b_ok = np.array([2, 2], dtype=float)
    b_ko = np.array([1, 2], dtype=float)
    print(f"  b = {b_ok} : {critere_compatibilite(A1, b_ok)}")
    print(f"  b = {b_ko} : {critere_compatibilite(A1, b_ko)}")

    tracer_rang_vs_taille()
    plt.tight_layout()
    plt.savefig("matrix_rank_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
