"""
subspaces.py
============

Sous-espaces vectoriels : noyau (Kern), image (Bild), théorème du rang.

Couvre :
    - Kern(A) = {x : Ax = 0} par échelonnement
    - Bild(A) = espace des colonnes
    - Théorème du rang : dim Kern(A) + dim Bild(A) = n
    - Visualisation des sous-espaces en R² et R³

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rref(A: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, list[int]]:
    """
    Forme échelonnée réduite (Reduced Row Echelon Form).

    Renvoie (R, pivot_cols).
    """
    A = np.asarray(A, dtype=float).copy()
    m, n = A.shape
    pivot_cols = []
    row = 0

    for col in range(n):
        i_max = row + np.argmax(np.abs(A[row:, col]))
        if abs(A[i_max, col]) < tol:
            continue
        A[[row, i_max]] = A[[i_max, row]]
        A[row] /= A[row, col]
        for i in range(m):
            if i != row:
                A[i] -= A[i, col] * A[row]
        pivot_cols.append(col)
        row += 1

    return A, pivot_cols


def kern(A: np.ndarray) -> list[np.ndarray]:
    """
    Base du noyau Kern(A) = {x : Ax = 0}.

    Méthode : RREF, puis pour chaque variable libre, construire
    un vecteur de la base du noyau.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    R, pivot_cols = rref(A)
    free_cols = [j for j in range(n) if j not in pivot_cols]

    if not free_cols:
        return []  # noyau trivial

    base_kern = []
    for fc in free_cols:
        x = np.zeros(n)
        x[fc] = 1.0
        for i, pc in enumerate(pivot_cols):
            x[pc] = -R[i, fc]
        base_kern.append(x)

    return base_kern


def bild(A: np.ndarray) -> list[np.ndarray]:
    """
    Base de l'image Bild(A) = espace des colonnes de A.

    Méthode : les colonnes correspondant aux pivots de RREF(A)
    forment une base de Bild(A) (prises dans A original).
    """
    A = np.asarray(A, dtype=float)
    _, pivot_cols = rref(A)
    return [A[:, j] for j in pivot_cols]


def rang(A: np.ndarray) -> int:
    """rang(A) = nombre de pivots."""
    _, pivot_cols = rref(A)
    return len(pivot_cols)


def verifier_rangsatz(A: np.ndarray) -> dict:
    """
    Vérifie le Rangsatz (théorème du rang) :
        dim Kern(A) + dim Bild(A) = n (nombre de colonnes).
    """
    m, n = A.shape
    k = kern(A)
    b = bild(A)
    dim_kern = len(k)
    dim_bild = len(b)
    return {
        "taille": f"{m}×{n}",
        "dim_kern": dim_kern,
        "dim_bild": dim_bild,
        "n": n,
        "rangsatz": dim_kern + dim_bild == n,
    }


def tracer_kern_bild_2d(A: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Visualise Kern(A) et Bild(A) pour une matrice 2×2."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    # Grille de points et leurs images
    t = np.linspace(-2, 2, 11)
    for s in t:
        for u in t:
            x = np.array([s, u])
            y = A @ x
            ax.plot(x[0], x[1], "b.", markersize=3, alpha=0.3)
            ax.plot(y[0], y[1], "r.", markersize=3, alpha=0.3)
            ax.annotate("", xy=y, xytext=x,
                        arrowprops=dict(arrowstyle="->", color="grey", alpha=0.1))

    # Kern
    k = kern(A)
    if k:
        v = k[0]
        ts = np.linspace(-3, 3, 50)
        ax.plot(v[0]*ts, v[1]*ts, "b-", linewidth=2, label=f"Kern(A) = span({v})")

    # Bild
    b_base = bild(A)
    if len(b_base) == 1:
        v = b_base[0]
        ts = np.linspace(-5, 5, 50)
        ax.plot(v[0]*ts, v[1]*ts, "r-", linewidth=2, label=f"Bild(A) = span({v})")

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"Kern et Bild de A (rang = {rang(A)})")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    return ax


if __name__ == "__main__":
    print("=== Exemple 1 : matrice de rang 2 (3×3) ===")
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    print(f"A =\n{A}")
    k = kern(A)
    b = bild(A)
    rs = verifier_rangsatz(A)
    print(f"Kern(A) : {[v.tolist() for v in k]}")
    print(f"Bild(A) : {[v.tolist() for v in b]}")
    print(f"Rangsatz : dim Kern + dim Bild = {rs['dim_kern']} + {rs['dim_bild']} = {rs['n']} ✓")

    print(f"\n=== Exemple 2 : matrice de rang plein (2×2) ===")
    B = np.array([[1, 2], [3, 4]], dtype=float)
    rs2 = verifier_rangsatz(B)
    print(f"dim Kern = {rs2['dim_kern']}, dim Bild = {rs2['dim_bild']}, n = {rs2['n']}")
    print(f"Kern trivial : {len(kern(B)) == 0} ✓")

    print(f"\n=== Exemple 3 : projection ===")
    P = np.array([[1, 0], [0, 0]], dtype=float)  # projection sur x
    print(f"P = {P.tolist()} (projection sur l'axe x)")
    k = kern(P)
    b = bild(P)
    print(f"Kern(P) = span({k[0].tolist()}) (axe y)")
    print(f"Bild(P) = span({b[0].tolist()}) (axe x)")

    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Matrice de rang 1
    C = np.array([[1, 2], [2, 4]], dtype=float)
    tracer_kern_bild_2d(C, ax=axes[0])
    axes[0].set_title("A = [[1,2],[2,4]] — rang 1")
    tracer_kern_bild_2d(P, ax=axes[1])
    axes[1].set_title("P = [[1,0],[0,0]] — projection")
    plt.tight_layout()
    plt.savefig("subspaces_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
