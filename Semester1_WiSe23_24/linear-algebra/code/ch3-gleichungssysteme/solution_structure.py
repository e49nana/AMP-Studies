"""
solution_structure.py
=====================

Structure de l'ensemble des solutions d'un système linéaire.

Couvre :
    - Système homogène Ax = 0 : l'ensemble des solutions est un sous-espace
    - Système inhomogène Ax = b : solution = particulière + homogène
    - Lösungsmenge = x_part + Kern(A)
    - Théorème de superposition
    - Visualisation en R² et R³

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def kern(A: np.ndarray, tol: float = 1e-10) -> list[np.ndarray]:
    """Base du noyau Kern(A) par RREF."""
    A = np.asarray(A, dtype=float).copy()
    m, n = A.shape
    # RREF
    pivot_cols = []
    row = 0
    for col in range(n):
        if row >= m:
            break
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

    free = [j for j in range(n) if j not in pivot_cols]
    base = []
    for fc in free:
        v = np.zeros(n)
        v[fc] = 1.0
        for i, pc in enumerate(pivot_cols):
            v[pc] = -A[i, fc]
        base.append(v)
    return base


def solution_particuliere(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Une solution particulière de Ax = b, ou None si incompatible."""
    try:
        x, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        if np.linalg.norm(A @ x - b) > 1e-8:
            return None
        return x
    except np.linalg.LinAlgError:
        return None


def analyser_systeme(A: np.ndarray, b: np.ndarray) -> dict:
    """Analyse complète de la structure de l'ensemble des solutions."""
    m, n = A.shape
    r = np.linalg.matrix_rank(A)
    aug = np.column_stack([A, b])
    r_aug = np.linalg.matrix_rank(aug)

    result = {
        "taille": f"{m}×{n}",
        "rang_A": r,
        "rang_augmentee": r_aug,
        "compatible": r == r_aug,
        "dim_kern": n - r,
    }

    if r == r_aug:
        if n - r == 0:
            result["type"] = "solution unique"
        else:
            result["type"] = f"∞ solutions ({n - r} paramètre(s))"
        result["x_part"] = solution_particuliere(A, b)
        result["kern_base"] = kern(A)
    else:
        result["type"] = "incompatible"
        result["x_part"] = None
        result["kern_base"] = []

    return result


def demo_superposition() -> None:
    """Illustre le théorème de superposition."""
    print("=== Théorème de superposition ===")
    print("  Si x₁ est sol. de Ax = b₁ et x₂ sol. de Ax = b₂,")
    print("  alors x₁ + x₂ est sol. de Ax = b₁ + b₂.\n")

    A = np.array([[1, 2], [3, 4]], dtype=float)
    b1 = np.array([5, 6], dtype=float)
    b2 = np.array([7, 8], dtype=float)

    x1 = np.linalg.solve(A, b1)
    x2 = np.linalg.solve(A, b2)
    x_sum = np.linalg.solve(A, b1 + b2)

    print(f"  Ax₁ = b₁ : x₁ = {x1}")
    print(f"  Ax₂ = b₂ : x₂ = {x2}")
    print(f"  A(x₁+x₂) = b₁+b₂ : x₁+x₂ = {x1 + x2}")
    print(f"  Solution directe   : x = {x_sum}")
    print(f"  Identiques ? {np.allclose(x1 + x2, x_sum)} ✓")


def tracer_solutions_2d(
    A: np.ndarray, b: np.ndarray, ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualise l'ensemble des solutions en R²."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    result = analyser_systeme(A, b)

    if result["type"] == "incompatible":
        # Tracer les droites qui ne se croisent pas
        x_range = np.linspace(-5, 5, 200)
        for i in range(A.shape[0]):
            if abs(A[i, 1]) > 1e-10:
                y = (b[i] - A[i, 0] * x_range) / A[i, 1]
                ax.plot(x_range, y, linewidth=2, label=f"éq. {i+1}")
            else:
                ax.axvline(b[i] / A[i, 0], linewidth=2, label=f"éq. {i+1}")
        ax.set_title("Incompatible — pas d'intersection")

    elif result["type"] == "solution unique":
        x_range = np.linspace(-5, 5, 200)
        for i in range(A.shape[0]):
            if abs(A[i, 1]) > 1e-10:
                y = (b[i] - A[i, 0] * x_range) / A[i, 1]
                ax.plot(x_range, y, linewidth=2, label=f"éq. {i+1}")
        x = result["x_part"]
        ax.plot(x[0], x[1], "ko", markersize=10, label=f"x* = ({x[0]:.2f}, {x[1]:.2f})")
        ax.set_title("Solution unique")

    else:
        x_part = result["x_part"]
        kern_base = result["kern_base"]
        if kern_base:
            d = kern_base[0]
            ts = np.linspace(-3, 3, 50)
            line = np.array([x_part + t * d for t in ts])
            ax.plot(line[:, 0], line[:, 1], "g-", linewidth=3,
                    label=f"x = x_part + t·k")
            ax.plot(x_part[0], x_part[1], "ro", markersize=8, label="x_part")
            ax.quiver(x_part[0], x_part[1], d[0], d[1],
                      angles="xy", scale_units="xy", scale=1,
                      color="orange", width=0.012, label="direction Kern")
        ax.set_title(f"{result['type']}")

    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    return ax


if __name__ == "__main__":
    print("=== Cas 1 : solution unique ===")
    A1 = np.array([[1, 1], [1, -1]], dtype=float)
    b1 = np.array([4, 2], dtype=float)
    r1 = analyser_systeme(A1, b1)
    print(f"  {r1['type']}, x = {r1['x_part']}\n")

    print("=== Cas 2 : infinité de solutions ===")
    A2 = np.array([[1, 2, 3], [2, 4, 6]], dtype=float)
    b2 = np.array([1, 2], dtype=float)
    r2 = analyser_systeme(A2, b2)
    print(f"  {r2['type']}")
    print(f"  x_part = {r2['x_part']}")
    print(f"  Kern = {[v.tolist() for v in r2['kern_base']]}\n")

    print("=== Cas 3 : incompatible ===")
    A3 = np.array([[1, 1], [1, 1]], dtype=float)
    b3 = np.array([1, 2], dtype=float)
    r3 = analyser_systeme(A3, b3)
    print(f"  {r3['type']}\n")

    demo_superposition()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_solutions_2d(A1, b1, ax=axes[0])
    tracer_solutions_2d(np.array([[1,2],[2,4]], dtype=float),
                        np.array([3, 6], dtype=float), ax=axes[1])
    tracer_solutions_2d(A3, b3, ax=axes[2])
    plt.tight_layout()
    plt.savefig("solution_structure_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
