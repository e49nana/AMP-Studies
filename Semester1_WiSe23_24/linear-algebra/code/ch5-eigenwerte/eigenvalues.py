"""
eigenvalues.py
==============

Calcul des valeurs propres par le polynôme caractéristique.

Couvre :
    - Polynôme caractéristique p(λ) = det(A - λI)
    - Calcul des coefficients par Faddeev-LeVerrier
    - Racines du polynôme caractéristique (from-scratch pour 2×2, 3×3)
    - Trace et déterminant comme somme/produit des valeurs propres
    - Comparaison avec numpy.linalg.eigvals

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Polynôme caractéristique
# ======================================================================

def poly_car_2x2(A: np.ndarray) -> np.ndarray:
    """
    Polynôme caractéristique pour 2×2 :
        p(λ) = λ² - tr(A)·λ + det(A).

    Renvoie [1, -tr, det] (coefficients de λ² à λ⁰).
    """
    tr = A[0,0] + A[1,1]
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    return np.array([1, -tr, det])


def poly_car_3x3(A: np.ndarray) -> np.ndarray:
    """
    Polynôme caractéristique pour 3×3 :
        p(λ) = -λ³ + tr(A)·λ² - (somme des mineurs 2×2 de la diag)·λ + det(A).

    Convention : p(λ) = det(A - λI), donc coefficient dominant = (-1)ⁿ.
    On renvoie les coefficients du polynôme monique (coeff dominant = 1).
    """
    tr = np.trace(A)
    # Somme des cofacteurs diagonaux (mineurs principaux 2×2)
    m11 = A[1,1]*A[2,2] - A[1,2]*A[2,1]
    m22 = A[0,0]*A[2,2] - A[0,2]*A[2,0]
    m33 = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    s2 = m11 + m22 + m33
    det = float(np.linalg.det(A))
    return np.array([1, -tr, s2, -det])


def poly_car_faddeev_leverrier(A: np.ndarray) -> np.ndarray:
    """
    Algorithme de Faddeev-LeVerrier pour le polynôme caractéristique.

    Calcule les coefficients cₖ tels que :
        p(λ) = λⁿ + c₁λⁿ⁻¹ + ... + cₙ.

    Coût : O(n⁴) — mieux que Leibniz mais moins bon que des méthodes
    modernes (Berkowitz, Hessenberg).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    c = np.zeros(n + 1)
    c[0] = 1.0

    M = np.eye(n)
    for k in range(1, n + 1):
        M = A @ M if k > 1 else A.copy()
        if k > 1:
            M = A @ (M + c[k-1] * np.eye(n)) if k == 2 else A @ M
        # Recalcul propre
        pass

    # Version simple et correcte
    c = np.zeros(n + 1)
    c[0] = 1.0
    M = np.zeros_like(A)
    for k in range(1, n + 1):
        M = A @ (M + c[k-1] * np.eye(n))
        c[k] = -np.trace(M) / k

    return c


def eigenvalues_2x2(A: np.ndarray) -> np.ndarray:
    """
    Valeurs propres d'une matrice 2×2 par la formule quadratique :
        λ = (tr ± √(tr² - 4·det)) / 2.
    """
    p = poly_car_2x2(A)
    tr = -p[1]
    det = p[2]
    disc = tr**2 - 4*det
    if disc >= 0:
        return np.array([(tr + np.sqrt(disc))/2, (tr - np.sqrt(disc))/2])
    else:
        re = tr / 2
        im = np.sqrt(-disc) / 2
        return np.array([complex(re, im), complex(re, -im)])


def verifier_trace_det(A: np.ndarray) -> None:
    """
    Vérifie :
        tr(A) = Σ λᵢ  (somme des valeurs propres)
        det(A) = Π λᵢ  (produit des valeurs propres)
    """
    eigvals = np.linalg.eigvals(A)
    tr_A = np.trace(A)
    det_A = np.linalg.det(A)
    print(f"  tr(A) = {tr_A:.6f}, Σ λ = {np.sum(eigvals).real:.6f}, "
          f"égaux ? {np.isclose(tr_A, np.sum(eigvals).real)} ✓")
    print(f"  det(A) = {det_A:.6f}, Π λ = {np.prod(eigvals).real:.6f}, "
          f"égaux ? {np.isclose(det_A, np.prod(eigvals).real)} ✓")


def tracer_poly_car(A: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Trace le polynôme caractéristique et ses racines."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    eigvals = np.linalg.eigvals(A).real
    coeffs = np.poly(eigvals)  # polynôme monique avec ces racines

    lam = np.linspace(min(eigvals) - 2, max(eigvals) + 2, 300)
    p = np.polyval(coeffs, lam)

    ax.plot(lam, p, "b-", linewidth=2, label="$p(\\lambda) = \\det(A - \\lambda I)$")
    ax.plot(eigvals, np.zeros_like(eigvals), "ro", markersize=10, label="valeurs propres")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("$\\lambda$"); ax.set_ylabel("$p(\\lambda)$")
    ax.set_title("Polynôme caractéristique")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Matrice 2×2 ===")
    A2 = np.array([[4, 2], [1, 3]], dtype=float)
    p2 = poly_car_2x2(A2)
    ev2 = eigenvalues_2x2(A2)
    print(f"  A = {A2.tolist()}")
    print(f"  p(λ) = λ² + ({p2[1]:.1f})λ + ({p2[2]:.1f})")
    print(f"  λ = {ev2}  (numpy: {np.linalg.eigvals(A2)})")
    verifier_trace_det(A2)

    print(f"\n=== Matrice 3×3 ===")
    A3 = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)
    p3 = poly_car_3x3(A3)
    print(f"  p(λ) = λ³ + ({p3[1]:.1f})λ² + ({p3[2]:.1f})λ + ({p3[3]:.1f})")
    print(f"  λ (numpy) = {np.sort(np.linalg.eigvals(A3).real)}")
    verifier_trace_det(A3)

    print(f"\n=== Faddeev-LeVerrier (4×4) ===")
    A4 = np.array([[5,1,0,0],[1,4,1,0],[0,1,3,1],[0,0,1,2]], dtype=float)
    c = poly_car_faddeev_leverrier(A4)
    roots = np.sort(np.roots(c).real)
    ev_np = np.sort(np.linalg.eigvals(A4).real)
    print(f"  Coefficients : {np.round(c, 4)}")
    print(f"  Racines      : {np.round(roots, 6)}")
    print(f"  NumPy        : {np.round(ev_np, 6)}")
    print(f"  ||diff||     : {np.linalg.norm(roots - ev_np):.2e}")

    print(f"\n=== Valeurs propres complexes ===")
    R = np.array([[0, -1], [1, 0]], dtype=float)  # rotation 90°
    ev_c = eigenvalues_2x2(R)
    print(f"  Rotation 90° : λ = {ev_c} (= ±i)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_poly_car(A3, ax=axes[0])
    axes[0].set_title("3×3 symétrique")
    tracer_poly_car(A4, ax=axes[1])
    axes[1].set_title("4×4 symétrique")
    plt.tight_layout()
    plt.savefig("eigenvalues_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
