"""
jacobi_eigenvalue.py
====================

Méthode de Jacobi pour les valeurs propres de matrices symétriques.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 6.5.

Couvre :
    - Rotations de Givens (section 6.5.2)
    - Jacobi classique : annuler le plus grand élément hors-diagonale
      (section 6.5.5, Variante 1)
    - Jacobi cyclique : parcourir systématiquement les éléments
      (section 6.5.5, Variante 2)
    - Convergence : mesure de off(A) = √(Σ_{i≠j} a_ij²) (section 6.5.7)

L'avantage du Jacobi : simple, robuste, et calcule aussi les vecteurs
propres (via accumulation des rotations). Inconvénient : O(n³) par
itération, convergence lente.

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Mesure hors-diagonale
# ======================================================================

def off(A: np.ndarray) -> float:
    """
    off(A) = √(Σ_{i≠j} a_ij²).

    Mesure la « distance à la diagonale ». Le Jacobi converge quand
    off(A) → 0 (section 6.5.7).
    """
    return float(np.sqrt(np.sum(A**2) - np.sum(np.diag(A)**2)))


# ======================================================================
#  2. Rotation de Givens (section 6.5.2)
# ======================================================================

def rotation_jacobi(A: np.ndarray, p: int, q: int) -> tuple[np.ndarray, float, float]:
    """
    Calcule la rotation de Givens qui annule a_pq et a_qp.

    Renvoie (A', c, s) où A' = Uᵀ A U avec U rotation d'angle θ
    dans le plan (p, q).

    Formules de la section 6.5.3 :
        θ = 0.5 · atan2(2 a_pq, a_qq - a_pp)
        c = cos(θ), s = sin(θ)
    """
    A = A.copy()
    if abs(A[p, q]) < 1e-300:
        return A, 1.0, 0.0

    if A[p, p] == A[q, q]:
        theta = np.pi / 4
    else:
        theta = 0.5 * np.arctan2(2 * A[p, q], A[q, q] - A[p, p])

    c = np.cos(theta)
    s = np.sin(theta)
    n = A.shape[0]

    # Appliquer Uᵀ A U (section 6.5.4)
    # Lignes p et q de A → rotées par Uᵀ (à gauche)
    for j in range(n):
        if j == p or j == q:
            continue
        aip = A[p, j]
        aiq = A[q, j]
        A[p, j] = c * aip - s * aiq
        A[q, j] = s * aip + c * aiq
        A[j, p] = A[p, j]  # symétrie
        A[j, q] = A[q, j]

    # Éléments diagonaux et hors-diagonale (p,q)
    app = A[p, p]
    aqq = A[q, q]
    apq = A[p, q]

    A[p, p] = c**2 * app - 2 * c * s * apq + s**2 * aqq
    A[q, q] = s**2 * app + 2 * c * s * apq + c**2 * aqq
    A[p, q] = 0.0
    A[q, p] = 0.0

    return A, c, s


# ======================================================================
#  3. Jacobi classique (section 6.5.5, Variante 1)
# ======================================================================

@dataclass
class ResultatJacobi:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    iterations: int
    converge: bool
    historique_off: list[float] = field(default_factory=list)


def jacobi_classique(
    A: np.ndarray,
    tol: float = 1e-12,
    n_max: int = 10_000,
) -> ResultatJacobi:
    """
    Jacobi classique : à chaque pas, annuler le plus grand |a_pq| (p ≠ q).

    Convergence quadratique garantie pour matrices symétriques (section 6.5.7).
    """
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    V = np.eye(n)  # accumulation des rotations → vecteurs propres
    hist_off = [off(A)]

    converge = False
    for k in range(1, n_max + 1):
        # Trouver le plus grand |a_pq| hors diagonale
        mask = np.ones_like(A, dtype=bool)
        np.fill_diagonal(mask, False)
        abs_A = np.abs(A) * mask
        idx = np.unravel_index(np.argmax(abs_A), A.shape)
        p, q = int(idx[0]), int(idx[1])

        if abs(A[p, q]) < tol:
            converge = True
            break

        # Rotation
        A, c, s = rotation_jacobi(A, p, q)

        # Accumulation : V ← V · U
        for i in range(n):
            vip = V[i, p]
            viq = V[i, q]
            V[i, p] = c * vip - s * viq
            V[i, q] = s * vip + c * viq

        hist_off.append(off(A))

    return ResultatJacobi(
        eigenvalues=np.diag(A),
        eigenvectors=V,
        iterations=k,
        converge=converge,
        historique_off=hist_off,
    )


def jacobi_cyclique(
    A: np.ndarray,
    tol: float = 1e-12,
    n_max: int = 1_000,
) -> ResultatJacobi:
    """
    Jacobi cyclique (Variante 2) : parcourir systématiquement tous
    les couples (p, q) avec p < q, un cycle complet = n(n-1)/2 rotations.
    """
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    V = np.eye(n)
    hist_off = [off(A)]

    converge = False
    sweep = 0
    for sweep in range(1, n_max + 1):
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p, q]) < tol * 1e-2:
                    continue
                A, c, s = rotation_jacobi(A, p, q)
                for i in range(n):
                    vip = V[i, p]
                    viq = V[i, q]
                    V[i, p] = c * vip - s * viq
                    V[i, q] = s * vip + c * viq

        hist_off.append(off(A))
        if off(A) < tol:
            converge = True
            break

    return ResultatJacobi(
        eigenvalues=np.diag(A),
        eigenvectors=V,
        iterations=sweep,
        converge=converge,
        historique_off=hist_off,
    )


# ======================================================================
#  4. Tracé
# ======================================================================

def tracer_convergence_jacobi(
    resultats: list[ResultatJacobi],
    labels: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(resultats):
        lbl = labels[i] if labels else f"Méthode {i+1}"
        ax.semilogy(r.historique_off, "o-", markersize=3, label=lbl)
    ax.set_xlabel("itération / sweep")
    ax.set_ylabel("off$(A)$")
    ax.set_title("Section 6.5.7 — convergence du Jacobi")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    print("=== Matrice symétrique 4×4 ===")
    A = np.array([
        [4, 1, 0.5, 0],
        [1, 3, 0.3, 0.1],
        [0.5, 0.3, 2, 0.2],
        [0, 0.1, 0.2, 1],
    ], dtype=float)

    res_c = jacobi_classique(A)
    res_cy = jacobi_cyclique(A)

    eigvals_np = np.sort(np.linalg.eigvalsh(A))
    eigvals_c = np.sort(res_c.eigenvalues)
    eigvals_cy = np.sort(res_cy.eigenvalues)

    print(f"numpy       : {eigvals_np}")
    print(f"Jac. class. : {eigvals_c}  ({res_c.iterations} rotations)")
    print(f"Jac. cycl.  : {eigvals_cy}  ({res_cy.iterations} sweeps)")
    print(f"||mine - numpy||_∞ = {np.max(np.abs(eigvals_c - eigvals_np)):.2e}")

    # Vérification AV = VΛ
    Lambda = np.diag(res_c.eigenvalues)
    err = np.linalg.norm(A @ res_c.eigenvectors - res_c.eigenvectors @ Lambda, np.inf)
    print(f"||AV - VΛ||_∞ = {err:.2e}")

    print("\n=== Tracé ===")
    tracer_convergence_jacobi([res_c, res_cy], ["Classique", "Cyclique"])
    plt.tight_layout()
    plt.savefig("jacobi_eigenvalue_demo.png", dpi=120)
    print("Figure sauvegardée : jacobi_eigenvalue_demo.png")
