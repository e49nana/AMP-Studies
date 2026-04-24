"""
qr_algorithm.py
===============

QR-Algorithmus pour le calcul des valeurs propres.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 6.7.

Couvre :
    - QR-Algorithmus de base (section 6.7.2)
    - QR-Algorithmus avec shift (section 6.7.3)
    - Réduction à la forme Hessenberg (section 6.7.4)
    - Convergence observée (section 6.7.5)

Principe : itérer A_{k+1} = R_k Q_k (où Q_k R_k = A_k est la QR-Zerlegung).
La suite A_k converge vers une matrice triangulaire supérieure dont les
éléments diagonaux sont les valeurs propres.

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Réduction à la forme Hessenberg supérieure (section 6.7.4)
# ======================================================================

def hessenberg(A: np.ndarray) -> np.ndarray:
    """
    Réduit A en forme Hessenberg supérieure par transformations
    orthogonales (réflexions de Householder).

    Coût : O(10n³/3). À faire une seule fois — ensuite chaque pas
    QR ne coûte que O(n²) au lieu de O(n³).
    """
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    for k in range(n - 2):
        # Réflexion de Householder sur la sous-colonne A[k+1:, k]
        x = A[k + 1:, k].copy()
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            continue
        sign = 1.0 if x[0] >= 0 else -1.0
        w = x.copy()
        w[0] += sign * norm_x
        ww = np.dot(w, w)
        # Q = I - 2 wwᵀ/||w||²
        # A ← QAQᵀ  (transformation de similitude)
        # Étape gauche : A[k+1:, :] ← (I - 2ww'/ww)A[k+1:, :]
        A[k + 1:, :] -= (2.0 / ww) * np.outer(w, w @ A[k + 1:, :])
        # Étape droite : A[:, k+1:] ← A[:, k+1:](I - 2ww'/ww)
        A[:, k + 1:] -= (2.0 / ww) * np.outer(A[:, k + 1:] @ w, w)
    return A


# ======================================================================
#  2. QR-Algorithmus de base (section 6.7.2)
# ======================================================================

@dataclass
class ResultatQRAlgo:
    eigenvalues: np.ndarray
    iterations: int
    converge: bool
    historique_off: list[float] = field(default_factory=list)


def off_diag(A: np.ndarray) -> float:
    """Somme des |a_ij| sous-diagonaux (mesure de convergence)."""
    n = A.shape[0]
    return float(sum(abs(A[i, j]) for i in range(n) for j in range(i)))


def qr_algorithmus(
    A: np.ndarray,
    tol: float = 1e-12,
    n_max: int = 10_000,
    shift: bool = True,
    use_hessenberg: bool = True,
) -> ResultatQRAlgo:
    """
    QR-Algorithmus.

    Chaque itération :
        1. (Optionnel) shift : μ = a_nn
        2. QR-Zerlegung de (A - μI)
        3. A ← RQ + μI

    Avec shift, convergence cubique pour matrices symétriques (section 6.7.3).
    Sans shift, convergence linéaire avec rate |λ_{n-1}/λ_n|.
    """
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]

    if use_hessenberg:
        A = hessenberg(A)

    hist_off = [off_diag(A)]
    converge = False

    for k in range(1, n_max + 1):
        # Shift de Wilkinson simplifié : prendre a_nn
        if shift:
            mu = A[-1, -1]
            A_shifted = A - mu * np.eye(n)
        else:
            mu = 0.0
            A_shifted = A

        # QR-Zerlegung
        Q, R = np.linalg.qr(A_shifted)

        # Nouvelle itération : A = RQ + μI
        A = R @ Q + mu * np.eye(n)

        o = off_diag(A)
        hist_off.append(o)
        if o < tol:
            converge = True
            break

    return ResultatQRAlgo(
        eigenvalues=np.sort(np.diag(A)),
        iterations=k,
        converge=converge,
        historique_off=hist_off,
    )


# ======================================================================
#  3. Tracé
# ======================================================================

def tracer_convergence_qr(
    resultats: list[ResultatQRAlgo],
    labels: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(resultats):
        lbl = labels[i] if labels else f"Variante {i+1}"
        vals = [max(v, 1e-18) for v in r.historique_off]
        ax.semilogy(vals, "-", linewidth=2, label=f"{lbl} ({r.iterations} it.)")
    ax.set_xlabel("itération")
    ax.set_ylabel("off$(A)$")
    ax.set_title("Section 6.7 — convergence du QR-Algorithmus")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    print("=== Matrice symétrique 5×5 ===")
    A = np.array([
        [5, 1, 0, 0.5, 0],
        [1, 4, 1, 0, 0.3],
        [0, 1, 3, 1, 0],
        [0.5, 0, 1, 2, 0.5],
        [0, 0.3, 0, 0.5, 1],
    ], dtype=float)

    eigvals_np = np.sort(np.linalg.eigvalsh(A))
    print(f"numpy          : {eigvals_np}")

    res_no = qr_algorithmus(A, shift=False, use_hessenberg=False)
    res_hess = qr_algorithmus(A, shift=False, use_hessenberg=True)
    res_shift = qr_algorithmus(A, shift=True, use_hessenberg=True)

    print(f"QR basique     : {res_no.eigenvalues}  ({res_no.iterations} it.)")
    print(f"QR + Hessenberg: {res_hess.eigenvalues}  ({res_hess.iterations} it.)")
    print(f"QR + shift     : {res_shift.eigenvalues}  ({res_shift.iterations} it.)")
    print(f"||shift - numpy|| = {np.max(np.abs(res_shift.eigenvalues - eigvals_np)):.2e}")

    print("\n=== Matrice non-symétrique 4×4 ===")
    B = np.array([[6, 2, 1, 0], [1, 5, 1, 1], [0.5, 1, 4, 0.5], [0, 0.5, 1, 3]], dtype=float)
    eigvals_B = np.sort(np.linalg.eigvals(B).real)
    res_B = qr_algorithmus(B, shift=True)
    print(f"numpy : {eigvals_B}")
    print(f"QR    : {res_B.eigenvalues}  ({res_B.iterations} it.)")

    print("\n=== Tracé ===")
    tracer_convergence_qr(
        [res_no, res_hess, res_shift],
        ["Sans Hessenberg, sans shift", "Hessenberg, sans shift", "Hessenberg + shift"],
    )
    plt.tight_layout()
    plt.savefig("qr_algorithm_demo.png", dpi=120)
    print("Figure sauvegardée : qr_algorithm_demo.png")
