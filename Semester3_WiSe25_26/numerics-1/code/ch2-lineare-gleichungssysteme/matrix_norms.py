"""
matrix_norms.py
===============

Normes matricielles induites, nombre de conditionnement, et rayon spectral.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", sections 2.1 et 2.2.

Couvre :
    - Zeilensummennorm ||A||_∞ (Satz 2.5)
    - Spaltensummennorm ||A||_1 (Satz 2.7)
    - Spektralnorm ||A||_2 = σ_max(A) (Satz 2.9)
    - Frobenius-Norm ||A||_F (Satz 2.10)
    - Gesamtnorm ||A||_G (section 2.1.2)
    - Rayon spectral ρ(A) et borne ρ(A) ≤ ||A|| (Satz 2.14)
    - Nombre de conditionnement κ(A) = ||A|| · ||A⁻¹|| (Satz 2.17)
    - Vérification de la submultiplikativité (Satz 2.2)
    - Vérification de la verträglichkeit (Def. 2.4)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Normes matricielles from-scratch
# ======================================================================

def norm_inf_mat(A: np.ndarray) -> float:
    """
    Zeilensummennorm (Satz 2.5) :
        ||A||_∞ = max_i Σ_j |a_ij|.

    C'est la norme matricielle induite par la norme vectorielle ∞.
    Astuce mnémotechnique du script : appliquée à un vecteur colonne,
    la Zeilensummennorm donne le max des composantes = norme ∞ du vecteur.
    """
    return float(np.max(np.sum(np.abs(A), axis=1)))


def norm_1_mat(A: np.ndarray) -> float:
    """
    Spaltensummennorm (Satz 2.7) :
        ||A||_1 = max_j Σ_i |a_ij|.

    Norme induite par la norme vectorielle 1.
    """
    return float(np.max(np.sum(np.abs(A), axis=0)))


def norm_2_mat(A: np.ndarray) -> float:
    """
    Spektralnorm (Satz 2.9) :
        ||A||_2 = √(λ_max(A* A)) = σ_max(A).

    Plus coûteuse à calculer (requiert les valeurs propres de A*A).
    """
    AhA = A.conj().T @ A
    eigvals = np.linalg.eigvalsh(AhA)  # hermitienne → eigvalsh
    return float(np.sqrt(np.max(eigvals)))


def norm_frobenius(A: np.ndarray) -> float:
    """
    Frobenius-Norm (Satz 2.10) :
        ||A||_F = √(Σ_ij |a_ij|²).

    Verträgliche avec la 2-norme vectorielle, mais pas identique
    à la Spektralnorm. En général ||A||_2 ≤ ||A||_F ≤ √n · ||A||_2.
    """
    return float(np.sqrt(np.sum(np.abs(A) ** 2)))


def norm_gesamt(A: np.ndarray) -> float:
    """
    Gesamtnorm (section 2.1.2) :
        ||A||_G = n · max_{i,j} |a_ij|.

    Verträgliche avec toutes les p-normes.
    """
    n = A.shape[0]
    return float(n * np.max(np.abs(A)))


# ======================================================================
#  2. Rayon spectral (Satz 2.14)
# ======================================================================

def rayon_spectral(A: np.ndarray) -> float:
    """
    ρ(A) = max_i |λ_i| (Définition 2.13).

    Par Satz 2.14 : ρ(A) ≤ ||A|| pour toute norme matricielle.
    Par Satz 2.15 : il existe une norme telle que ||A|| ≤ ρ(A) + ε.
    """
    return float(np.max(np.abs(np.linalg.eigvals(A))))


# ======================================================================
#  3. Nombre de conditionnement (Satz 2.17)
# ======================================================================

def condition(A: np.ndarray, p: float | str = np.inf) -> float:
    """
    κ_p(A) = ||A||_p · ||A⁻¹||_p  (Satz 2.17).

    Interprétation : κ(A) est le facteur de grossissement maximal
    des erreurs relatives dans la résolution de Ax = b.

    Paramètres
    ----------
    p : 1, 2, np.inf, ou 'fro' pour Frobenius.
    """
    norms = {
        1: norm_1_mat,
        2: norm_2_mat,
        np.inf: norm_inf_mat,
        "fro": norm_frobenius,
    }
    norm_fn = norms.get(p)
    if norm_fn is None:
        raise ValueError(f"p doit être 1, 2, np.inf ou 'fro', reçu {p}.")

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float("inf")
    return norm_fn(A) * norm_fn(A_inv)


# ======================================================================
#  4. Vérifications numériques
# ======================================================================

def verifier_submultiplicativite(
    A: np.ndarray, B: np.ndarray, n_normes: int = 4,
) -> dict[str, tuple[float, float, bool]]:
    """
    Vérifie ||AB|| ≤ ||A|| · ||B|| (Satz 2.2) pour toutes les normes.
    """
    norms = {
        "||·||_∞": norm_inf_mat,
        "||·||_1": norm_1_mat,
        "||·||_2": norm_2_mat,
        "||·||_F": norm_frobenius,
    }
    AB = A @ B
    result = {}
    for name, fn in norms.items():
        lhs = fn(AB)
        rhs = fn(A) * fn(B)
        result[name] = (lhs, rhs, lhs <= rhs + 1e-12)
    return result


def verifier_borne_spectrale(A: np.ndarray) -> dict[str, tuple[float, float]]:
    """
    Vérifie ρ(A) ≤ ||A|| pour chaque norme (Satz 2.14).
    """
    rho = rayon_spectral(A)
    norms = {
        "||·||_∞": norm_inf_mat(A),
        "||·||_1": norm_1_mat(A),
        "||·||_2": norm_2_mat(A),
        "||·||_F": norm_frobenius(A),
        "||·||_G": norm_gesamt(A),
    }
    return {name: (rho, val) for name, val in norms.items()}


# ======================================================================
#  5. Comparaison from-scratch ↔ NumPy
# ======================================================================

def comparer_avec_numpy(A: np.ndarray) -> None:
    """Tableau comparatif de toutes les normes."""
    print(f"{'norme':>12} | {'from-scratch':>14} | {'numpy':>14} | {'écart':>10}")
    print("-" * 58)
    cas = [
        ("||·||_∞", norm_inf_mat(A), np.linalg.norm(A, np.inf)),
        ("||·||_1", norm_1_mat(A), np.linalg.norm(A, 1)),
        ("||·||_2", norm_2_mat(A), np.linalg.norm(A, 2)),
        ("||·||_F", norm_frobenius(A), np.linalg.norm(A, "fro")),
    ]
    for name, mine, ref in cas:
        print(f"{name:>12} | {mine:>14.8f} | {ref:>14.8f} | {abs(mine - ref):>10.2e}")


# ======================================================================
#  6. Tracé : κ(A) en fonction de la taille (matrices de Hilbert)
# ======================================================================

def tracer_condition_hilbert(
    tailles: tuple[int, ...] = tuple(range(2, 16)),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trace κ_∞(H_n) en fonction de n pour les matrices de Hilbert."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    conds = []
    for n in tailles:
        i = np.arange(1, n + 1)
        H = 1.0 / (i[:, None] + i[None, :] - 1)
        conds.append(condition(H, np.inf))

    ax.semilogy(list(tailles), conds, "bo-", markersize=5)
    ax.set_xlabel("taille $n$")
    ax.set_ylabel("$\\kappa_\\infty(H_n)$")
    ax.set_title("Satz 2.17 — conditionnement des matrices de Hilbert")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_borne_spectrale_aleatoire(
    n: int = 5, n_matrices: int = 200, ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Pour des matrices aléatoires, trace ρ(A) vs ||A|| pour
    vérifier visuellement que ρ(A) ≤ ||A|| (Satz 2.14).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    rng = np.random.default_rng(42)
    rhos, norms_inf = [], []
    for _ in range(n_matrices):
        A = rng.standard_normal((n, n))
        rhos.append(rayon_spectral(A))
        norms_inf.append(norm_inf_mat(A))

    ax.scatter(norms_inf, rhos, alpha=0.5, s=15)
    lim = max(max(norms_inf), max(rhos)) * 1.1
    ax.plot([0, lim], [0, lim], "r--", label="$\\rho(A) = \\|A\\|$")
    ax.set_xlabel("$\\|A\\|_\\infty$")
    ax.set_ylabel("$\\rho(A)$")
    ax.set_title("Satz 2.14 — tous les points sous la diagonale")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    print("=== Übung 2.6 : matrice 3×3 complexe ===")
    A = np.array([
        [3, 0, 2],
        [-4, 1 + 1j * np.sqrt(3), 2],
        [2j, -1, 1 - 1j],
    ])
    comparer_avec_numpy(A)
    print(f"\nρ(A) = {rayon_spectral(A):.6f}")

    print("\n=== Vérification ρ(A) ≤ ||A|| (Satz 2.14) ===")
    bornes = verifier_borne_spectrale(A)
    for name, (rho, val) in bornes.items():
        print(f"  ρ(A) = {rho:.4f} ≤ {name} = {val:.4f}  ✓" if rho <= val + 1e-10
              else f"  ρ(A) = {rho:.4f} > {name} = {val:.4f}  ✗")

    print("\n=== Submultiplikativité (Satz 2.2) ===")
    rng = np.random.default_rng(0)
    B = rng.standard_normal((3, 3))
    C = rng.standard_normal((3, 3))
    for name, (lhs, rhs, ok) in verifier_submultiplicativite(B, C).items():
        print(f"  {name}: ||BC|| = {lhs:.4f} ≤ ||B||·||C|| = {rhs:.4f}  {'✓' if ok else '✗'}")

    print("\n=== Conditionnement (Satz 2.17) ===")
    for n in [3, 5, 8, 10, 12]:
        i = np.arange(1, n + 1)
        H = 1.0 / (i[:, None] + i[None, :] - 1)
        print(f"  H_{n:>2} : κ_∞ = {condition(H, np.inf):>14.2e}  (numpy: {np.linalg.cond(H, np.inf):>14.2e})")

    print("\n=== Tracés ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_condition_hilbert(ax=axes[0])
    tracer_borne_spectrale_aleatoire(ax=axes[1])
    plt.tight_layout()
    plt.savefig("matrix_norms_demo.png", dpi=120)
    print("Figure sauvegardée : matrix_norms_demo.png")
