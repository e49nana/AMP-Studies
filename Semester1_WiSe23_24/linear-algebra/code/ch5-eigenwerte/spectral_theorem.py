"""
spectral_theorem.py
===================

Théorème spectral pour matrices symétriques : A = QΛQᵀ.

Couvre :
    - Théorème spectral : toute matrice symétrique réelle a une base
      orthonormée de vecteurs propres
    - Décomposition A = QΛQᵀ avec Q orthogonale
    - Valeurs propres réelles (preuve numérique)
    - Vecteurs propres orthogonaux (preuve numérique)
    - Quotient de Rayleigh : R(x) = xᵀAx / xᵀx
    - Matrices définies positives via le spectre

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def decomposition_spectrale(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    A = QΛQᵀ pour A symétrique.

    Q : matrice orthogonale (colonnes = vecteurs propres).
    Λ : matrice diagonale (valeurs propres).
    """
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T):
        raise ValueError("Matrice non symétrique.")
    eigvals, Q = np.linalg.eigh(A)  # eigh pour symétriques
    return Q, np.diag(eigvals)


def verifier_spectral(A: np.ndarray) -> dict:
    """Vérifie toutes les propriétés du théorème spectral."""
    Q, Lambda = decomposition_spectrale(A)
    eigvals = np.diag(Lambda)
    n = A.shape[0]
    return {
        "λ réelles": all(np.isreal(eigvals)),
        "QᵀQ = I": np.allclose(Q.T @ Q, np.eye(n)),
        "QQᵀ = I": np.allclose(Q @ Q.T, np.eye(n)),
        "QΛQᵀ = A": np.allclose(Q @ Lambda @ Q.T, A),
        "det(Q) = ±1": np.isclose(abs(np.linalg.det(Q)), 1),
    }


def rayleigh_quotient(A: np.ndarray, x: np.ndarray) -> float:
    """
    Quotient de Rayleigh : R(x) = xᵀAx / xᵀx.

    Propriétés :
        - λ_min ≤ R(x) ≤ λ_max pour tout x ≠ 0
        - R(vᵢ) = λᵢ pour un vecteur propre vᵢ
        - Le min/max de R donne les valeurs propres extrêmes
    """
    x = np.asarray(x, dtype=float)
    return float(x @ A @ x) / float(x @ x)


def est_definie_positive_spectral(A: np.ndarray) -> bool:
    """A est SPD ssi toutes les valeurs propres sont > 0."""
    eigvals = np.linalg.eigvalsh(A)
    return bool(np.all(eigvals > 0))


def decomposition_en_rang_1(A: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """
    Décomposition spectrale comme somme de matrices de rang 1 :
        A = Σ λᵢ qᵢ qᵢᵀ.

    Chaque terme λᵢ qᵢqᵢᵀ est une matrice de rang 1 (projection pondérée).
    """
    Q, Lambda = decomposition_spectrale(A)
    eigvals = np.diag(Lambda)
    termes = []
    for i in range(len(eigvals)):
        qi = Q[:, i]
        termes.append((eigvals[i], qi))
    return termes


def tracer_rayleigh(A: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Trace R(x) sur le cercle unité en R² (pour A 2×2)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    theta = np.linspace(0, 2*np.pi, 500)
    R_vals = []
    for t in theta:
        x = np.array([np.cos(t), np.sin(t)])
        R_vals.append(rayleigh_quotient(A, x))

    eigvals = np.linalg.eigvalsh(A)
    ax.plot(np.degrees(theta), R_vals, "b-", linewidth=2, label="$R(x)$")
    ax.axhline(eigvals[0], color="red", linestyle="--", label=f"$\\lambda_{{min}} = {eigvals[0]:.2f}$")
    ax.axhline(eigvals[1], color="green", linestyle="--", label=f"$\\lambda_{{max}} = {eigvals[1]:.2f}$")
    ax.set_xlabel("angle θ (degrés)")
    ax.set_ylabel("$R(x) = x^T A x / x^T x$")
    ax.set_title("Quotient de Rayleigh sur le cercle unité")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_ellipse_spectrale(A: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Montre que Ax transforme le cercle unité en une ellipse dont les
    axes sont les vecteurs propres et les demi-axes sont les valeurs propres.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    Q, Lambda = decomposition_spectrale(A)
    eigvals = np.diag(Lambda)

    theta = np.linspace(0, 2*np.pi, 200)
    cercle = np.array([np.cos(theta), np.sin(theta)])
    ellipse = A @ cercle

    ax.plot(cercle[0], cercle[1], "b-", alpha=0.4, linewidth=1, label="cercle unité")
    ax.plot(ellipse[0], ellipse[1], "r-", linewidth=2, label="$Ax$")

    # Axes = vecteurs propres × valeurs propres
    for i in range(2):
        v = Q[:, i] * eigvals[i]
        ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
                  color="green", width=0.012,
                  label=f"$\\lambda_{i+1} v_{i+1}$ = {np.round(v, 2)}")

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Théorème spectral : les axes de l'ellipse = vecteurs propres")
    ax.legend(fontsize=8)
    lim = max(abs(eigvals)) * 1.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    return ax


if __name__ == "__main__":
    print("=== Théorème spectral ===")
    A = np.array([[4, 2], [2, 3]], dtype=float)
    Q, Lambda = decomposition_spectrale(A)
    print(f"A =\n{A}")
    print(f"Q (vecteurs propres) =\n{np.round(Q, 6)}")
    print(f"Λ (valeurs propres) =\n{np.round(Lambda, 6)}")
    print(f"\nVérifications :")
    for prop, ok in verifier_spectral(A).items():
        print(f"  {prop:15s} : {ok} ✓")

    print(f"\n=== Décomposition en rang 1 ===")
    termes = decomposition_en_rang_1(A)
    somme = np.zeros_like(A)
    for lam, q in termes:
        rang1 = lam * np.outer(q, q)
        somme += rang1
        print(f"  λ={lam:.4f} · qqᵀ =\n{np.round(rang1, 4)}")
    print(f"  Somme =\n{np.round(somme, 6)}")
    print(f"  ||Somme - A|| = {np.linalg.norm(somme - A):.2e}")

    print(f"\n=== Quotient de Rayleigh ===")
    eigvals = np.diag(Lambda)
    for i in range(2):
        R = rayleigh_quotient(A, Q[:, i])
        print(f"  R(v_{i+1}) = {R:.6f} = λ_{i+1} = {eigvals[i]:.6f} ✓")

    print(f"\n=== Définie positive ===")
    for name, M in [("A", A), ("I", np.eye(2)), ("-A", -A)]:
        print(f"  {name} SPD ? {est_definie_positive_spectral(M)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_rayleigh(A, ax=axes[0])
    tracer_ellipse_spectrale(A, ax=axes[1])
    plt.tight_layout()
    plt.savefig("spectral_theorem_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
