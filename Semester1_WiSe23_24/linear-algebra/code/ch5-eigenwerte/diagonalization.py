"""
diagonalization.py
==================

Diagonalisation : A = P D P⁻¹.

Couvre :
    - Construction de P (matrice des vecteurs propres)
    - Construction de D (matrice diagonale des valeurs propres)
    - Vérification A = PDP⁻¹
    - Critère de diagonalisabilité
    - Application : calcul de Aⁿ via PD^nP⁻¹

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def diagonaliser(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonalise A = P D P⁻¹.

    P : matrice dont les colonnes sont les vecteurs propres.
    D : matrice diagonale des valeurs propres.
    P_inv : inverse de P.

    Lève LinAlgError si A n'est pas diagonalisable.
    """
    A = np.asarray(A, dtype=float)
    eigvals, eigvecs = np.linalg.eig(A)

    # Vérifier que P est inversible (= A diagonalisable)
    P = eigvecs
    if abs(np.linalg.det(P)) < 1e-10:
        raise np.linalg.LinAlgError("Matrice non diagonalisable (P singulière).")

    D = np.diag(eigvals)
    P_inv = np.linalg.inv(P)

    return P, D, P_inv


def puissance_par_diag(A: np.ndarray, k: int) -> np.ndarray:
    """
    Aᵏ = P Dᵏ P⁻¹.

    Avantage : Dᵏ est trivial (élever chaque diag à la puissance k).
    Coût : O(n³) pour la diagonalisation + O(n) pour Dᵏ.
    Vs multiplication naïve : O(k·n³).
    """
    P, D, P_inv = diagonaliser(A)
    Dk = np.diag(np.diag(D)**k)
    return P @ Dk @ P_inv


def fibonacci_par_matrice(n: int) -> int:
    """
    Calcul de F_n par diagonalisation de la matrice de Fibonacci :
        [[1, 1], [1, 0]]^n = [[F_{n+1}, F_n], [F_n, F_{n-1}]].
    """
    A = np.array([[1, 1], [1, 0]], dtype=float)
    An = puissance_par_diag(A, n)
    return int(round(An[0, 1]))


def tracer_puissances(A: np.ndarray, k_max: int = 6, ax: plt.Axes | None = None) -> plt.Axes:
    """Montre l'effet de Aᵏ sur le cercle unité pour k = 0, 1, ..., k_max."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    theta = np.linspace(0, 2*np.pi, 100)
    cercle = np.array([np.cos(theta), np.sin(theta)])

    colors = plt.cm.viridis(np.linspace(0, 1, k_max + 1))
    for k in range(k_max + 1):
        Ak = puissance_par_diag(A, k)
        image = Ak @ cercle
        ax.plot(image[0], image[1], color=colors[k], linewidth=1.5,
                label=f"$A^{k}$", alpha=0.7)

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8); ax.set_title("$A^k$ appliqué au cercle unité")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    return ax


if __name__ == "__main__":
    print("=== Diagonalisation ===")
    A = np.array([[4, 1], [2, 3]], dtype=float)
    P, D, P_inv = diagonaliser(A)
    print(f"A =\n{A}")
    print(f"P (vecteurs propres) =\n{np.round(P, 6)}")
    print(f"D (valeurs propres)  =\n{np.round(D, 6)}")
    print(f"||PDP⁻¹ - A||_∞ = {np.linalg.norm(P @ D @ P_inv - A, np.inf):.2e}")

    print(f"\n=== Puissances de A ===")
    for k in [2, 5, 10]:
        Ak = puissance_par_diag(A, k)
        Ak_naive = np.linalg.matrix_power(A, k)
        print(f"  A^{k:>2} : ||diag - naive||_∞ = {np.linalg.norm(Ak - Ak_naive, np.inf):.2e}")

    print(f"\n=== Fibonacci par diagonalisation ===")
    for n in [5, 10, 20, 30]:
        print(f"  F_{n} = {fibonacci_par_matrice(n)}")

    print(f"\n=== Matrice non diagonalisable ===")
    J = np.array([[2, 1], [0, 2]], dtype=float)
    try:
        diagonaliser(J)
    except np.linalg.LinAlgError as e:
        print(f"  A = {J.tolist()} : {e} ✓")

    print(f"\n=== Matrices de contraction / expansion ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Contraction (|λ| < 1)
    C = np.array([[0.8, 0.1], [0.1, 0.6]])
    tracer_puissances(C, k_max=10, ax=axes[0])
    axes[0].set_title("Contraction (|λ| < 1) → spirale vers 0")
    # Expansion (|λ| > 1)
    E = np.array([[1.2, 0.3], [0.1, 1.1]])
    tracer_puissances(E, k_max=5, ax=axes[1])
    axes[1].set_title("Expansion (|λ| > 1) → spirale vers ∞")
    plt.tight_layout()
    plt.savefig("diagonalization_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
