"""
matrix_powers.py
================

Applications des puissances de matrices via diagonalisation.

Couvre :
    - Suite de Fibonacci : F_n par matrice 2×2
    - Chaînes de Markov : convergence vers l'état stationnaire
    - Systèmes dynamiques linéaires xₖ₊₁ = Axₖ
    - Stabilité : |λ| < 1 → convergence, |λ| > 1 → divergence

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Fibonacci
# ======================================================================

def fibonacci_matrice(n: int) -> int:
    """F_n via [[1,1],[1,0]]^n."""
    A = np.array([[1, 1], [1, 0]], dtype=float)
    An = np.linalg.matrix_power(A, n)
    return int(round(An[0, 1]))


def fibonacci_diag(n: int) -> float:
    """F_n via la formule de Binet (diagonalisation explicite)."""
    phi = (1 + np.sqrt(5)) / 2
    psi = (1 - np.sqrt(5)) / 2
    return (phi**n - psi**n) / np.sqrt(5)


# ======================================================================
#  2. Chaîne de Markov
# ======================================================================

def markov_iterer(P: np.ndarray, x0: np.ndarray, n_steps: int) -> np.ndarray:
    """Itère xₖ₊₁ = P x_k pendant n_steps."""
    trajectory = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        x = P @ x
        trajectory.append(x.copy())
    return np.array(trajectory)


def etat_stationnaire(P: np.ndarray) -> np.ndarray:
    """
    Vecteur stationnaire π : Pπ = π, Σπᵢ = 1.
    C'est le vecteur propre de valeur propre 1, normalisé.
    """
    eigvals, eigvecs = np.linalg.eig(P)
    # Trouver λ = 1
    idx = np.argmin(np.abs(eigvals - 1))
    pi = eigvecs[:, idx].real
    return pi / np.sum(pi)


# ======================================================================
#  3. Système dynamique linéaire
# ======================================================================

def simuler_dynamique(A: np.ndarray, x0: np.ndarray, n_steps: int) -> np.ndarray:
    """Simule xₖ₊₁ = Axₖ."""
    trajectory = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        x = A @ x
        trajectory.append(x.copy())
    return np.array(trajectory)


def classifier_stabilite(A: np.ndarray) -> str:
    """Classifie la stabilité selon le rayon spectral."""
    rho = max(abs(np.linalg.eigvals(A)))
    if rho < 1 - 1e-10:
        return f"asymptotiquement stable (ρ = {rho:.4f} < 1)"
    elif rho > 1 + 1e-10:
        return f"instable (ρ = {rho:.4f} > 1)"
    else:
        return f"marginalement stable (ρ = {rho:.4f} ≈ 1)"


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_markov(P: np.ndarray, x0: np.ndarray, n_steps: int = 20,
                   ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    traj = markov_iterer(P, x0, n_steps)
    n_states = len(x0)
    for i in range(n_states):
        ax.plot(traj[:, i], "o-", markersize=4, label=f"état {i+1}")
    pi = etat_stationnaire(P)
    for i in range(n_states):
        ax.axhline(pi[i], color=f"C{i}", linestyle="--", alpha=0.5)
    ax.set_xlabel("itération $k$"); ax.set_ylabel("probabilité")
    ax.set_title("Convergence vers l'état stationnaire")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_dynamique_2d(A: np.ndarray, x0s: list[np.ndarray], n_steps: int = 30,
                         ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    for x0 in x0s:
        traj = simuler_dynamique(A, x0, n_steps)
        ax.plot(traj[:, 0], traj[:, 1], "o-", markersize=3, alpha=0.7)
        ax.plot(x0[0], x0[1], "ko", markersize=6)
    eigvals = np.linalg.eigvals(A)
    ax.set_title(f"xₖ₊₁ = Axₖ, λ = {np.round(eigvals, 3)}\n{classifier_stabilite(A)}")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    return ax


if __name__ == "__main__":
    print("=== Fibonacci ===")
    for n in [5, 10, 20, 30, 50]:
        fm = fibonacci_matrice(n)
        fb = fibonacci_diag(n)
        print(f"  F_{n:>2} = {fm:>15}  (Binet: {fb:.0f})")

    print(f"\n=== Chaîne de Markov ===")
    # Météo : Soleil ↔ Pluie
    P = np.array([[0.9, 0.5], [0.1, 0.5]])  # colonnes = probabilités de transition
    x0 = np.array([1.0, 0.0])  # commence au soleil
    pi = etat_stationnaire(P)
    print(f"  P = {P.tolist()}")
    print(f"  État stationnaire π = {np.round(pi, 4)}")
    print(f"  Vérif Pπ = π : {np.allclose(P @ pi, pi)} ✓")

    print(f"\n=== Stabilité de systèmes dynamiques ===")
    stable = np.array([[0.5, -0.3], [0.2, 0.4]])
    instable = np.array([[1.1, 0.2], [0.3, 1.2]])
    print(f"  A₁ : {classifier_stabilite(stable)}")
    print(f"  A₂ : {classifier_stabilite(instable)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_markov(P, x0, ax=axes[0])
    x0s = [np.array([c, s]) for c, s in zip(np.cos(np.linspace(0, 2*np.pi, 8, endpoint=False)),
                                              np.sin(np.linspace(0, 2*np.pi, 8, endpoint=False)))]
    tracer_dynamique_2d(stable, x0s, n_steps=15, ax=axes[1])
    tracer_dynamique_2d(instable, x0s, n_steps=8, ax=axes[2])
    plt.tight_layout()
    plt.savefig("matrix_powers_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
