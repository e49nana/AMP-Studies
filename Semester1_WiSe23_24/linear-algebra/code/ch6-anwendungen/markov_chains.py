"""
markov_chains.py
================

Chaînes de Markov et algèbre linéaire.

Couvre :
    - Matrice de transition stochastique (colonnes sommant à 1)
    - Distribution stationnaire π : Pπ = π comme problème de vecteur propre
    - Convergence Pⁿ → matrice de rang 1 (pour chaînes ergodiques)
    - Exemples : météo, marche aléatoire, PageRank simplifié
    - Temps de convergence lié à |λ₂| (second plus grand eigenvalue)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def est_stochastique(P: np.ndarray, tol: float = 1e-10) -> bool:
    """Vérifie que P est stochastique par colonnes (Σ_i P_ij = 1, P_ij ≥ 0)."""
    return (np.all(P >= -tol) and
            np.allclose(np.sum(P, axis=0), 1, atol=tol))


def distribution_stationnaire(P: np.ndarray) -> np.ndarray:
    """
    Vecteur stationnaire π : Pπ = π, Σπᵢ = 1.
    = vecteur propre de valeur propre 1, normalisé.
    """
    eigvals, eigvecs = np.linalg.eig(P)
    idx = np.argmin(np.abs(eigvals - 1))
    pi = eigvecs[:, idx].real
    return pi / np.sum(pi)


def iterer(P: np.ndarray, x0: np.ndarray, n: int) -> np.ndarray:
    """Calcule x₀, Px₀, P²x₀, ..., Pⁿx₀."""
    traj = [x0.copy()]
    x = x0.copy()
    for _ in range(n):
        x = P @ x
        traj.append(x.copy())
    return np.array(traj)


def vitesse_convergence(P: np.ndarray) -> float:
    """
    |λ₂| = second plus grand module de valeur propre.
    Plus |λ₂| est proche de 0, plus la convergence est rapide.
    """
    eigvals = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    if len(eigvals) < 2:
        return 0.0
    return float(eigvals[1])


# ======================================================================
#  Exemples
# ======================================================================

def exemple_meteo() -> tuple[np.ndarray, list[str]]:
    """
    Modèle météo simplifié :
        - Soleil → Soleil : 0.8,  Soleil → Pluie : 0.2
        - Pluie → Soleil : 0.4,  Pluie → Pluie : 0.6
    """
    P = np.array([[0.8, 0.4],
                   [0.2, 0.6]])
    return P, ["Soleil", "Pluie"]


def exemple_marche_aleatoire(n: int = 5) -> tuple[np.ndarray, list[str]]:
    """Marche aléatoire sur {0, 1, ..., n} avec bords réfléchissants."""
    P = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        if i == 0:
            P[1, 0] = 1.0  # rebond
        elif i == n:
            P[n - 1, n] = 1.0  # rebond
        else:
            P[i - 1, i] = 0.5
            P[i + 1, i] = 0.5
    return P, [str(i) for i in range(n + 1)]


def exemple_pagerank(n: int = 5, damping: float = 0.85) -> np.ndarray:
    """
    PageRank simplifié : P = d·H + (1-d)/n · 1·1ᵀ.
    H = matrice de liens (colonne j → probabilités de suivre un lien depuis j).
    """
    rng = np.random.default_rng(42)
    # Graphe aléatoire
    H = np.zeros((n, n))
    for j in range(n):
        liens = rng.choice(n, size=rng.integers(1, n), replace=False)
        liens = liens[liens != j]  # pas de self-loop
        if len(liens) > 0:
            H[liens, j] = 1.0 / len(liens)
        else:
            H[:, j] = 1.0 / n

    P = damping * H + (1 - damping) / n * np.ones((n, n))
    return P


# ======================================================================
#  Tracés
# ======================================================================

def tracer_convergence(P, x0, etats, n_steps=30, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    traj = iterer(P, x0, n_steps)
    pi = distribution_stationnaire(P)
    for i, nom in enumerate(etats):
        ax.plot(traj[:, i], "o-", markersize=3, label=nom)
        ax.axhline(pi[i], color=f"C{i}", linestyle="--", alpha=0.4)
    ax.set_xlabel("itération k"); ax.set_ylabel("probabilité")
    ax.set_title(f"Convergence (|λ₂| = {vitesse_convergence(P):.3f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_puissances_matrice(P, etats, ax=None):
    """Montre Pⁿ → matrice de rang 1."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    pi = distribution_stationnaire(P)
    n_states = len(etats)
    ks = [1, 5, 10, 50]
    erreurs = []
    for k in range(1, 51):
        Pk = np.linalg.matrix_power(P, k)
        rang1 = np.outer(pi, np.ones(n_states))
        erreurs.append(np.linalg.norm(Pk - rang1, np.inf))

    ax.semilogy(range(1, 51), erreurs, "b-", linewidth=2)
    ax.set_xlabel("k"); ax.set_ylabel("$\\|P^k - \\pi \\mathbf{1}^T\\|_\\infty$")
    ax.set_title("$P^k$ converge vers la matrice de rang 1")
    ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Météo ===")
    P, etats = exemple_meteo()
    pi = distribution_stationnaire(P)
    print(f"  P = {P.tolist()}")
    print(f"  Stochastique ? {est_stochastique(P)} ✓")
    print(f"  π = {np.round(pi, 4)} → {pi[0]*100:.0f}% soleil, {pi[1]*100:.0f}% pluie")
    print(f"  Vérif Pπ = π : {np.allclose(P @ pi, pi)} ✓")
    print(f"  |λ₂| = {vitesse_convergence(P):.3f}")

    print(f"\n=== Marche aléatoire (n=5) ===")
    P2, etats2 = exemple_marche_aleatoire(5)
    pi2 = distribution_stationnaire(P2)
    print(f"  π = {np.round(pi2, 4)}")
    print(f"  Uniforme ? {np.allclose(pi2, 1/6)} "
          f"({'oui ✓' if np.allclose(pi2, 1/6) else 'non'})")

    print(f"\n=== PageRank simplifié (5 pages) ===")
    P3 = exemple_pagerank(5)
    pi3 = distribution_stationnaire(P3)
    ranking = np.argsort(-pi3)
    print(f"  Ranking : {['page '+str(i) for i in ranking]}")
    print(f"  Scores  : {np.round(pi3[ranking], 4)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_convergence(P, np.array([1, 0], dtype=float), etats, ax=axes[0])
    axes[0].set_title("Météo")
    tracer_convergence(P2, np.eye(6)[0], etats2, n_steps=50, ax=axes[1])
    axes[1].set_title("Marche aléatoire")
    tracer_puissances_matrice(P, etats, ax=axes[2])
    plt.tight_layout()
    plt.savefig("markov_chains_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
