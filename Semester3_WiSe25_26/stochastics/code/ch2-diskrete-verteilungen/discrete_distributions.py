"""
discrete_distributions.py
==========================

Distributions discrètes classiques.

Couvre :
    - Bernoulli : X ∈ {0, 1}, P(X=1) = p
    - Binomiale : B(n,p), nombre de succès en n essais
    - Poisson : Po(λ), événements rares
    - Géométrique : nombre d'essais jusqu'au premier succès
    - Hypergéométrique : tirage sans remise
    - Comparaison Binomiale → Poisson (quand n grand, p petit)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from math import comb, factorial, exp

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Distributions from-scratch
# ======================================================================

def bernoulli_pmf(k: int, p: float) -> float:
    """P(X = k) pour k ∈ {0, 1}."""
    if k == 1:
        return p
    elif k == 0:
        return 1 - p
    return 0


def binomiale_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) = C(n,k) p^k (1-p)^{n-k}."""
    if k < 0 or k > n:
        return 0
    return comb(n, k) * p**k * (1-p)**(n-k)


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) = λ^k e^{-λ} / k!."""
    if k < 0:
        return 0
    return lam**k * exp(-lam) / factorial(k)


def geometrique_pmf(k: int, p: float) -> float:
    """P(X = k) = (1-p)^{k-1} p, k = 1, 2, 3, ..."""
    if k < 1:
        return 0
    return (1 - p)**(k - 1) * p


def hypergeometrique_pmf(k: int, N: int, K: int, n: int) -> float:
    """
    P(X = k) = C(K,k)·C(N-K,n-k) / C(N,n).
    N = population, K = succès dans la population, n = tirages.
    """
    if k < max(0, n-(N-K)) or k > min(n, K):
        return 0
    return comb(K, k) * comb(N-K, n-k) / comb(N, n)


# ======================================================================
#  2. Propriétés
# ======================================================================

def esperance_variance(distribution: str, **params) -> dict:
    """E[X] et Var(X) pour les distributions classiques."""
    if distribution == "bernoulli":
        p = params["p"]
        return {"E": p, "Var": p*(1-p)}
    elif distribution == "binomiale":
        n, p = params["n"], params["p"]
        return {"E": n*p, "Var": n*p*(1-p)}
    elif distribution == "poisson":
        lam = params["lam"]
        return {"E": lam, "Var": lam}
    elif distribution == "geometrique":
        p = params["p"]
        return {"E": 1/p, "Var": (1-p)/p**2}
    raise ValueError(f"Distribution inconnue : {distribution}")


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_binomiale(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    n = 20
    for p in [0.2, 0.5, 0.7]:
        k = np.arange(0, n+1)
        pmf = [binomiale_pmf(ki, n, p) for ki in k]
        ev = esperance_variance("binomiale", n=n, p=p)
        ax.bar(k + (p-0.5)*0.25, pmf, width=0.25, alpha=0.7,
               label=f"$B({n}, {p})$, $E={ev['E']:.0f}$")

    ax.set_xlabel("$k$"); ax.set_ylabel("$P(X = k)$")
    ax.set_title("Distribution binomiale")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    return ax


def tracer_poisson(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for lam in [1, 4, 10]:
        k_max = int(lam + 4*np.sqrt(lam)) + 1
        k = np.arange(0, k_max)
        pmf = [poisson_pmf(ki, lam) for ki in k]
        ax.bar(k + (lam/10 - 0.5)*0.3, pmf, width=0.3, alpha=0.7,
               label=f"$Po({lam})$")

    ax.set_xlabel("$k$"); ax.set_ylabel("$P(X = k)$")
    ax.set_title("Distribution de Poisson")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    return ax


def tracer_binomiale_vs_poisson(ax: plt.Axes | None = None) -> plt.Axes:
    """Approximation de Poisson : B(n,p) ≈ Po(np) quand n grand, p petit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    lam = 5
    for n, p in [(10, 0.5), (50, 0.1), (500, 0.01)]:
        k = np.arange(0, 15)
        binom = [binomiale_pmf(ki, n, p) for ki in k]
        ax.plot(k, binom, "o-", markersize=4, linewidth=1.5,
                label=f"$B({n}, {p})$")

    pois = [poisson_pmf(ki, lam) for ki in np.arange(0, 15)]
    ax.plot(np.arange(0, 15), pois, "ks--", markersize=6, linewidth=2,
            label=f"$Po({lam})$", alpha=0.7)

    ax.set_xlabel("$k$"); ax.set_ylabel("$P(X = k)$")
    ax.set_title("$B(n, \\lambda/n) \\to Po(\\lambda)$ quand $n \\to \\infty$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    return ax


def tracer_geometrique(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for p in [0.1, 0.3, 0.5, 0.8]:
        k = np.arange(1, 20)
        pmf = [geometrique_pmf(ki, p) for ki in k]
        ax.bar(k + (p-0.4)*0.2, pmf, width=0.2, alpha=0.6,
               label=f"$p = {p}$, $E = {1/p:.1f}$")

    ax.set_xlabel("$k$ (essai du premier succès)")
    ax.set_ylabel("$P(X = k)$")
    ax.set_title("Distribution géométrique (sans mémoire)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    return ax


if __name__ == "__main__":
    print("=== Binomiale B(10, 0.3) ===\n")
    n, p = 10, 0.3
    ev = esperance_variance("binomiale", n=n, p=p)
    print(f"  E[X] = np = {ev['E']}, Var(X) = np(1-p) = {ev['Var']}")
    print(f"  P(X=3) = {binomiale_pmf(3, n, p):.6f}")
    print(f"  P(X≤3) = {sum(binomiale_pmf(k, n, p) for k in range(4)):.6f}")

    print(f"\n=== Poisson Po(5) ===\n")
    lam = 5
    ev = esperance_variance("poisson", lam=lam)
    print(f"  E[X] = Var(X) = λ = {ev['E']}")
    print(f"  P(X=5) = {poisson_pmf(5, lam):.6f}")
    print(f"  P(X≤5) = {sum(poisson_pmf(k, lam) for k in range(6)):.6f}")

    print(f"\n=== Approximation Poisson ===\n")
    lam = 3
    for n in [10, 50, 100, 1000]:
        p = lam / n
        err = max(abs(binomiale_pmf(k, n, p) - poisson_pmf(k, lam)) for k in range(10))
        print(f"  B({n:>4}, {p:.4f}) vs Po({lam}) : max erreur = {err:.6f}")

    print(f"\n=== Géométrique (p=0.2) ===\n")
    p = 0.2
    ev = esperance_variance("geometrique", p=p)
    print(f"  E[X] = 1/p = {ev['E']}")
    print(f"  P(X=1) = {geometrique_pmf(1, p):.4f} (succès au 1er essai)")
    print(f"  P(X≤5) = {sum(geometrique_pmf(k, p) for k in range(1,6)):.4f}")
    print(f"  P(X>10) = {1 - sum(geometrique_pmf(k, p) for k in range(1,11)):.4f} = (1-p)^10 = {(1-p)**10:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_binomiale(ax=axes[0, 0])
    tracer_poisson(ax=axes[0, 1])
    tracer_binomiale_vs_poisson(ax=axes[1, 0])
    tracer_geometrique(ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("discrete_distributions_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
