"""
generating_functions.py
=======================

Fonctions génératrices des probabilités.

Couvre :
    - FGP : G_X(s) = E[s^X] = Σ P(X=k) s^k
    - Propriétés : G(1) = 1, G'(1) = E[X], G''(1) = E[X(X-1)]
    - FGP des distributions classiques
    - Somme de v.a. indépendantes : G_{X+Y} = G_X · G_Y
    - Application : somme de Poisson = Poisson

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from math import factorial, comb, exp
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. FGP des distributions classiques
# ======================================================================

def fgp_bernoulli(s: float, p: float) -> float:
    """G(s) = (1-p) + ps = 1 - p + ps."""
    return 1 - p + p * s


def fgp_binomiale(s: float, n: int, p: float) -> float:
    """G(s) = (1 - p + ps)^n."""
    return (1 - p + p * s)**n


def fgp_poisson(s: float, lam: float) -> float:
    """G(s) = e^{λ(s-1)}."""
    return exp(lam * (s - 1))


def fgp_geometrique(s: float, p: float) -> float:
    """G(s) = ps / (1 - (1-p)s) pour la version k = 1, 2, ..."""
    return p * s / (1 - (1-p) * s) if abs(1 - (1-p)*s) > 1e-15 else float("inf")


def fgp_empirique(s: float, valeurs: np.ndarray, probs: np.ndarray) -> float:
    """G(s) = Σ pₖ s^{xₖ} (pour distribution quelconque)."""
    return float(np.sum(probs * s**valeurs))


# ======================================================================
#  2. Extraction des moments
# ======================================================================

def moments_depuis_fgp(G: Callable, h: float = 1e-5) -> dict:
    """
    E[X] = G'(1), Var(X) = G''(1) + G'(1) - (G'(1))².
    Dérivées numériques.
    """
    G1 = G(1)
    Gp1 = (G(1 + h) - G(1 - h)) / (2 * h)  # G'(1)
    Gpp1 = (G(1 + h) - 2*G(1) + G(1 - h)) / h**2  # G''(1)
    EX = Gp1
    VarX = Gpp1 + Gp1 - Gp1**2
    return {"G(1)": G1, "E[X]": EX, "Var(X)": VarX, "G'(1)": Gp1, "G''(1)": Gpp1}


# ======================================================================
#  3. Somme de v.a. indépendantes
# ======================================================================

def demo_somme_poisson():
    """X ~ Po(λ₁), Y ~ Po(λ₂) indép. → X+Y ~ Po(λ₁+λ₂)."""
    print("=== Somme de Poisson ===\n")
    lam1, lam2 = 3, 5

    print(f"  X ~ Po({lam1}), Y ~ Po({lam2})")
    print(f"  G_X(s) = e^{{{lam1}(s-1)}}, G_Y(s) = e^{{{lam2}(s-1)}}")
    print(f"  G_{'{X+Y}'}(s) = G_X · G_Y = e^{{{lam1+lam2}(s-1)}}")
    print(f"  → X + Y ~ Po({lam1 + lam2}) ✓")

    # Vérification
    s = 0.7
    G_produit = fgp_poisson(s, lam1) * fgp_poisson(s, lam2)
    G_somme = fgp_poisson(s, lam1 + lam2)
    print(f"\n  Vérification en s={s} :")
    print(f"    G_X · G_Y = {G_produit:.10f}")
    print(f"    G_{{X+Y}}  = {G_somme:.10f}")
    print(f"    Erreur = {abs(G_produit - G_somme):.2e} ✓")


def demo_somme_binomiale():
    """X ~ B(n₁,p), Y ~ B(n₂,p) indép. → X+Y ~ B(n₁+n₂, p)."""
    print("\n=== Somme de Binomiales (même p) ===\n")
    n1, n2, p = 10, 15, 0.3
    print(f"  X ~ B({n1},{p}), Y ~ B({n2},{p})")
    print(f"  G_X = (1-p+ps)^{n1}, G_Y = (1-p+ps)^{n2}")
    print(f"  G_{{X+Y}} = (1-p+ps)^{n1+n2}")
    print(f"  → X + Y ~ B({n1+n2}, {p}) ✓")


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_fgp(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    s = np.linspace(0, 1, 200)

    distributions = [
        ("Bernoulli(0.3)", lambda si: fgp_bernoulli(si, 0.3)),
        ("B(10, 0.3)", lambda si: fgp_binomiale(si, 10, 0.3)),
        ("Po(3)", lambda si: fgp_poisson(si, 3)),
        ("Géom(0.4)", lambda si: fgp_geometrique(si, 0.4)),
    ]

    for nom, G in distributions:
        y = [G(si) for si in s]
        ax.plot(s, y, linewidth=2, label=nom)

    ax.set_xlabel("$s$"); ax.set_ylabel("$G_X(s)$")
    ax.set_title("Fonctions génératrices $G_X(s) = E[s^X]$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.axvline(1, color="grey", linestyle=":", alpha=0.3)
    return ax


def tracer_somme_verification(ax: plt.Axes | None = None) -> plt.Axes:
    """Vérifie G_{X+Y} = G_X · G_Y par simulation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(42)
    lam1, lam2 = 3, 5
    n = 100_000

    X = rng.poisson(lam1, n)
    Y = rng.poisson(lam2, n)
    Z = X + Y

    k_max = 20
    k = np.arange(0, k_max)
    p_sim = np.array([np.mean(Z == ki) for ki in k])
    p_theo = np.array([exp(-(lam1+lam2)) * (lam1+lam2)**ki / factorial(ki) for ki in k])

    ax.bar(k - 0.15, p_sim, width=0.3, alpha=0.7, label="simulation $X + Y$")
    ax.bar(k + 0.15, p_theo, width=0.3, alpha=0.7, label=f"$Po({lam1+lam2})$ théorique")
    ax.set_xlabel("$k$"); ax.set_ylabel("$P(X+Y = k)$")
    ax.set_title(f"$Po({lam1}) + Po({lam2}) = Po({lam1+lam2})$ vérifié")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    return ax


if __name__ == "__main__":
    print("=== Moments par FGP ===\n")
    distributions = [
        ("Bernoulli(0.3)", lambda s: fgp_bernoulli(s, 0.3), 0.3, 0.3*0.7),
        ("B(10, 0.3)", lambda s: fgp_binomiale(s, 10, 0.3), 3, 2.1),
        ("Po(5)", lambda s: fgp_poisson(s, 5), 5, 5),
        ("Géom(0.4)", lambda s: fgp_geometrique(s, 0.4), 2.5, 3.75),
    ]
    print(f"  {'Distribution':>15} | {'E[X] (FGP)':>12} | {'E[X] (théo)':>12} | {'Var (FGP)':>10} | {'Var (théo)':>10}")
    print("  " + "-" * 70)
    for nom, G, E_theo, V_theo in distributions:
        m = moments_depuis_fgp(G)
        print(f"  {nom:>15} | {m['E[X]']:>12.4f} | {E_theo:>12.4f} | {m['Var(X)']:>10.4f} | {V_theo:>10.4f}")

    print()
    demo_somme_poisson()
    demo_somme_binomiale()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_fgp(ax=axes[0])
    tracer_somme_verification(ax=axes[1])
    plt.tight_layout()
    plt.savefig("generating_functions_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
