"""
expected_value.py
=================

Espérance, variance et moments.

Couvre :
    - E[X] = Σ xᵢ P(X = xᵢ) (définition)
    - Linéarité : E[aX + b] = aE[X] + b
    - Var(X) = E[X²] - (E[X])² (formule de König)
    - Écart-type σ = √Var
    - Inégalité de Tchebychev : P(|X - μ| ≥ kσ) ≤ 1/k²
    - Inégalité de Markov : P(X ≥ a) ≤ E[X]/a

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def esperance(valeurs: np.ndarray, probabilites: np.ndarray) -> float:
    """E[X] = Σ xᵢ pᵢ."""
    return float(np.sum(valeurs * probabilites))


def moment(valeurs: np.ndarray, probabilites: np.ndarray, k: int) -> float:
    """E[X^k] = Σ xᵢ^k pᵢ."""
    return float(np.sum(valeurs**k * probabilites))


def variance(valeurs: np.ndarray, probabilites: np.ndarray) -> float:
    """Var(X) = E[X²] - (E[X])²."""
    EX = esperance(valeurs, probabilites)
    EX2 = moment(valeurs, probabilites, 2)
    return EX2 - EX**2


def ecart_type(valeurs: np.ndarray, probabilites: np.ndarray) -> float:
    """σ = √Var(X)."""
    return np.sqrt(variance(valeurs, probabilites))


def tchebychev(k: float) -> float:
    """P(|X - μ| ≥ kσ) ≤ 1/k²."""
    return 1 / k**2 if k > 0 else 1


def markov(EX: float, a: float) -> float:
    """P(X ≥ a) ≤ E[X]/a pour X ≥ 0."""
    return EX / a if a > 0 else 1


# ======================================================================
#  Exemples
# ======================================================================

def exemple_de():
    """E[X] et Var(X) pour un dé équilibré."""
    print("=== Dé équilibré ===\n")
    x = np.arange(1, 7, dtype=float)
    p = np.full(6, 1/6)
    E = esperance(x, p)
    V = variance(x, p)
    print(f"  E[X] = {E:.4f} (= 7/2 = {7/2})")
    print(f"  E[X²] = {moment(x, p, 2):.4f}")
    print(f"  Var(X) = {V:.4f} (= 35/12 = {35/12:.4f})")
    print(f"  σ = {np.sqrt(V):.4f}")


def exemple_linearite():
    """E[aX+b] = aE[X] + b, Var(aX+b) = a²Var(X)."""
    print("\n=== Linéarité ===\n")
    x = np.arange(1, 7, dtype=float)
    p = np.full(6, 1/6)
    a, b = 3, -2
    y = a * x + b
    print(f"  Y = {a}X + ({b})")
    print(f"  E[X] = {esperance(x, p):.4f}")
    print(f"  E[Y] = {esperance(y, p):.4f} = {a}·E[X] + ({b}) = {a*esperance(x, p)+b:.4f} ✓")
    print(f"  Var(X) = {variance(x, p):.4f}")
    print(f"  Var(Y) = {variance(y, p):.4f} = {a}²·Var(X) = {a**2*variance(x, p):.4f} ✓")


def exemple_tchebychev():
    """Vérification de l'inégalité de Tchebychev par simulation."""
    print("\n=== Inégalité de Tchebychev ===\n")
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, 100_000)  # μ=0, σ=1
    for k in [1, 2, 3]:
        borne = tchebychev(k)
        reel = np.mean(np.abs(X) >= k)
        print(f"  k={k} : P(|X| ≥ {k}σ) ≤ {borne:.4f} (réel: {reel:.4f})")
    print(f"  → La borne est très lâche mais universelle (toute distribution)")


# ======================================================================
#  Tracés
# ======================================================================

def tracer_esperance_visuelle(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Dé pipé
    x = np.arange(1, 7, dtype=float)
    p_equi = np.full(6, 1/6)
    p_pipe = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])  # 6 favori

    for prob, nom, style in [(p_equi, "équilibré", "steelblue"), (p_pipe, "pipé (6 favori)", "coral")]:
        E = esperance(x, prob)
        ax.bar(x + (0.15 if "pipé" in nom else -0.15), prob, width=0.3,
               color=style, alpha=0.7, label=f"{nom} ($E = {E:.2f}$)")
        ax.axvline(E, color=style, linestyle="--", linewidth=2, alpha=0.5)

    ax.set_xlabel("$x$"); ax.set_ylabel("$P(X = x)$")
    ax.set_title("Espérance = centre de gravité de la distribution")
    ax.set_xticks(range(1, 7)); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    return ax


def tracer_tchebychev(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ks = np.linspace(1, 5, 100)
    borne_tcheb = [1/k**2 for k in ks]

    # Valeurs réelles pour la normale
    from scipy import stats
    reel_normal = [2 * (1 - stats.norm.cdf(k)) for k in ks]

    ax.plot(ks, borne_tcheb, "r-", linewidth=2, label="Tchebychev : $1/k^2$")
    ax.plot(ks, reel_normal, "b-", linewidth=2, label="Normale (réel)")
    ax.set_xlabel("$k$"); ax.set_ylabel("$P(|X - \\mu| \\geq k\\sigma)$")
    ax.set_title("Tchebychev vs réalité (normale)")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1)
    return ax


def tracer_variance_interpretation(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(-5, 5, 300)
    for sigma in [0.5, 1, 2]:
        y = np.exp(-x**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        ax.plot(x, y, linewidth=2, label=f"$\\sigma = {sigma}$")

    ax.set_xlabel("$x$"); ax.set_ylabel("densité")
    ax.set_title("Variance = mesure de la dispersion")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    exemple_de()
    exemple_linearite()
    exemple_tchebychev()

    print(f"\n=== Inégalité de Markov ===\n")
    print(f"  Dé : E[X] = 3.5")
    for a in [4, 5, 6]:
        borne = markov(3.5, a)
        reel = sum(1/6 for x in range(a, 7))
        print(f"  P(X ≥ {a}) ≤ {borne:.4f} (réel: {reel:.4f})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_esperance_visuelle(ax=axes[0])
    tracer_tchebychev(ax=axes[1])
    tracer_variance_interpretation(ax=axes[2])
    plt.tight_layout()
    plt.savefig("expected_value_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
