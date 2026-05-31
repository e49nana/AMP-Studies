"""
normal_distribution.py
======================

La loi normale en profondeur.

Couvre :
    - Densité φ(x) et CDF Φ(x) from-scratch (série de Taylor + erf)
    - Standardisation Z = (X - μ) / σ
    - Table de la loi normale (calcul et lecture)
    - Intervalles de confiance symétriques
    - Somme de normales : X + Y ~ N(μ₁+μ₂, σ₁²+σ₂²)
    - Approximation normale de la binomiale (correction de continuité)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# ======================================================================
#  1. Densité et CDF from-scratch
# ======================================================================

def phi(x: float) -> float:
    """Densité de N(0,1) : φ(x) = (1/√2π) exp(-x²/2)."""
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)


def erf_scratch(x: float, n_terms: int = 50) -> float:
    """erf(x) = (2/√π) Σ (-1)^n x^{2n+1} / (n!(2n+1))."""
    s = 0.0
    for n in range(n_terms):
        s += (-1)**n * x**(2*n+1) / (factorial(n) * (2*n+1))
    return 2 / np.sqrt(np.pi) * s


def Phi_scratch(x: float) -> float:
    """CDF de N(0,1) : Φ(x) = (1 + erf(x/√2)) / 2."""
    return 0.5 * (1 + erf_scratch(x / np.sqrt(2)))


def Phi(x: float, mu: float = 0, sigma: float = 1) -> float:
    """CDF de N(μ, σ²)."""
    return Phi_scratch((x - mu) / sigma)


def quantile_normal(p: float, mu: float = 0, sigma: float = 1) -> float:
    """z_p tel que Φ(z_p) = p (par scipy)."""
    return float(stats.norm.ppf(p, mu, sigma))


# ======================================================================
#  2. Table de la loi normale
# ======================================================================

def table_normale(z_max: float = 3.5, step: float = 0.1) -> None:
    """Affiche une mini-table de Φ(z)."""
    print(f"  {'z':>4}", end="")
    for d in range(10):
        print(f"  .{d:02d}", end="")
    print()
    print("  " + "-" * 55)
    z = 0.0
    while z <= z_max + 0.01:
        print(f"  {z:>4.1f}", end="")
        for d in range(10):
            val = float(stats.norm.cdf(z + d*0.01))
            print(f" {val:.4f}" if val < 0.99995 else " 1.000", end="")
        print()
        z = round(z + step, 1)


# ======================================================================
#  3. Intervalles de confiance
# ======================================================================

def intervalle_confiance_mu(x_bar: float, sigma: float, n: int,
                             alpha: float = 0.05) -> tuple[float, float]:
    """
    IC bilatéral pour μ (σ connu) :
        x̄ ± z_{1-α/2} · σ/√n.
    """
    z = quantile_normal(1 - alpha/2)
    marge = z * sigma / np.sqrt(n)
    return (x_bar - marge, x_bar + marge)


# ======================================================================
#  4. Somme de normales
# ======================================================================

def somme_normales(mu1: float, sigma1: float, mu2: float, sigma2: float) -> dict:
    """X ~ N(μ₁,σ₁²), Y ~ N(μ₂,σ₂²) indép. → X+Y ~ N(μ₁+μ₂, σ₁²+σ₂²)."""
    return {"mu": mu1+mu2, "sigma": np.sqrt(sigma1**2 + sigma2**2)}


# ======================================================================
#  5. Approximation de la binomiale
# ======================================================================

def binomiale_approx_normale(k: int, n: int, p: float, correction: bool = True) -> float:
    """
    P(X ≤ k) pour X ~ B(n,p) ≈ Φ((k + 0.5 - np) / √(np(1-p))).
    Correction de continuité : k → k + 0.5.
    """
    mu = n * p
    sigma = np.sqrt(n * p * (1 - p))
    k_corr = k + 0.5 if correction else k
    return float(stats.norm.cdf((k_corr - mu) / sigma))


# ======================================================================
#  6. Tracés
# ======================================================================

def tracer_phi_Phi(ax1: plt.Axes | None = None, ax2: plt.Axes | None = None) -> None:
    if ax1 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(-4, 4, 500)
    ax1.plot(x, [phi(xi) for xi in x], "b-", linewidth=2)
    ax1.fill_between(x, [phi(xi) for xi in x], alpha=0.15)
    ax1.set_title("Densité $\\varphi(x)$"); ax1.set_xlabel("$x$")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, [Phi_scratch(xi) for xi in x], "r-", linewidth=2, label="from-scratch")
    ax2.plot(x, stats.norm.cdf(x), "k--", linewidth=1, alpha=0.5, label="scipy")
    ax2.set_title("CDF $\\Phi(x)$"); ax2.set_xlabel("$x$")
    ax2.legend(); ax2.grid(True, alpha=0.3)


def tracer_binomiale_approx(n: int = 50, p: float = 0.3,
                              ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    from math import comb
    k = np.arange(0, n+1)
    binom_exact = [comb(n, ki) * p**ki * (1-p)**(n-ki) for ki in k]

    mu, sigma = n*p, np.sqrt(n*p*(1-p))
    x = np.linspace(0, n, 300)
    normal_approx = stats.norm.pdf(x, mu, sigma)

    ax.bar(k, binom_exact, alpha=0.5, color="steelblue", label=f"$B({n}, {p})$")
    ax.plot(x, normal_approx, "r-", linewidth=2,
            label=f"$N({mu:.0f}, {sigma:.1f}^2)$")
    ax.set_xlabel("$k$"); ax.set_ylabel("probabilité / densité")
    ax.set_title(f"Approximation normale de $B({n}, {p})$")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    return ax


def tracer_ic(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(42)
    mu_vrai, sigma = 50, 10
    n = 25
    n_ic = 30

    for i in range(n_ic):
        echantillon = rng.normal(mu_vrai, sigma, n)
        x_bar = np.mean(echantillon)
        lo, hi = intervalle_confiance_mu(x_bar, sigma, n, 0.05)
        contient = lo <= mu_vrai <= hi
        color = "blue" if contient else "red"
        ax.plot([lo, hi], [i, i], color=color, linewidth=2, alpha=0.6)
        ax.plot(x_bar, i, "o", color=color, markersize=4)

    ax.axvline(mu_vrai, color="green", linewidth=2, linestyle="--", label=f"$\\mu = {mu_vrai}$")
    ax.set_xlabel("valeur"); ax.set_ylabel("échantillon #")
    ax.set_title(f"IC 95% ($n={n}$, $\\sigma={sigma}$) — rouge = ne contient pas $\\mu$")
    ax.legend(); ax.grid(True, alpha=0.3, axis="x")
    return ax


if __name__ == "__main__":
    print("=== CDF from-scratch vs scipy ===\n")
    for z in [-2, -1, 0, 1, 1.96, 2, 3]:
        mine = Phi_scratch(z)
        ref = float(stats.norm.cdf(z))
        print(f"  Φ({z:>5.2f}) = {mine:.8f} (scipy: {ref:.8f}, err = {abs(mine-ref):.2e})")

    print(f"\n=== Quantiles ===\n")
    for p in [0.025, 0.05, 0.5, 0.95, 0.975]:
        print(f"  z_{{{p}}} = {quantile_normal(p):.4f}")

    print(f"\n=== Intervalle de confiance ===\n")
    lo, hi = intervalle_confiance_mu(100, 15, 36, 0.05)
    print(f"  x̄=100, σ=15, n=36, α=0.05")
    print(f"  IC 95% = [{lo:.2f}, {hi:.2f}]")
    print(f"  Marge = ±{(hi-lo)/2:.2f}")

    print(f"\n=== Somme de normales ===\n")
    r = somme_normales(10, 3, 20, 4)
    print(f"  N(10,9) + N(20,16) = N({r['mu']}, {r['sigma']:.4f}²)")

    print(f"\n=== Approx. normale de B(100, 0.3) ===\n")
    from math import comb
    n, p = 100, 0.3
    for k in [25, 30, 35]:
        exact = sum(comb(n, i)*p**i*(1-p)**(n-i) for i in range(k+1))
        approx = binomiale_approx_normale(k, n, p, correction=True)
        no_corr = binomiale_approx_normale(k, n, p, correction=False)
        print(f"  P(X≤{k}) : exact={exact:.6f}, approx={approx:.6f} (sans corr: {no_corr:.6f})")

    print(f"\n=== Mini-table de Φ(z) ===\n")
    table_normale(2.0, 0.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_phi_Phi(ax1=axes[0, 0], ax2=axes[0, 1])
    tracer_binomiale_approx(50, 0.3, ax=axes[1, 0])
    tracer_ic(ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("normal_distribution_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
