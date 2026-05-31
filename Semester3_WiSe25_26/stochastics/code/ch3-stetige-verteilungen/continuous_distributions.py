"""
continuous_distributions.py
===========================

Distributions continues classiques.

Couvre :
    - Uniforme U(a,b) : f(x) = 1/(b-a)
    - Exponentielle Exp(λ) : f(x) = λe^{-λx}, sans mémoire
    - Normale N(μ,σ²) : f(x) = (1/σ√2π) exp(-(x-μ)²/(2σ²))
    - Densité, CDF, quantiles
    - Vérification ∫f = 1 et E[X], Var(X)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, stats
import matplotlib.pyplot as plt


# ======================================================================
#  1. Uniforme
# ======================================================================

def uniforme_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.where((x >= a) & (x <= b), 1/(b-a), 0)

def uniforme_cdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip((x - a) / (b - a), 0, 1)

def uniforme_moments(a: float, b: float) -> dict:
    return {"E": (a+b)/2, "Var": (b-a)**2/12}


# ======================================================================
#  2. Exponentielle
# ======================================================================

def exponentielle_pdf(x: np.ndarray, lam: float) -> np.ndarray:
    return np.where(x >= 0, lam * np.exp(-lam * x), 0)

def exponentielle_cdf(x: np.ndarray, lam: float) -> np.ndarray:
    return np.where(x >= 0, 1 - np.exp(-lam * x), 0)

def exponentielle_moments(lam: float) -> dict:
    return {"E": 1/lam, "Var": 1/lam**2}

def exponentielle_sans_memoire(lam: float, t: float, s: float) -> dict:
    """P(X > t+s | X > t) = P(X > s) (propriété sans mémoire)."""
    lhs = np.exp(-lam * s)  # P(X > s)
    rhs = np.exp(-lam * (t+s)) / np.exp(-lam * t)  # P(X > t+s) / P(X > t)
    return {"P(X>s)": lhs, "P(X>t+s|X>t)": rhs, "égaux": np.isclose(lhs, rhs)}


# ======================================================================
#  3. Normale
# ======================================================================

def normale_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-(x-mu)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def normale_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """CDF par scipy (pas de formule fermée)."""
    return stats.norm.cdf(x, mu, sigma)

def standardiser(x: float, mu: float, sigma: float) -> float:
    """Z = (X - μ) / σ."""
    return (x - mu) / sigma

def regle_empirique(mu: float, sigma: float) -> dict:
    """Règle 68-95-99.7."""
    return {
        "68%": (mu - sigma, mu + sigma),
        "95%": (mu - 2*sigma, mu + 2*sigma),
        "99.7%": (mu - 3*sigma, mu + 3*sigma),
    }


# ======================================================================
#  4. Vérification des propriétés
# ======================================================================

def verifier_densite(pdf, a: float, b: float, params: dict) -> dict:
    """Vérifie ∫f = 1 et calcule E[X], Var(X) numériquement."""
    integral, _ = integrate.quad(lambda x: pdf(np.array([x]), **params)[0], a, b)
    EX, _ = integrate.quad(lambda x: x * pdf(np.array([x]), **params)[0], a, b)
    EX2, _ = integrate.quad(lambda x: x**2 * pdf(np.array([x]), **params)[0], a, b)
    return {"∫f": integral, "E[X]": EX, "Var": EX2 - EX**2}


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_uniforme(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(-1, 6, 500)
    for a, b in [(0, 1), (1, 4), (2, 5)]:
        ax.plot(x, uniforme_pdf(x, a, b), linewidth=2, label=f"$U({a},{b})$")
    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax.set_title("Distribution uniforme"); ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_exponentielle(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0, 5, 300)
    for lam in [0.5, 1, 2, 5]:
        m = exponentielle_moments(lam)
        ax.plot(x, exponentielle_pdf(x, lam), linewidth=2,
                label=f"$\\lambda={lam}$ ($E={m['E']:.1f}$)")
    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax.set_title("Distribution exponentielle"); ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_normale(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(-6, 10, 500)
    for mu, sigma in [(0, 1), (0, 2), (2, 0.5), (3, 1.5)]:
        ax.plot(x, normale_pdf(x, mu, sigma), linewidth=2,
                label=f"$N({mu}, {sigma}^2)$")
    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax.set_title("Distribution normale"); ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_regle_empirique(mu: float = 0, sigma: float = 1,
                            ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    y = normale_pdf(x, mu, sigma)
    ax.plot(x, y, "b-", linewidth=2)

    colors = ["green", "orange", "red"]
    for (pct, (lo, hi)), c in zip(regle_empirique(mu, sigma).items(), colors):
        mask = (x >= lo) & (x <= hi)
        ax.fill_between(x[mask], y[mask], alpha=0.2, color=c, label=f"{pct}")

    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax.set_title("Règle 68-95-99.7")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Uniforme U(0, 1) ===\n")
    m = uniforme_moments(0, 1)
    v = verifier_densite(uniforme_pdf, -1, 2, {"a": 0, "b": 1})
    print(f"  E[X] = {m['E']}, Var = {m['Var']:.4f}")
    print(f"  ∫f = {v['∫f']:.6f} (= 1 ✓)")

    print(f"\n=== Exponentielle Exp(2) ===\n")
    lam = 2
    m = exponentielle_moments(lam)
    print(f"  E[X] = 1/λ = {m['E']}, Var = 1/λ² = {m['Var']}")
    sm = exponentielle_sans_memoire(lam, 3, 2)
    print(f"  Sans mémoire : P(X>2) = {sm['P(X>s)']:.6f}, "
          f"P(X>5|X>3) = {sm['P(X>t+s|X>t)']:.6f} ✓")

    print(f"\n=== Normale N(0, 1) ===\n")
    print(f"  P(|Z| < 1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f} ≈ 68.3%")
    print(f"  P(|Z| < 2) = {stats.norm.cdf(2) - stats.norm.cdf(-2):.4f} ≈ 95.4%")
    print(f"  P(|Z| < 3) = {stats.norm.cdf(3) - stats.norm.cdf(-3):.6f} ≈ 99.73%")

    print(f"\n  Standardisation : X ~ N(100, 15²)")
    print(f"  P(X > 130) = P(Z > {standardiser(130, 100, 15):.2f}) = "
          f"{1 - stats.norm.cdf(standardiser(130, 100, 15)):.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_uniforme(ax=axes[0, 0])
    tracer_exponentielle(ax=axes[0, 1])
    tracer_normale(ax=axes[1, 0])
    tracer_regle_empirique(ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("continuous_distributions_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
