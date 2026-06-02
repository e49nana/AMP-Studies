"""
hypothesis_testing.py
=====================

Tests d'hypothèses classiques.

Couvre :
    - Logique des tests : H₀, H₁, α, p-value
    - Test Z (σ connu) et test t (σ inconnu)
    - Test t pour deux échantillons (indépendants et appariés)
    - Test du χ² d'indépendance
    - Puissance d'un test et erreurs de type I/II
    - Interprétation correcte de la p-value

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


@dataclass
class TestResult:
    nom: str
    statistique: float
    p_value: float
    alpha: float
    rejet: bool
    conclusion: str


# ======================================================================
#  1. Test Z (σ connu)
# ======================================================================

def test_z(data: np.ndarray, mu0: float, sigma: float,
            alternative: str = "two-sided", alpha: float = 0.05) -> TestResult:
    """
    H₀ : μ = μ₀.
    Z = (x̄ - μ₀) / (σ/√n).
    """
    n = len(data)
    x_bar = np.mean(data)
    z = (x_bar - mu0) / (sigma / np.sqrt(n))

    if alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p = 1 - stats.norm.cdf(z)
    else:
        p = stats.norm.cdf(z)

    rejet = p < alpha
    conclusion = f"Rejet H₀ (p = {p:.4f} < {alpha})" if rejet else f"Non-rejet H₀ (p = {p:.4f} ≥ {alpha})"
    return TestResult("Test Z", z, p, alpha, rejet, conclusion)


# ======================================================================
#  2. Test t (σ inconnu)
# ======================================================================

def test_t_1echantillon(data: np.ndarray, mu0: float,
                          alternative: str = "two-sided", alpha: float = 0.05) -> TestResult:
    """
    H₀ : μ = μ₀.
    t = (x̄ - μ₀) / (s/√n), df = n-1.
    """
    n = len(data)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    t_stat = (x_bar - mu0) / (s / np.sqrt(n))
    df = n - 1

    if alternative == "two-sided":
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    elif alternative == "greater":
        p = 1 - stats.t.cdf(t_stat, df)
    else:
        p = stats.t.cdf(t_stat, df)

    rejet = p < alpha
    conclusion = f"Rejet H₀ (p = {p:.4f})" if rejet else f"Non-rejet H₀ (p = {p:.4f})"
    return TestResult("Test t (1 éch.)", t_stat, p, alpha, rejet, conclusion)


def test_t_2echantillons(data1: np.ndarray, data2: np.ndarray,
                           alpha: float = 0.05) -> TestResult:
    """
    H₀ : μ₁ = μ₂ (Welch's t-test, variances inégales).
    """
    t_stat, p = stats.ttest_ind(data1, data2, equal_var=False)
    rejet = p < alpha
    conclusion = f"Rejet H₀ (p = {p:.4f})" if rejet else f"Non-rejet H₀ (p = {p:.4f})"
    return TestResult("Welch t-test", float(t_stat), float(p), alpha, rejet, conclusion)


def test_t_apparie(data1: np.ndarray, data2: np.ndarray,
                     alpha: float = 0.05) -> TestResult:
    """H₀ : μ_d = 0 (différences appariées)."""
    t_stat, p = stats.ttest_rel(data1, data2)
    rejet = p < alpha
    conclusion = f"Rejet H₀ (p = {p:.4f})" if rejet else f"Non-rejet H₀ (p = {p:.4f})"
    return TestResult("t-test apparié", float(t_stat), float(p), alpha, rejet, conclusion)


# ======================================================================
#  3. Test du χ² d'indépendance
# ======================================================================

def test_chi2_independance(tableau: np.ndarray, alpha: float = 0.05) -> TestResult:
    """
    H₀ : les variables sont indépendantes.
    χ² = Σ (O - E)² / E.
    """
    chi2, p, dof, expected = stats.chi2_contingency(tableau)
    rejet = p < alpha
    conclusion = f"Rejet H₀ — dépendants (p = {p:.4f})" if rejet else f"Non-rejet H₀ — indépendants (p = {p:.4f})"
    return TestResult(f"χ² indép. (df={dof})", float(chi2), float(p), alpha, rejet, conclusion)


# ======================================================================
#  4. Puissance d'un test
# ======================================================================

def puissance_test_z(mu0: float, mu1: float, sigma: float, n: int,
                      alpha: float = 0.05) -> float:
    """
    Puissance = P(rejeter H₀ | μ = μ₁).
    = 1 - β = P(Z > z_α - (μ₁-μ₀)/(σ/√n)).
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    delta = (mu1 - mu0) / (sigma / np.sqrt(n))
    power = 1 - stats.norm.cdf(z_alpha - delta) + stats.norm.cdf(-z_alpha - delta)
    return power


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_test_z_visuel(data: np.ndarray, mu0: float, sigma: float,
                           ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = test_z(data, mu0, sigma)
    x = np.linspace(-4, 4, 300)
    y = stats.norm.pdf(x)
    ax.plot(x, y, "b-", linewidth=2, label="$N(0,1)$ sous $H_0$")
    ax.fill_between(x[x <= -1.96], y[x <= -1.96], alpha=0.3, color="red")
    ax.fill_between(x[x >= 1.96], y[x >= 1.96], alpha=0.3, color="red", label="zone de rejet (α=5%)")
    ax.axvline(r.statistique, color="green", linewidth=2, linestyle="--",
                label=f"$Z = {r.statistique:.2f}$ (p = {r.p_value:.4f})")
    ax.set_xlabel("$Z$"); ax.set_ylabel("densité")
    ax.set_title(f"Test Z : {r.conclusion}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


def tracer_puissance(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    mu0, sigma = 0, 1
    mu1_range = np.linspace(0, 2, 200)

    for n in [10, 30, 100, 300]:
        power = [puissance_test_z(mu0, mu1, sigma, n) for mu1 in mu1_range]
        ax.plot(mu1_range, power, linewidth=2, label=f"$n = {n}$")

    ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="puissance = 80%")
    ax.axhline(0.05, color="grey", linestyle=":", alpha=0.3, label="α = 5%")
    ax.set_xlabel("effet réel $\\mu_1 - \\mu_0$"); ax.set_ylabel("puissance")
    ax.set_title("Puissance du test Z vs taille d'effet")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_p_value_interpretation(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(42)
    # Simulation : 1000 tests sous H₀ (pas d'effet)
    p_values = []
    for _ in range(1000):
        data = rng.normal(0, 1, 30)
        r = test_t_1echantillon(data, 0)
        p_values.append(r.p_value)

    ax.hist(p_values, bins=20, density=True, alpha=0.7, color="steelblue",
            edgecolor="white")
    ax.axhline(1, color="red", linewidth=2, linestyle="--", label="U(0,1) attendue")
    ax.axvline(0.05, color="green", linewidth=2, linestyle=":", label="α = 0.05")
    faux_positifs = sum(1 for p in p_values if p < 0.05)
    ax.set_title(f"p-values sous H₀ : {faux_positifs}/1000 faux positifs ({faux_positifs/10:.1f}%)")
    ax.set_xlabel("p-value"); ax.set_ylabel("densité")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== Test Z (σ connu) ===\n")
    data = rng.normal(52, 10, 50)
    r = test_z(data, 50, 10)
    print(f"  H₀: μ = 50, σ = 10, n = 50")
    print(f"  x̄ = {np.mean(data):.2f}, Z = {r.statistique:.4f}, p = {r.p_value:.4f}")
    print(f"  → {r.conclusion}")

    print(f"\n=== Test t (σ inconnu) ===\n")
    r = test_t_1echantillon(data, 50)
    print(f"  t = {r.statistique:.4f}, p = {r.p_value:.4f}")
    print(f"  → {r.conclusion}")

    print(f"\n=== Test t — 2 échantillons ===\n")
    data1 = rng.normal(50, 10, 40)
    data2 = rng.normal(55, 10, 40)
    r = test_t_2echantillons(data1, data2)
    print(f"  x̄₁ = {np.mean(data1):.2f}, x̄₂ = {np.mean(data2):.2f}")
    print(f"  → {r.conclusion}")

    print(f"\n=== χ² d'indépendance ===\n")
    tableau = np.array([[30, 10], [15, 25]])
    r = test_chi2_independance(tableau)
    print(f"  Tableau : {tableau.tolist()}")
    print(f"  χ² = {r.statistique:.2f}, p = {r.p_value:.4f}")
    print(f"  → {r.conclusion}")

    print(f"\n=== Puissance ===\n")
    for n in [10, 30, 100]:
        pw = puissance_test_z(0, 0.5, 1, n)
        print(f"  n={n:>3}, effet=0.5σ : puissance = {pw:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_test_z_visuel(data, 50, 10, ax=axes[0])
    tracer_puissance(ax=axes[1])
    tracer_p_value_interpretation(ax=axes[2])
    plt.tight_layout()
    plt.savefig("hypothesis_testing_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
