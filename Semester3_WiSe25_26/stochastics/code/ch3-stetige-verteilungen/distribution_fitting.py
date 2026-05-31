"""
distribution_fitting.py
=======================

Ajustement de distributions et tests d'adéquation.

Couvre :
    - QQ-plot : comparaison graphique avec une loi théorique
    - Test de Kolmogorov-Smirnov (KS)
    - Test du χ² d'adéquation (Pearson)
    - Maximum de vraisemblance (MLE) pour la normale
    - Comparaison de plusieurs ajustements

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# ======================================================================
#  1. QQ-Plot
# ======================================================================

def qq_data(data: np.ndarray, distribution: str = "norm", **params) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les quantiles théoriques vs empiriques pour un QQ-plot.
    """
    n = len(data)
    data_sorted = np.sort(data)
    # Quantiles théoriques
    p = (np.arange(1, n+1) - 0.5) / n
    dist = getattr(stats, distribution)
    theo = dist.ppf(p, **params)
    return theo, data_sorted


def tracer_qq(data: np.ndarray, distribution: str = "norm",
               nom: str = "", ax: plt.Axes | None = None, **params) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    theo, emp = qq_data(data, distribution, **params)
    ax.scatter(theo, emp, s=10, alpha=0.5, color="steelblue")
    lims = [min(theo.min(), emp.min()), max(theo.max(), emp.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="$y = x$")
    ax.set_xlabel("quantiles théoriques"); ax.set_ylabel("quantiles observés")
    ax.set_title(f"QQ-plot {nom}")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    return ax


# ======================================================================
#  2. Test de Kolmogorov-Smirnov
# ======================================================================

def test_ks(data: np.ndarray, distribution: str = "norm", **params) -> dict:
    """
    H₀ : les données suivent la distribution spécifiée.
    Rejeter si p < α (typiquement 0.05).
    """
    stat, p = stats.kstest(data, distribution, args=tuple(params.values()))
    return {"statistic": stat, "p_value": p, "rejet_5%": p < 0.05}


# ======================================================================
#  3. Test du χ²
# ======================================================================

def test_chi2(data: np.ndarray, n_bins: int = 10,
               distribution: str = "norm", **params) -> dict:
    """
    Test du χ² d'adéquation (Pearson).
    Compare les fréquences observées aux fréquences attendues.
    """
    dist = getattr(stats, distribution)

    # Bins par quantiles pour des effectifs attendus égaux
    edges = dist.ppf(np.linspace(0, 1, n_bins + 1), **params)
    edges[0] = -np.inf
    edges[-1] = np.inf

    observed, _ = np.histogram(data, bins=edges)
    expected = np.full(n_bins, len(data) / n_bins)

    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = n_bins - 1  # degrés de liberté
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    return {"chi2": chi2_stat, "df": df, "p_value": p_value, "rejet_5%": p_value < 0.05}


# ======================================================================
#  4. Maximum de vraisemblance (MLE)
# ======================================================================

def mle_normale(data: np.ndarray) -> dict:
    """
    MLE pour N(μ, σ²) :
        μ̂ = x̄, σ̂² = (1/n) Σ(xᵢ - x̄)² (biaisé).
    """
    mu_hat = np.mean(data)
    sigma2_hat = np.mean((data - mu_hat)**2)
    sigma2_unbiased = np.var(data, ddof=1)
    return {
        "mu_hat": mu_hat,
        "sigma2_hat_mle": sigma2_hat,
        "sigma2_hat_unbiased": sigma2_unbiased,
        "sigma_hat": np.sqrt(sigma2_hat),
    }


def mle_exponentielle(data: np.ndarray) -> dict:
    """MLE pour Exp(λ) : λ̂ = 1/x̄."""
    lam_hat = 1 / np.mean(data)
    return {"lambda_hat": lam_hat, "mean": np.mean(data)}


# ======================================================================
#  5. Comparaison de distributions
# ======================================================================

def comparer_ajustements(data: np.ndarray) -> list[dict]:
    """Teste plusieurs distributions et renvoie les résultats triés par p-value."""
    results = []
    for nom, dist_name in [("Normale", "norm"), ("Exponentielle", "expon"),
                             ("Log-normale", "lognorm"), ("Uniforme", "uniform")]:
        try:
            params = getattr(stats, dist_name).fit(data)
            ks = stats.kstest(data, dist_name, args=params)
            results.append({"nom": nom, "params": params, "KS": ks.statistic, "p": ks.pvalue})
        except Exception:
            pass
    return sorted(results, key=lambda r: -r["p"])


# ======================================================================
#  6. Tracés
# ======================================================================

def tracer_ajustement(data: np.ndarray, nom: str = "",
                       ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.hist(data, bins=40, density=True, alpha=0.5, color="steelblue",
            edgecolor="white", label="données")

    x = np.linspace(data.min(), data.max(), 300)

    # Ajustement normal
    mu, sigma = np.mean(data), np.std(data)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2,
            label=f"$N({mu:.2f}, {sigma:.2f}^2)$")

    ax.set_xlabel("$x$"); ax.set_ylabel("densité")
    ax.set_title(f"Ajustement {nom}")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_ecdf_vs_cdf(data: np.ndarray, distribution: str = "norm",
                         ax: plt.Axes | None = None, **params) -> plt.Axes:
    """ECDF empirique vs CDF théorique (pour visualiser le test KS)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    data_sorted = np.sort(data)
    ecdf = np.arange(1, len(data)+1) / len(data)
    dist = getattr(stats, distribution)
    cdf_theo = dist.cdf(data_sorted, **params)

    ax.step(data_sorted, ecdf, "b-", linewidth=2, label="ECDF (empirique)")
    ax.plot(data_sorted, cdf_theo, "r--", linewidth=2, label="CDF (théorique)")

    # Distance KS
    D = np.max(np.abs(ecdf - cdf_theo))
    idx = np.argmax(np.abs(ecdf - cdf_theo))
    ax.vlines(data_sorted[idx], cdf_theo[idx], ecdf[idx], colors="green",
              linewidth=2, label=f"$D_n = {D:.4f}$")

    ax.set_xlabel("$x$"); ax.set_ylabel("probabilité cumulée")
    ax.set_title("Test KS : ECDF vs CDF")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== MLE pour la normale ===\n")
    data = rng.normal(5, 2, 500)
    mle = mle_normale(data)
    print(f"  Vrais paramètres : μ = 5, σ = 2")
    print(f"  μ̂ = {mle['mu_hat']:.4f}")
    print(f"  σ̂ (MLE) = {mle['sigma_hat']:.4f}")
    print(f"  σ̂² (non biaisé) = {mle['sigma2_hat_unbiased']:.4f}")

    print(f"\n=== Test KS ===\n")
    ks = test_ks(data, "norm", loc=mle["mu_hat"], scale=mle["sigma_hat"])
    print(f"  H₀ : données ~ N({mle['mu_hat']:.2f}, {mle['sigma_hat']:.2f}²)")
    print(f"  KS stat = {ks['statistic']:.4f}, p = {ks['p_value']:.4f}")
    print(f"  Rejet à 5% ? {ks['rejet_5%']}")

    print(f"\n=== Test χ² ===\n")
    chi2 = test_chi2(data, 10, "norm", loc=mle["mu_hat"], scale=mle["sigma_hat"])
    print(f"  χ² = {chi2['chi2']:.2f}, df = {chi2['df']}, p = {chi2['p_value']:.4f}")

    print(f"\n=== Comparaison des ajustements ===\n")
    data_exp = rng.exponential(2, 300)
    results = comparer_ajustements(data_exp)
    print(f"  Données : Exp(0.5) (vrai)")
    for r in results:
        print(f"  {r['nom']:>15} : KS = {r['KS']:.4f}, p = {r['p']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_ajustement(data, "N(5, 4)", ax=axes[0, 0])
    tracer_qq(data, nom="N(5, 2²)", ax=axes[0, 1])
    tracer_ecdf_vs_cdf(data, "norm", ax=axes[1, 0],
                        loc=mle["mu_hat"], scale=mle["sigma_hat"])
    tracer_qq(data_exp, "expon", nom="Exp (données exp.)", ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("distribution_fitting_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
