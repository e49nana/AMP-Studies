"""
regression_stats.py
===================

Régression linéaire avec inférence statistique.

Couvre :
    - Moindres carrés : y = β₀ + β₁x
    - R² et R² ajusté
    - Erreur standard des coefficients
    - Test t pour β₁ (H₀: β₁ = 0, pas de relation linéaire)
    - Intervalles de confiance pour β₀, β₁
    - Bandes de prédiction et de confiance
    - Analyse des résidus (normalité, homoscédasticité)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


@dataclass
class RegressionResult:
    beta0: float         # intercept
    beta1: float         # pente
    R2: float
    R2_adj: float
    se_beta0: float      # erreur standard intercept
    se_beta1: float      # erreur standard pente
    t_beta1: float       # statistique t pour β₁
    p_beta1: float       # p-value pour H₀: β₁ = 0
    s: float             # erreur standard des résidus
    n: int
    residus: np.ndarray
    y_pred: np.ndarray


def regression_lineaire(x: np.ndarray, y: np.ndarray) -> RegressionResult:
    """
    Régression y = β₀ + β₁x par moindres carrés.
    Avec inférence statistique complète.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    Sxx = np.sum((x - x_bar)**2)
    Sxy = np.sum((x - x_bar) * (y - y_bar))
    Syy = np.sum((y - y_bar)**2)

    beta1 = Sxy / Sxx
    beta0 = y_bar - beta1 * x_bar

    y_pred = beta0 + beta1 * x
    residus = y - y_pred
    SSE = np.sum(residus**2)
    SSR = np.sum((y_pred - y_bar)**2)

    R2 = 1 - SSE / Syy if Syy > 0 else 0
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - 2)

    s = np.sqrt(SSE / (n - 2))  # erreur standard résidus
    se_beta1 = s / np.sqrt(Sxx)
    se_beta0 = s * np.sqrt(1/n + x_bar**2/Sxx)

    t_beta1 = beta1 / se_beta1 if se_beta1 > 0 else 0
    p_beta1 = 2 * (1 - stats.t.cdf(abs(t_beta1), n - 2))

    return RegressionResult(
        beta0=beta0, beta1=beta1, R2=R2, R2_adj=R2_adj,
        se_beta0=se_beta0, se_beta1=se_beta1,
        t_beta1=t_beta1, p_beta1=p_beta1,
        s=s, n=n, residus=residus, y_pred=y_pred,
    )


def ic_coefficients(r: RegressionResult, alpha: float = 0.05) -> dict:
    """IC pour β₀ et β₁."""
    t_crit = stats.t.ppf(1 - alpha/2, r.n - 2)
    return {
        "β₀": (r.beta0 - t_crit*r.se_beta0, r.beta0 + t_crit*r.se_beta0),
        "β₁": (r.beta1 - t_crit*r.se_beta1, r.beta1 + t_crit*r.se_beta1),
    }


def bande_confiance(
    x: np.ndarray, r: RegressionResult, x_data: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Bande de confiance pour E[Y|x]."""
    x_bar = np.mean(x_data)
    Sxx = np.sum((x_data - x_bar)**2)
    t_crit = stats.t.ppf(1 - alpha/2, r.n - 2)
    y_hat = r.beta0 + r.beta1 * x
    se = r.s * np.sqrt(1/r.n + (x - x_bar)**2 / Sxx)
    return y_hat - t_crit * se, y_hat + t_crit * se


def bande_prediction(
    x: np.ndarray, r: RegressionResult, x_data: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Bande de prédiction pour une nouvelle observation Y."""
    x_bar = np.mean(x_data)
    Sxx = np.sum((x_data - x_bar)**2)
    t_crit = stats.t.ppf(1 - alpha/2, r.n - 2)
    y_hat = r.beta0 + r.beta1 * x
    se = r.s * np.sqrt(1 + 1/r.n + (x - x_bar)**2 / Sxx)
    return y_hat - t_crit * se, y_hat + t_crit * se


def afficher_regression(r: RegressionResult) -> None:
    print(f"  ŷ = {r.beta0:.4f} + {r.beta1:.4f}·x")
    print(f"  R² = {r.R2:.4f}, R²_adj = {r.R2_adj:.4f}")
    print(f"  s (résidus) = {r.s:.4f}")
    print(f"  β₁ : t = {r.t_beta1:.4f}, p = {r.p_beta1:.6f} "
          f"{'→ significatif ✓' if r.p_beta1 < 0.05 else '→ non significatif'}")
    ic = ic_coefficients(r)
    print(f"  IC 95% β₀ : [{ic['β₀'][0]:.4f}, {ic['β₀'][1]:.4f}]")
    print(f"  IC 95% β₁ : [{ic['β₁'][0]:.4f}, {ic['β₁'][1]:.4f}]")


# ======================================================================
#  Tracés
# ======================================================================

def tracer_regression(x: np.ndarray, y: np.ndarray, nom: str = "",
                       ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = regression_lineaire(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = r.beta0 + r.beta1 * x_line

    # Bandes
    conf_lo, conf_hi = bande_confiance(x_line, r, x)
    pred_lo, pred_hi = bande_prediction(x_line, r, x)

    ax.scatter(x, y, s=20, alpha=0.6, color="steelblue", label="données")
    ax.plot(x_line, y_line, "r-", linewidth=2,
            label=f"ŷ = {r.beta0:.2f} + {r.beta1:.2f}x ($R^2={r.R2:.3f}$)")
    ax.fill_between(x_line, conf_lo, conf_hi, alpha=0.2, color="red", label="IC 95% (E[Y|x])")
    ax.fill_between(x_line, pred_lo, pred_hi, alpha=0.1, color="blue", label="prédiction 95%")

    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"Régression linéaire {nom}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


def tracer_residus(x: np.ndarray, y: np.ndarray,
                     axes: list | None = None) -> None:
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    r = regression_lineaire(x, y)

    # Résidus vs x
    axes[0].scatter(x, r.residus, s=15, alpha=0.6)
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("$x$"); axes[0].set_ylabel("résidus")
    axes[0].set_title("Résidus vs $x$"); axes[0].grid(True, alpha=0.3)

    # QQ-plot des résidus
    res_std = (r.residus - np.mean(r.residus)) / np.std(r.residus)
    stats.probplot(res_std, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-plot des résidus")

    # Histogramme des résidus
    axes[2].hist(r.residus, bins=20, density=True, alpha=0.7, color="steelblue",
                  edgecolor="white")
    x_norm = np.linspace(r.residus.min(), r.residus.max(), 100)
    axes[2].plot(x_norm, stats.norm.pdf(x_norm, 0, r.s), "r-", linewidth=2)
    axes[2].set_title("Distribution des résidus")
    axes[2].set_xlabel("résidu"); axes[2].grid(True, alpha=0.3)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== Régression linéaire ===\n")
    x = np.linspace(0, 10, 50)
    y = 3 + 2.5 * x + rng.normal(0, 2, 50)  # vrai : β₀=3, β₁=2.5, σ=2
    r = regression_lineaire(x, y)
    print(f"  Vrais paramètres : β₀=3, β₁=2.5, σ=2")
    afficher_regression(r)

    print(f"\n=== Pas de relation (β₁ = 0) ===\n")
    y_null = 5 + rng.normal(0, 3, 50)
    r_null = regression_lineaire(x, y_null)
    afficher_regression(r_null)

    print(f"\n=== Vérification vs scipy ===\n")
    slope, intercept, r_val, p_val, se = stats.linregress(x, y)
    print(f"  scipy : β₁={slope:.4f}, β₀={intercept:.4f}, R²={r_val**2:.4f}, p={p_val:.6f}")
    print(f"  mine  : β₁={r.beta1:.4f}, β₀={r.beta0:.4f}, R²={r.R2:.4f}, p={r.p_beta1:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_regression(x, y, "(signal)", ax=axes[0, 0])
    tracer_regression(x, y_null, "(bruit pur)", ax=axes[0, 1])
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    tracer_residus(x, y, axes2)
    fig2.tight_layout()
    fig2.savefig("residuals_analysis.png", dpi=120)

    axes[1, 0].scatter(x, r.residus, s=15, alpha=0.6)
    axes[1, 0].axhline(0, color="red"); axes[1, 0].set_title("Résidus")
    axes[1, 0].set_xlabel("$x$"); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(r.residus, bins=15, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    axes[1, 1].set_title("Distribution des résidus"); axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("regression_stats_demo.png", dpi=120)
    print("\nFigures sauvegardées.")
