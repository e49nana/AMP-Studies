"""
estimation.py
=============

Estimation ponctuelle et par intervalle.

Couvre :
    - Estimateur : biais, convergence, efficacité
    - MLE (maximum de vraisemblance) pour normale, Poisson, Bernoulli
    - Méthode des moments
    - Intervalles de confiance : Z (σ connu), t (σ inconnu)
    - Taille d'échantillon nécessaire

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# ======================================================================
#  1. MLE (Maximum de vraisemblance)
# ======================================================================

def mle_bernoulli(data: np.ndarray) -> dict:
    """p̂ = x̄ (proportion de succès)."""
    p_hat = np.mean(data)
    return {"p_hat": p_hat, "se": np.sqrt(p_hat*(1-p_hat)/len(data))}


def mle_poisson(data: np.ndarray) -> dict:
    """λ̂ = x̄."""
    lam_hat = np.mean(data)
    return {"lambda_hat": lam_hat, "se": np.sqrt(lam_hat/len(data))}


def mle_normale(data: np.ndarray) -> dict:
    """μ̂ = x̄, σ̂² = (1/n)Σ(xᵢ-x̄)²."""
    mu = np.mean(data)
    sigma2_mle = np.mean((data - mu)**2)
    sigma2_unb = np.var(data, ddof=1)
    return {
        "mu_hat": mu, "sigma2_mle": sigma2_mle,
        "sigma2_unbiased": sigma2_unb,
        "se_mu": np.sqrt(sigma2_unb / len(data)),
    }


def mle_exponentielle(data: np.ndarray) -> dict:
    """λ̂ = 1/x̄."""
    lam = 1 / np.mean(data)
    return {"lambda_hat": lam, "se": lam / np.sqrt(len(data))}


# ======================================================================
#  2. Méthode des moments
# ======================================================================

def moments_methode(data: np.ndarray) -> dict:
    """Estime μ et σ² par les moments empiriques."""
    m1 = np.mean(data)         # = μ
    m2 = np.mean(data**2)      # = σ² + μ²
    return {"mu_hat": m1, "sigma2_hat": m2 - m1**2}


# ======================================================================
#  3. Intervalles de confiance
# ======================================================================

def ic_z(data: np.ndarray, sigma: float, alpha: float = 0.05) -> tuple[float, float]:
    """IC pour μ (σ connu) : x̄ ± z_{α/2} · σ/√n."""
    x_bar = np.mean(data)
    z = stats.norm.ppf(1 - alpha/2)
    marge = z * sigma / np.sqrt(len(data))
    return (x_bar - marge, x_bar + marge)


def ic_t(data: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """IC pour μ (σ inconnu) : x̄ ± t_{α/2, n-1} · s/√n."""
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    n = len(data)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    marge = t_crit * s / np.sqrt(n)
    return (x_bar - marge, x_bar + marge)


def ic_proportion(data: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """IC pour p (Wald) : p̂ ± z_{α/2} · √(p̂(1-p̂)/n)."""
    p_hat = np.mean(data)
    n = len(data)
    z = stats.norm.ppf(1 - alpha/2)
    marge = z * np.sqrt(p_hat * (1 - p_hat) / n)
    return (p_hat - marge, p_hat + marge)


def taille_echantillon_mu(sigma: float, marge: float, alpha: float = 0.05) -> int:
    """n nécessaire : n ≥ (z_{α/2} · σ / marge)²."""
    z = stats.norm.ppf(1 - alpha/2)
    return int(np.ceil((z * sigma / marge)**2))


def taille_echantillon_proportion(marge: float, alpha: float = 0.05) -> int:
    """n nécessaire (pire cas p=0.5) : n ≥ (z_{α/2})² / (4·marge²)."""
    z = stats.norm.ppf(1 - alpha/2)
    return int(np.ceil(z**2 / (4 * marge**2)))


# ======================================================================
#  4. Biais et convergence
# ======================================================================

def demo_biais_variance(n_sim: int = 10000, seed: int = 42) -> dict:
    """Montre que σ̂²_MLE est biaisé et s² est non biaisé."""
    rng = np.random.default_rng(seed)
    mu_vrai, sigma_vrai = 0, 1
    n = 10

    sigma2_mle = []
    sigma2_unb = []
    for _ in range(n_sim):
        sample = rng.normal(mu_vrai, sigma_vrai, n)
        m = np.mean(sample)
        sigma2_mle.append(np.mean((sample - m)**2))
        sigma2_unb.append(np.var(sample, ddof=1))

    return {
        "E[σ²_MLE]": np.mean(sigma2_mle),
        "E[s²]": np.mean(sigma2_unb),
        "vrai σ²": sigma_vrai**2,
        "biais MLE": np.mean(sigma2_mle) - sigma_vrai**2,
        "biais s²": np.mean(sigma2_unb) - sigma_vrai**2,
    }


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_ic_couverture(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre le taux de couverture des IC."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(42)
    mu_vrai = 50
    n_ic = 50
    contient = 0

    for i in range(n_ic):
        data = rng.normal(mu_vrai, 10, 25)
        lo, hi = ic_t(data, 0.05)
        ok = lo <= mu_vrai <= hi
        contient += ok
        color = "blue" if ok else "red"
        ax.plot([lo, hi], [i, i], color=color, linewidth=2, alpha=0.6)

    ax.axvline(mu_vrai, color="green", linewidth=2, linestyle="--",
                label=f"$\\mu = {mu_vrai}$ ({contient}/{n_ic} = {contient/n_ic*100:.0f}%)")
    ax.set_xlabel("valeur"); ax.set_ylabel("échantillon #")
    ax.set_title("Couverture des IC à 95% (Student)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="x")
    return ax


def tracer_taille_echantillon(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    marges = np.linspace(0.5, 5, 100)
    for sigma in [5, 10, 15]:
        ns = [taille_echantillon_mu(sigma, m) for m in marges]
        ax.plot(marges, ns, linewidth=2, label=f"$\\sigma = {sigma}$")

    ax.set_xlabel("marge d'erreur souhaitée"); ax.set_ylabel("$n$ nécessaire")
    ax.set_title("Taille d'échantillon vs précision (IC 95%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    return ax


def tracer_convergence_estimateur(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(42)
    mu_vrai = 5
    ns = np.arange(5, 500)

    data_full = rng.normal(mu_vrai, 2, 500)
    moyennes = [np.mean(data_full[:n]) for n in ns]
    ic_lo = [np.mean(data_full[:n]) - 1.96*2/np.sqrt(n) for n in ns]
    ic_hi = [np.mean(data_full[:n]) + 1.96*2/np.sqrt(n) for n in ns]

    ax.plot(ns, moyennes, "b-", linewidth=1.5, label="$\\bar{X}_n$")
    ax.fill_between(ns, ic_lo, ic_hi, alpha=0.2, color="blue", label="IC 95%")
    ax.axhline(mu_vrai, color="red", linewidth=2, linestyle="--", label=f"$\\mu = {mu_vrai}$")
    ax.set_xlabel("$n$"); ax.set_ylabel("$\\bar{X}_n$")
    ax.set_title("Convergence de l'estimateur $\\bar{X}_n \\to \\mu$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== MLE ===\n")
    data_n = rng.normal(10, 3, 100)
    r = mle_normale(data_n)
    print(f"  Normale (vrai μ=10, σ=3) :")
    print(f"    μ̂ = {r['mu_hat']:.4f} ± {r['se_mu']:.4f}")
    print(f"    σ̂² (MLE) = {r['sigma2_mle']:.4f}, s² = {r['sigma2_unbiased']:.4f}")

    data_p = rng.poisson(7, 200)
    r = mle_poisson(data_p)
    print(f"  Poisson (vrai λ=7) : λ̂ = {r['lambda_hat']:.4f} ± {r['se']:.4f}")

    print(f"\n=== Biais ===\n")
    b = demo_biais_variance()
    print(f"  σ² vrai = {b['vrai σ²']}")
    print(f"  E[σ̂²_MLE] = {b['E[σ²_MLE]']:.4f} (biais = {b['biais MLE']:.4f})")
    print(f"  E[s²]      = {b['E[s²]']:.4f} (biais = {b['biais s²']:.6f} ≈ 0 ✓)")

    print(f"\n=== Intervalles de confiance ===\n")
    data = rng.normal(50, 10, 36)
    lo_z, hi_z = ic_z(data, 10, 0.05)
    lo_t, hi_t = ic_t(data, 0.05)
    print(f"  n=36, σ connu=10  : IC_Z  = [{lo_z:.2f}, {hi_z:.2f}]")
    print(f"  n=36, σ inconnu   : IC_t  = [{lo_t:.2f}, {hi_t:.2f}]")
    print(f"  → IC_t légèrement plus large (incertitude sur σ)")

    print(f"\n=== Taille d'échantillon ===\n")
    for marge in [1, 2, 5]:
        n = taille_echantillon_mu(10, marge)
        print(f"  σ=10, marge={marge} : n ≥ {n}")
    n_prop = taille_echantillon_proportion(0.03)
    print(f"  Proportion, marge=3% : n ≥ {n_prop}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_ic_couverture(ax=axes[0])
    tracer_taille_echantillon(ax=axes[1])
    tracer_convergence_estimateur(ax=axes[2])
    plt.tight_layout()
    plt.savefig("estimation_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
