"""
central_limit_theorem.py
========================

Théorème central limite (TCL / Zentraler Grenzwertsatz).

Couvre :
    - TCL : (X̄ - μ) / (σ/√n) → N(0,1) quelle que soit la distribution
    - Démonstration visuelle pour uniforme, exponentielle, Bernoulli
    - Vitesse de convergence selon la distribution
    - Application : pourquoi la normale est partout

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def moyennes_echantillon(
    distribution: str, n: int, n_samples: int = 50_000,
    seed: int = 42, **params,
) -> np.ndarray:
    """
    Génère n_samples moyennes de n observations iid.
    Standardise : Z = (X̄ - μ) / (σ/√n).
    """
    rng = np.random.default_rng(seed)

    if distribution == "uniforme":
        a, b = params.get("a", 0), params.get("b", 1)
        data = rng.uniform(a, b, (n_samples, n))
        mu, sigma = (a+b)/2, (b-a)/np.sqrt(12)
    elif distribution == "exponentielle":
        lam = params.get("lam", 1)
        data = rng.exponential(1/lam, (n_samples, n))
        mu, sigma = 1/lam, 1/lam
    elif distribution == "bernoulli":
        p = params.get("p", 0.5)
        data = rng.binomial(1, p, (n_samples, n)).astype(float)
        mu, sigma = p, np.sqrt(p*(1-p))
    elif distribution == "poisson":
        lam = params.get("lam", 3)
        data = rng.poisson(lam, (n_samples, n)).astype(float)
        mu, sigma = lam, np.sqrt(lam)
    elif distribution == "de":
        data = rng.integers(1, 7, (n_samples, n)).astype(float)
        mu, sigma = 3.5, np.sqrt(35/12)
    else:
        raise ValueError(f"Distribution inconnue : {distribution}")

    x_bar = data.mean(axis=1)
    z = (x_bar - mu) / (sigma / np.sqrt(n))
    return z


def test_normalite_ks(z: np.ndarray) -> dict:
    """Test de Kolmogorov-Smirnov : H₀ = z suit N(0,1)."""
    statistic, pvalue = stats.kstest(z, "norm")
    return {"KS_stat": statistic, "p_value": pvalue, "normal": pvalue > 0.05}


# ======================================================================
#  Tracés
# ======================================================================

def tracer_tcl_convergence(
    distribution: str, ns: tuple[int, ...] = (1, 2, 5, 30),
    ax: plt.Axes | None = None, **params,
) -> plt.Axes:
    """Montre la convergence vers N(0,1) pour n croissant."""
    if ax is None:
        fig, axes = plt.subplots(1, len(ns), figsize=(4*len(ns), 4))
    else:
        axes = [ax] * len(ns)

    x_norm = np.linspace(-4, 4, 200)
    y_norm = stats.norm.pdf(x_norm)

    for ax_i, n in zip(axes, ns):
        z = moyennes_echantillon(distribution, n, **params)
        ax_i.hist(z, bins=60, density=True, alpha=0.6, color="steelblue",
                   edgecolor="white", linewidth=0.3)
        ax_i.plot(x_norm, y_norm, "r-", linewidth=2)
        ks = test_normalite_ks(z)
        ax_i.set_title(f"$n = {n}$ (KS p={ks['p_value']:.3f})")
        ax_i.set_xlim(-4, 4); ax_i.set_ylim(0, 0.6)
        ax_i.grid(True, alpha=0.3)

    return axes[0]


def tracer_tcl_multi_distributions(n: int = 30,
                                     ax: plt.Axes | None = None) -> plt.Axes:
    """Compare le TCL pour différentes distributions initiales."""
    if ax is None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    else:
        axes = np.array([[ax]*3]*2)

    x_norm = np.linspace(-4, 4, 200)
    y_norm = stats.norm.pdf(x_norm)

    distributions = [
        ("uniforme", {}, "U(0,1)"),
        ("exponentielle", {"lam": 1}, "Exp(1)"),
        ("bernoulli", {"p": 0.3}, "Bernoulli(0.3)"),
        ("poisson", {"lam": 3}, "Po(3)"),
        ("de", {}, "Dé"),
        ("bernoulli", {"p": 0.01}, "Bernoulli(0.01)"),
    ]

    for ax_i, (dist, params, nom) in zip(axes.flat, distributions):
        z = moyennes_echantillon(dist, n, **params)
        ax_i.hist(z, bins=60, density=True, alpha=0.6, color="steelblue",
                   edgecolor="white", linewidth=0.3)
        ax_i.plot(x_norm, y_norm, "r-", linewidth=2)
        ax_i.set_title(f"{nom} ($n={n}$)")
        ax_i.set_xlim(-4, 4)
        ax_i.grid(True, alpha=0.3)

    return axes[0, 0]


def tracer_vitesse_convergence(ax: plt.Axes | None = None) -> plt.Axes:
    """KS statistique vs n pour différentes distributions."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ns = [2, 3, 5, 10, 20, 50, 100, 200]
    distributions = [
        ("uniforme", {}, "U(0,1)"),
        ("exponentielle", {"lam": 1}, "Exp(1)"),
        ("bernoulli", {"p": 0.5}, "Bern(0.5)"),
        ("bernoulli", {"p": 0.1}, "Bern(0.1)"),
    ]

    for dist, params, nom in distributions:
        ks_stats = []
        for n in ns:
            z = moyennes_echantillon(dist, n, n_samples=10000, **params)
            ks = test_normalite_ks(z)
            ks_stats.append(ks["KS_stat"])
        ax.loglog(ns, ks_stats, "o-", markersize=5, linewidth=2, label=nom)

    ax.axhline(0.01, color="green", linestyle="--", alpha=0.5, label="seuil pratique")
    ax.set_xlabel("$n$"); ax.set_ylabel("statistique KS")
    ax.set_title("Vitesse de convergence du TCL")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== TCL : convergence pour différentes distributions ===\n")
    for dist, params, nom in [("uniforme", {}, "U(0,1)"),
                                ("exponentielle", {"lam": 1}, "Exp(1)"),
                                ("bernoulli", {"p": 0.3}, "Bern(0.3)"),
                                ("de", {}, "Dé")]:
        print(f"  {nom:15s} :", end="")
        for n in [1, 5, 30, 100]:
            z = moyennes_echantillon(dist, n, **params)
            ks = test_normalite_ks(z)
            status = "✓" if ks["normal"] else "✗"
            print(f"  n={n:>3} KS={ks['KS_stat']:.4f}{status}", end="")
        print()

    print(f"\n=== Application du TCL ===\n")
    print(f"  Sondage : p = 0.6, n = 1000")
    print(f"  X̄ ~ N(p, p(1-p)/n) = N(0.6, {0.6*0.4/1000:.6f})")
    sigma = np.sqrt(0.6*0.4/1000)
    print(f"  σ_{'{X̄}'} = {sigma:.4f}")
    print(f"  IC 95% : [{0.6-1.96*sigma:.4f}, {0.6+1.96*sigma:.4f}]")
    print(f"  → Marge d'erreur ± {1.96*sigma:.4f} = ± {1.96*sigma*100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Convergence uniforme
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 3.5))
    tracer_tcl_convergence("exponentielle", (1, 2, 5, 30), axes2[0], lam=1)
    fig2.suptitle("TCL : Exp(1) → N(0,1)")
    fig2.tight_layout()
    fig2.savefig("clt_convergence.png", dpi=120)

    tracer_vitesse_convergence(ax=axes[0])
    # Multi distributions
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
    tracer_tcl_multi_distributions(30, axes3[0, 0])
    fig3.suptitle("TCL ($n=30$) pour 6 distributions différentes")
    fig3.tight_layout()
    fig3.savefig("clt_multi.png", dpi=120)

    axes[1].text(0.5, 0.5, "voir clt_convergence.png\net clt_multi.png",
                 transform=axes[1].transAxes, ha="center", va="center", fontsize=14)
    axes[2].text(0.5, 0.5, "figures séparées\npour plus de clarté",
                 transform=axes[2].transAxes, ha="center", va="center", fontsize=14)

    plt.tight_layout()
    print("\nFigures sauvegardées.")
