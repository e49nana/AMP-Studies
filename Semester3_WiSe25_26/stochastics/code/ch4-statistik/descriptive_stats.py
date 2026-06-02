"""
descriptive_stats.py
====================

Statistiques descriptives from-scratch.

Couvre :
    - Mesures de position : moyenne, médiane, mode, quantiles
    - Mesures de dispersion : variance, écart-type, IQR, étendue
    - Mesures de forme : asymétrie (skewness), aplatissement (kurtosis)
    - Visualisations : histogramme, boxplot, stem-and-leaf
    - Données groupées vs individuelles

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Mesures de position
# ======================================================================

def moyenne(data: np.ndarray) -> float:
    """x̄ = (1/n) Σ xᵢ."""
    return float(np.sum(data) / len(data))


def mediane(data: np.ndarray) -> float:
    """Valeur centrale des données triées."""
    s = np.sort(data)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n//2 - 1] + s[n//2]) / 2)


def mode(data: np.ndarray, n_bins: int = 20) -> float:
    """Valeur la plus fréquente (approx. par histogramme pour données continues)."""
    counts, edges = np.histogram(data, bins=n_bins)
    idx = np.argmax(counts)
    return float((edges[idx] + edges[idx + 1]) / 2)


def quantile(data: np.ndarray, q: float) -> float:
    """q-quantile (interpolation linéaire)."""
    s = np.sort(data)
    pos = q * (len(s) - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (pos - lo) * (s[hi] - s[lo]))


def quartiles(data: np.ndarray) -> dict:
    """Q1, Q2 (médiane), Q3."""
    return {"Q1": quantile(data, 0.25), "Q2": mediane(data), "Q3": quantile(data, 0.75)}


# ======================================================================
#  2. Mesures de dispersion
# ======================================================================

def variance_pop(data: np.ndarray) -> float:
    """σ² = (1/n) Σ(xᵢ - x̄)² (population)."""
    m = moyenne(data)
    return float(np.sum((data - m)**2) / len(data))


def variance_echantillon(data: np.ndarray) -> float:
    """s² = (1/(n-1)) Σ(xᵢ - x̄)² (Bessel correction)."""
    m = moyenne(data)
    return float(np.sum((data - m)**2) / (len(data) - 1))


def ecart_type(data: np.ndarray, population: bool = False) -> float:
    if population:
        return np.sqrt(variance_pop(data))
    return np.sqrt(variance_echantillon(data))


def iqr(data: np.ndarray) -> float:
    """IQR = Q3 - Q1 (interquartile range)."""
    q = quartiles(data)
    return q["Q3"] - q["Q1"]


def etendue(data: np.ndarray) -> float:
    """Range = max - min."""
    return float(np.max(data) - np.min(data))


def coefficient_variation(data: np.ndarray) -> float:
    """CV = σ/μ (dispersion relative)."""
    m = moyenne(data)
    return ecart_type(data) / abs(m) if m != 0 else float("inf")


# ======================================================================
#  3. Mesures de forme
# ======================================================================

def skewness(data: np.ndarray) -> float:
    """Asymétrie : γ₁ = E[(X-μ)³] / σ³. > 0 = queue à droite."""
    m = moyenne(data)
    s = ecart_type(data, population=True)
    n = len(data)
    return float(np.sum((data - m)**3) / (n * s**3)) if s > 0 else 0


def kurtosis(data: np.ndarray) -> float:
    """Aplatissement : γ₂ = E[(X-μ)⁴]/σ⁴ - 3. Normale = 0."""
    m = moyenne(data)
    s = ecart_type(data, population=True)
    n = len(data)
    return float(np.sum((data - m)**4) / (n * s**4) - 3) if s > 0 else 0


# ======================================================================
#  4. Résumé complet
# ======================================================================

def resume_statistique(data: np.ndarray, nom: str = "données") -> dict:
    q = quartiles(data)
    return {
        "nom": nom, "n": len(data),
        "moyenne": moyenne(data), "médiane": mediane(data),
        "écart-type": ecart_type(data), "variance": variance_echantillon(data),
        "min": float(np.min(data)), "max": float(np.max(data)),
        "Q1": q["Q1"], "Q3": q["Q3"], "IQR": iqr(data),
        "skewness": skewness(data), "kurtosis": kurtosis(data),
    }


def afficher_resume(r: dict) -> None:
    print(f"  {r['nom']} (n = {r['n']})")
    print(f"    Moyenne    = {r['moyenne']:.4f}")
    print(f"    Médiane    = {r['médiane']:.4f}")
    print(f"    Écart-type = {r['écart-type']:.4f}")
    print(f"    Min = {r['min']:.2f}, Q1 = {r['Q1']:.2f}, Q3 = {r['Q3']:.2f}, Max = {r['max']:.2f}")
    print(f"    IQR = {r['IQR']:.4f}")
    print(f"    Skewness = {r['skewness']:.4f}, Kurtosis = {r['kurtosis']:.4f}")


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_histogramme_boxplot(data: np.ndarray, nom: str = "",
                                 axes: list | None = None) -> None:
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(data, bins=30, density=True, alpha=0.7, color="steelblue",
                  edgecolor="white")
    m = moyenne(data)
    med = mediane(data)
    axes[0].axvline(m, color="red", linewidth=2, linestyle="--", label=f"moyenne = {m:.2f}")
    axes[0].axvline(med, color="green", linewidth=2, linestyle=":", label=f"médiane = {med:.2f}")
    axes[0].set_xlabel("valeur"); axes[0].set_ylabel("densité")
    axes[0].set_title(f"Histogramme {nom}")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    bp = axes[1].boxplot(data, vert=True, patch_artist=True,
                          boxprops=dict(facecolor="lightskyblue", alpha=0.7))
    q = quartiles(data)
    axes[1].set_title(f"Boxplot {nom}\nIQR = {iqr(data):.2f}")
    axes[1].grid(True, alpha=0.3)


def tracer_comparaison_distributions(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare moyenne vs médiane pour données symétriques et asymétriques."""
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = [ax]*3

    rng = np.random.default_rng(42)
    datasets = [
        ("Symétrique (normale)", rng.normal(50, 10, 1000)),
        ("Asymétrique droite (exp)", rng.exponential(10, 1000)),
        ("Avec outliers", np.concatenate([rng.normal(50, 5, 980), rng.normal(100, 2, 20)])),
    ]

    for ax_i, (nom, data) in zip(axes, datasets):
        ax_i.hist(data, bins=30, density=True, alpha=0.6, color="steelblue", edgecolor="white")
        m, med = moyenne(data), mediane(data)
        ax_i.axvline(m, color="red", linewidth=2, linestyle="--", label=f"moy={m:.1f}")
        ax_i.axvline(med, color="green", linewidth=2, linestyle=":", label=f"méd={med:.1f}")
        ax_i.set_title(f"{nom}\nskew = {skewness(data):.2f}")
        ax_i.legend(fontsize=8); ax_i.grid(True, alpha=0.3)

    return axes[0]


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== Résumé statistique ===\n")
    data_norm = rng.normal(100, 15, 200)
    afficher_resume(resume_statistique(data_norm, "N(100, 15²)"))

    print()
    data_exp = rng.exponential(10, 200)
    afficher_resume(resume_statistique(data_exp, "Exp(0.1)"))

    print(f"\n=== Vérification from-scratch vs NumPy ===\n")
    print(f"  moyenne : {moyenne(data_norm):.6f} vs {np.mean(data_norm):.6f}")
    print(f"  var(s²) : {variance_echantillon(data_norm):.4f} vs {np.var(data_norm, ddof=1):.4f}")
    print(f"  médiane : {mediane(data_norm):.6f} vs {np.median(data_norm):.6f}")
    print(f"  Q1      : {quantile(data_norm, 0.25):.4f} vs {np.quantile(data_norm, 0.25):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tracer_histogramme_boxplot(data_norm, "N(100, 15²)", axes)
    plt.tight_layout()
    plt.savefig("descriptive_stats_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
