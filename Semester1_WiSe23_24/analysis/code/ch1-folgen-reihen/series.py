"""
series.py
=========

Séries numériques : convergence et critères.

Couvre :
    - Sommes partielles S_n = Σ_{k=0}^n a_k
    - Série géométrique : Σ q^k = 1/(1-q) pour |q| < 1
    - Série harmonique : Σ 1/k diverge
    - Série harmonique alternée : Σ (-1)^k/k = ln 2
    - Critère de d'Alembert (quotient)
    - Critère de Cauchy (racine)
    - Critère de Leibniz (alternée)
    - Série de Riemann : Σ 1/k^p converge ssi p > 1

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def sommes_partielles(termes: np.ndarray) -> np.ndarray:
    """S_n = Σ_{k=0}^n a_k."""
    return np.cumsum(termes)


# ======================================================================
#  Séries classiques
# ======================================================================

def serie_geometrique(q: float, n: int) -> np.ndarray:
    """Termes q^k pour k = 0, ..., n-1."""
    return q ** np.arange(n)


def serie_harmonique(n: int) -> np.ndarray:
    """Termes 1/k pour k = 1, ..., n."""
    return 1.0 / np.arange(1, n + 1)


def serie_harmonique_alternee(n: int) -> np.ndarray:
    """Termes (-1)^{k+1}/k pour k = 1, ..., n."""
    k = np.arange(1, n + 1)
    return (-1.0) ** (k + 1) / k


def serie_riemann(p: float, n: int) -> np.ndarray:
    """Termes 1/k^p pour k = 1, ..., n."""
    return 1.0 / np.arange(1, n + 1) ** p


def serie_exponentielle(x: float, n: int) -> np.ndarray:
    """Termes x^k / k! pour k = 0, ..., n-1 (→ e^x)."""
    termes = np.zeros(n)
    termes[0] = 1.0
    for k in range(1, n):
        termes[k] = termes[k - 1] * x / k
    return termes


# ======================================================================
#  Critères de convergence
# ======================================================================

def critere_dalembert(termes: np.ndarray) -> dict:
    """
    Critère du quotient (d'Alembert) :
        L = lim |a_{k+1}/a_k|.
        L < 1 → converge absolument.
        L > 1 → diverge.
        L = 1 → indéterminé.
    """
    abs_termes = np.abs(termes)
    ratios = abs_termes[1:] / np.maximum(abs_termes[:-1], 1e-300)
    L = ratios[-1] if len(ratios) > 0 else float("nan")
    if L < 1 - 1e-10:
        verdict = "converge (L < 1)"
    elif L > 1 + 1e-10:
        verdict = "diverge (L > 1)"
    else:
        verdict = "indéterminé (L ≈ 1)"
    return {"L": float(L), "verdict": verdict, "ratios_tail": ratios[-5:].tolist()}


def critere_cauchy(termes: np.ndarray) -> dict:
    """
    Critère de la racine (Cauchy) :
        L = lim |a_k|^{1/k}.
        Mêmes conclusions que d'Alembert.
    """
    k = np.arange(1, len(termes) + 1)
    roots = np.abs(termes) ** (1.0 / k)
    L = roots[-1]
    if L < 1 - 1e-10:
        verdict = "converge (L < 1)"
    elif L > 1 + 1e-10:
        verdict = "diverge (L > 1)"
    else:
        verdict = "indéterminé (L ≈ 1)"
    return {"L": float(L), "verdict": verdict}


def critere_leibniz(termes: np.ndarray) -> dict:
    """
    Critère de Leibniz (séries alternées) :
        Si |a_k| décroît vers 0 et les signes alternent → converge.
    """
    abs_t = np.abs(termes)
    decroissant = bool(np.all(np.diff(abs_t) <= 1e-12))
    vers_zero = abs_t[-1] < 1e-8
    signes_alternent = bool(np.all(np.diff(np.sign(termes)) != 0))
    converge = decroissant and vers_zero and signes_alternent
    return {
        "décroissant": decroissant,
        "vers_zero": vers_zero,
        "alternée": signes_alternent,
        "converge": converge,
    }


# ======================================================================
#  Tracés
# ======================================================================

def tracer_series(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare plusieurs séries classiques."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    n = 50
    series = [
        ("géom. q=0.5 → 2", sommes_partielles(serie_geometrique(0.5, n))),
        ("harm. alternée → ln 2", sommes_partielles(serie_harmonique_alternee(n))),
        ("Riemann p=2 → π²/6", sommes_partielles(serie_riemann(2, n))),
        ("harmonique (diverge)", sommes_partielles(serie_harmonique(n))),
    ]
    limites = [2, np.log(2), np.pi**2/6, None]

    for (nom, sp), lim in zip(series, limites):
        line, = ax.plot(sp, linewidth=2, label=nom)
        if lim is not None:
            ax.axhline(lim, color=line.get_color(), linestyle="--", alpha=0.4)

    ax.set_xlabel("$n$"); ax.set_ylabel("$S_n$")
    ax.set_title("Séries classiques — sommes partielles")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_riemann(ax: plt.Axes | None = None) -> plt.Axes:
    """Série de Riemann Σ 1/k^p : converge ssi p > 1."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    n = 500
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        sp = sommes_partielles(serie_riemann(p, n))
        ax.plot(sp, linewidth=2, label=f"p = {p}")

    ax.set_xlabel("$n$"); ax.set_ylabel("$S_n = \\sum 1/k^p$")
    ax.set_title("Série de Riemann : converge ssi $p > 1$")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 20)
    return ax


if __name__ == "__main__":
    print("=== Série géométrique ===")
    for q in [0.5, 0.9, -0.5, 1.1]:
        sp = sommes_partielles(serie_geometrique(q, 100))
        theo = 1/(1-q) if abs(q) < 1 else float("inf")
        print(f"  q = {q:>5} : S_100 = {sp[-1]:>12.6f}, théorique = {theo}")

    print(f"\n=== Séries spéciales ===")
    print(f"  Σ(-1)^k/k = {sommes_partielles(serie_harmonique_alternee(10000))[-1]:.10f} "
          f"(ln 2 = {np.log(2):.10f})")
    print(f"  Σ 1/k²    = {sommes_partielles(serie_riemann(2, 10000))[-1]:.10f} "
          f"(π²/6 = {np.pi**2/6:.10f})")
    print(f"  e = Σ 1/k! = {sommes_partielles(serie_exponentielle(1, 20))[-1]:.15f} "
          f"(exact = {np.e:.15f})")

    print(f"\n=== Critères de convergence ===")
    cas = [
        ("géom. q=0.5", serie_geometrique(0.5, 50)),
        ("harmonique", serie_harmonique(50)),
        ("1/k²", serie_riemann(2, 50)),
        ("exp(1)", serie_exponentielle(1, 20)),
    ]
    for nom, t in cas:
        da = critere_dalembert(t)
        print(f"  {nom:20s} : d'Alembert L = {da['L']:.6f} → {da['verdict']}")

    print(f"\n=== Leibniz ===")
    lb = critere_leibniz(serie_harmonique_alternee(100))
    print(f"  Σ(-1)^k/k : {lb}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_series(ax=axes[0])
    tracer_riemann(ax=axes[1])
    plt.tight_layout()
    plt.savefig("series_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
