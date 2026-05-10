"""
sequences.py
=============

Suites numériques : convergence, monotonie, bornitude.

Couvre :
    - Suites arithmétiques et géométriques
    - Convergence et divergence (expérimentale)
    - Monotonie et bornitude (critère de convergence)
    - Suites récursives : x_{n+1} = f(x_n)
    - Suites classiques : sqrt(2) par récurrence, Heron, (1+1/n)^n → e
    - Visualisation de la convergence

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class AnalyseSuite:
    """Diagnostic de convergence d'une suite."""
    nom: str
    termes: np.ndarray
    limite_estimee: float | None
    monotone: str  # "croissante", "décroissante", "non monotone"
    bornee: bool


def suite_arithmetique(a0: float, d: float, n: int) -> np.ndarray:
    """a_k = a₀ + k·d."""
    return a0 + d * np.arange(n)


def suite_geometrique(a0: float, q: float, n: int) -> np.ndarray:
    """a_k = a₀ · q^k."""
    return a0 * q ** np.arange(n)


def suite_recursive(f: Callable[[float], float], x0: float, n: int) -> np.ndarray:
    """x_{k+1} = f(x_k), x₀ donné."""
    termes = [x0]
    x = x0
    for _ in range(n - 1):
        x = f(x)
        termes.append(x)
    return np.array(termes)


def estimer_limite(termes: np.ndarray, tol: float = 1e-10) -> float | None:
    """Estime la limite par les derniers termes, ou None si diverge."""
    if len(termes) < 10:
        return None
    tail = termes[-10:]
    if np.std(tail) < tol:
        return float(np.mean(tail))
    # Vérifier divergence
    if np.any(np.abs(tail) > 1e15):
        return None
    return float(tail[-1])


def analyser_monotonie(termes: np.ndarray) -> str:
    """Détecte si la suite est monotone."""
    diffs = np.diff(termes)
    if np.all(diffs >= -1e-12):
        return "croissante"
    elif np.all(diffs <= 1e-12):
        return "décroissante"
    return "non monotone"


def analyser(nom: str, termes: np.ndarray) -> AnalyseSuite:
    """Analyse complète d'une suite."""
    return AnalyseSuite(
        nom=nom,
        termes=termes,
        limite_estimee=estimer_limite(termes),
        monotone=analyser_monotonie(termes),
        bornee=bool(np.all(np.abs(termes) < 1e15)),
    )


# ======================================================================
#  Suites classiques
# ======================================================================

def heron_sqrt(a: float, x0: float, n: int) -> np.ndarray:
    """
    Méthode de Héron pour √a :
        x_{k+1} = (x_k + a/x_k) / 2.

    Converge quadratiquement vers √a.
    """
    return suite_recursive(lambda x: (x + a / x) / 2, x0, n)


def suite_euler(n_terms: int) -> np.ndarray:
    """(1 + 1/n)^n → e."""
    ns = np.arange(1, n_terms + 1)
    return (1 + 1.0 / ns) ** ns


def suite_harmonique_partielle(n: int) -> np.ndarray:
    """H_n = Σ_{k=1}^n 1/k (diverge vers +∞, mais lentement)."""
    return np.cumsum(1.0 / np.arange(1, n + 1))


# ======================================================================
#  Tracés
# ======================================================================

def tracer_suite(termes: np.ndarray, nom: str, limite: float | None = None,
                  ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(termes, "o-", markersize=3, linewidth=1, label=nom)
    if limite is not None:
        ax.axhline(limite, color="red", linestyle="--", alpha=0.5,
                    label=f"limite = {limite:.6f}")
    ax.set_xlabel("$n$"); ax.set_ylabel("$a_n$")
    ax.set_title(nom)
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_erreur_convergence(termes: np.ndarray, limite: float, nom: str,
                                ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    erreurs = np.abs(termes - limite)
    erreurs = np.maximum(erreurs, 1e-17)  # floor pour log
    ax.semilogy(erreurs, "o-", markersize=3, label=nom)
    ax.set_xlabel("$n$"); ax.set_ylabel("$|a_n - L|$")
    ax.set_title(f"Convergence de {nom}")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Suites arithmétiques et géométriques ===")
    arith = suite_arithmetique(1, 3, 10)
    geom_conv = suite_geometrique(1, 0.5, 20)
    geom_div = suite_geometrique(1, 2, 10)
    print(f"  Arithmétique (a₀=1, d=3) : {arith[:6]}...")
    print(f"  Géométrique (q=0.5) : converge vers {estimer_limite(geom_conv)}")
    print(f"  Géométrique (q=2)   : diverge ({geom_div[-1]:.0f})")

    print(f"\n=== Héron pour √2 ===")
    heron = heron_sqrt(2, 1.0, 8)
    for k, x in enumerate(heron):
        print(f"  x_{k} = {x:.15f}  (erreur = {abs(x - np.sqrt(2)):.2e})")
    print(f"  → Convergence quadratique (erreur² à chaque pas)")

    print(f"\n=== (1 + 1/n)^n → e ===")
    euler = suite_euler(1000)
    print(f"  n=10    : {euler[9]:.10f}")
    print(f"  n=100   : {euler[99]:.10f}")
    print(f"  n=1000  : {euler[999]:.10f}")
    print(f"  e exact : {np.e:.10f}")

    print(f"\n=== Suite harmonique (diverge) ===")
    H = suite_harmonique_partielle(10000)
    print(f"  H_10    = {H[9]:.4f}")
    print(f"  H_100   = {H[99]:.4f}")
    print(f"  H_10000 = {H[9999]:.4f}")
    print(f"  → Diverge, mais très lentement (∼ ln n)")

    print(f"\n=== Analyse automatique ===")
    suites = [
        ("Héron √2", heron_sqrt(2, 1.0, 20)),
        ("(1+1/n)^n", suite_euler(500)),
        ("géom. q=0.9", suite_geometrique(1, 0.9, 100)),
        ("géom. q=-0.5", suite_geometrique(1, -0.5, 30)),
    ]
    for nom, t in suites:
        a = analyser(nom, t)
        print(f"  {nom:20s} : limite ≈ {a.limite_estimee}, {a.monotone}, bornée={a.bornee}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_suite(heron_sqrt(2, 1.0, 10), "Héron √2", np.sqrt(2), ax=axes[0, 0])
    tracer_erreur_convergence(heron_sqrt(2, 1.0, 10), np.sqrt(2), "Héron √2", ax=axes[0, 1])
    tracer_suite(suite_euler(200), "(1+1/n)^n", np.e, ax=axes[1, 0])
    tracer_suite(suite_geometrique(1, -0.5, 20), "géom. q=-0.5", 0, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("sequences_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
