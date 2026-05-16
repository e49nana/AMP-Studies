"""
fourier_series.py
=================

Séries de Fourier et phénomène de Gibbs.

Couvre :
    - Coefficients de Fourier : a₀, aₙ, bₙ
    - Somme partielle S_N(x) = a₀/2 + Σ (aₙ cos(nx) + bₙ sin(nx))
    - Signaux classiques : carré, dent de scie, triangle
    - Phénomène de Gibbs : oscillation ~9% aux discontinuités
    - Identité de Parseval : Σ(aₙ² + bₙ²) = (1/π) ∫|f|²
    - Convergence L² vs convergence ponctuelle

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# ======================================================================
#  1. Coefficients de Fourier
# ======================================================================

def fourier_coefficients(
    f: Callable, N: int, T: float = 2*np.pi,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calcule a₀, a₁..aₙ, b₁..bₙ pour f de période T.

    a₀ = (1/π) ∫₋π^π f(x) dx
    aₙ = (1/π) ∫₋π^π f(x)cos(nx) dx
    bₙ = (1/π) ∫₋π^π f(x)sin(nx) dx
    """
    L = T / 2
    a0, _ = integrate.quad(lambda x: f(x), -L, L)
    a0 /= L

    an = np.zeros(N)
    bn = np.zeros(N)
    for n in range(1, N + 1):
        an[n-1], _ = integrate.quad(lambda x, n=n: f(x) * np.cos(n*np.pi*x/L), -L, L)
        an[n-1] /= L
        bn[n-1], _ = integrate.quad(lambda x, n=n: f(x) * np.sin(n*np.pi*x/L), -L, L)
        bn[n-1] /= L

    return a0, an, bn


def fourier_partielle(
    a0: float, an: np.ndarray, bn: np.ndarray, x: np.ndarray, T: float = 2*np.pi,
) -> np.ndarray:
    """Somme partielle S_N(x) = a₀/2 + Σ_{n=1}^N (aₙcos(nωx) + bₙsin(nωx))."""
    L = T / 2
    result = np.full_like(x, a0 / 2, dtype=float)
    for n in range(1, len(an) + 1):
        result += an[n-1] * np.cos(n * np.pi * x / L)
        result += bn[n-1] * np.sin(n * np.pi * x / L)
    return result


# ======================================================================
#  2. Signaux classiques
# ======================================================================

def signal_carre(x: float) -> float:
    """Signal carré de période 2π : +1 sur [0,π), -1 sur [π,2π)."""
    x = x % (2*np.pi)
    return 1.0 if x < np.pi else -1.0


def signal_dent_de_scie(x: float) -> float:
    """Dent de scie : f(x) = x/π sur (-π, π)."""
    x = ((x + np.pi) % (2*np.pi)) - np.pi
    return x / np.pi


def signal_triangle(x: float) -> float:
    """Triangle : |x|/π sur (-π, π)."""
    x = ((x + np.pi) % (2*np.pi)) - np.pi
    return abs(x) / np.pi


# Coefficients analytiques
def fourier_carre_analytique(N: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Carré : bₙ = 4/(nπ) pour n impair, 0 sinon."""
    an = np.zeros(N)
    bn = np.array([4/(n*np.pi) if n % 2 == 1 else 0 for n in range(1, N+1)])
    return 0.0, an, bn


def fourier_dent_analytique(N: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Dent de scie : bₙ = 2(-1)^{n+1} / (nπ)."""
    an = np.zeros(N)
    bn = np.array([2*(-1)**(n+1) / (n*np.pi) for n in range(1, N+1)])
    return 0.0, an, bn


# ======================================================================
#  3. Phénomène de Gibbs
# ======================================================================

def mesurer_gibbs(N: int) -> float:
    """Mesure le dépassement de Gibbs pour le signal carré avec N termes."""
    _, an, bn = fourier_carre_analytique(N)
    # Le maximum est juste après la discontinuité en x = 0⁺
    x_test = np.linspace(0.001, 0.5, 5000)
    S = fourier_partielle(0, an, bn, x_test)
    return float(np.max(S))


# ======================================================================
#  4. Parseval
# ======================================================================

def verifier_parseval(f: Callable, N: int) -> dict:
    """
    Identité de Parseval :
        a₀²/2 + Σ(aₙ² + bₙ²) = (1/π) ∫₋π^π |f(x)|² dx.
    """
    a0, an, bn = fourier_coefficients(f, N)
    lhs = a0**2 / 2 + np.sum(an**2 + bn**2)
    rhs, _ = integrate.quad(lambda x: f(x)**2, -np.pi, np.pi)
    rhs /= np.pi
    return {"lhs (Fourier)": lhs, "rhs (intégrale)": rhs, "erreur": abs(lhs - rhs)}


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_fourier(
    f: Callable, nom: str, Ns: tuple[int, ...] = (1, 3, 5, 15, 50),
    a0_an_bn_fn: Callable | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    x = np.linspace(-np.pi, np.pi, 1000)
    f_vec = np.vectorize(f)
    ax.plot(x, f_vec(x), "k-", linewidth=2.5, label=f"${nom}(x)$")

    for N in Ns:
        if a0_an_bn_fn is not None:
            a0, an, bn = a0_an_bn_fn(N)
        else:
            a0, an, bn = fourier_coefficients(f, N)
        S = fourier_partielle(a0, an, bn, x)
        ax.plot(x, S, linewidth=1.5, alpha=0.7, label=f"$S_{{{N}}}$")

    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"Fourier de ${nom}$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


def tracer_gibbs(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    Ns = range(5, 200, 5)
    depassements = [mesurer_gibbs(N) for N in Ns]
    gibbs_limit = 1 + 2 * integrate.quad(lambda x: np.sin(x)/x, 0, np.pi)[0] / np.pi - 1

    ax.plot(list(Ns), depassements, "b-", linewidth=2)
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5, label="signal ($f = 1$)")
    ax.axhline(1 + 0.0895, color="red", linestyle="--", alpha=0.5,
                label="limite Gibbs (≈ 1.09)")
    ax.set_xlabel("$N$ (nombre de termes)")
    ax.set_ylabel("max $S_N$ près de la discontinuité")
    ax.set_title("Phénomène de Gibbs : ~9% de dépassement persistant")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Coefficients de Fourier — Signal carré ===\n")
    a0, an, bn = fourier_carre_analytique(10)
    print(f"  a₀ = {a0}")
    print(f"  aₙ = {np.round(an[:5], 6)} (tous nuls)")
    print(f"  bₙ = {np.round(bn[:5], 6)}")
    print(f"  → Seulement des sinus (fonction impaire)")

    print(f"\n=== Vérification numérique vs analytique ===\n")
    a0_num, an_num, bn_num = fourier_coefficients(signal_carre, 10)
    print(f"  ||bₙ(num) - bₙ(exact)||_∞ = {np.max(np.abs(bn_num - bn)):.2e}")

    print(f"\n=== Phénomène de Gibbs ===\n")
    for N in [10, 50, 100, 500]:
        g = mesurer_gibbs(N)
        print(f"  N = {N:>3} : max S_N = {g:.6f} (dépassement = {(g-1)*100:.2f}%)")
    print(f"  → Le dépassement converge vers ≈ 8.95%, ne disparaît jamais !")

    print(f"\n=== Parseval ===\n")
    for nom, f in [("carré", signal_carre), ("triangle", signal_triangle)]:
        p = verifier_parseval(f, 50)
        print(f"  {nom:10s} : Fourier = {p['lhs (Fourier)']:.6f}, "
              f"intégrale = {p['rhs (intégrale)']:.6f}, "
              f"erreur = {p['erreur']:.2e}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_fourier(signal_carre, "carré", (1, 3, 5, 15, 50),
                    fourier_carre_analytique, ax=axes[0, 0])
    tracer_fourier(signal_dent_de_scie, "dent\\_de\\_scie", (1, 3, 10, 30),
                    fourier_dent_analytique, ax=axes[0, 1])
    tracer_fourier(signal_triangle, "triangle", (1, 3, 10, 30), ax=axes[1, 0])
    tracer_gibbs(ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("fourier_series_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
