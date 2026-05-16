"""
special_functions.py
====================

Fonctions spéciales : Gamma, Beta, hyperboliques, et identités.

Couvre :
    - Fonction Gamma : Γ(n) = (n-1)! et Γ(x) = ∫₀^∞ t^{x-1} e^{-t} dt
    - Fonction Beta : B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    - Fonctions hyperboliques : sinh, cosh, tanh et leurs inverses
    - Identités remarquables (vérification numérique)
    - Fonction erreur : erf(x) = 2/√π ∫₀^x e^{-t²} dt

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from math import factorial, gamma as math_gamma

import numpy as np
from scipy import integrate, special
import matplotlib.pyplot as plt


# ======================================================================
#  1. Fonction Gamma
# ======================================================================

def gamma_integrale(x: float) -> float:
    """Γ(x) = ∫₀^∞ t^{x-1} e^{-t} dt (from-scratch par quadrature)."""
    if x <= 0:
        raise ValueError("Γ(x) non définie pour x ≤ 0 entier.")
    val, _ = integrate.quad(lambda t: t**(x-1) * np.exp(-t), 0, 200)
    return val


def gamma_stirling(n: float) -> float:
    """Approximation de Stirling : Γ(n+1) ≈ √(2πn) · (n/e)^n."""
    return np.sqrt(2 * np.pi * n) * (n / np.e)**n


def verifier_gamma_proprietes() -> None:
    """Vérifie Γ(n) = (n-1)! et Γ(x+1) = xΓ(x)."""
    print("  Γ(n) = (n-1)! :")
    for n in range(1, 8):
        g = gamma_integrale(n)
        print(f"    Γ({n}) = {g:.6f}, {n-1}! = {factorial(n-1)}")

    print(f"\n  Γ(x+1) = x·Γ(x) :")
    for x in [0.5, 1.5, 2.7, 3.14]:
        lhs = gamma_integrale(x + 1)
        rhs = x * gamma_integrale(x)
        print(f"    Γ({x+1:.2f}) = {lhs:.6f}, {x:.2f}·Γ({x:.2f}) = {rhs:.6f}, "
              f"err = {abs(lhs-rhs):.2e}")

    print(f"\n  Γ(1/2) = √π :")
    g_half = gamma_integrale(0.5)
    print(f"    Γ(0.5) = {g_half:.10f}, √π = {np.sqrt(np.pi):.10f}")


# ======================================================================
#  2. Fonction Beta
# ======================================================================

def beta_integrale(a: float, b: float) -> float:
    """B(a,b) = ∫₀¹ t^{a-1}(1-t)^{b-1} dt."""
    val, _ = integrate.quad(lambda t: t**(a-1) * (1-t)**(b-1), 0, 1)
    return val


def beta_par_gamma(a: float, b: float) -> float:
    """B(a,b) = Γ(a)Γ(b)/Γ(a+b)."""
    return math_gamma(a) * math_gamma(b) / math_gamma(a + b)


# ======================================================================
#  3. Fonctions hyperboliques
# ======================================================================

def sinh_scratch(x: float) -> float:
    """sinh(x) = (e^x - e^{-x}) / 2."""
    return (np.exp(x) - np.exp(-x)) / 2


def cosh_scratch(x: float) -> float:
    """cosh(x) = (e^x + e^{-x}) / 2."""
    return (np.exp(x) + np.exp(-x)) / 2


def tanh_scratch(x: float) -> float:
    """tanh(x) = sinh(x) / cosh(x)."""
    return sinh_scratch(x) / cosh_scratch(x)


def verifier_identites_hyperboliques(x: float) -> None:
    """Vérifie les identités fondamentales."""
    print(f"  x = {x} :")
    # cosh² - sinh² = 1
    lhs = cosh_scratch(x)**2 - sinh_scratch(x)**2
    print(f"    cosh²(x) - sinh²(x) = {lhs:.15f} = 1 ✓")

    # d/dx sinh = cosh
    h = 1e-7
    dsinh = (sinh_scratch(x+h) - sinh_scratch(x-h)) / (2*h)
    print(f"    sinh'(x) = {dsinh:.10f}, cosh(x) = {cosh_scratch(x):.10f} ✓")

    # d/dx cosh = sinh
    dcosh = (cosh_scratch(x+h) - cosh_scratch(x-h)) / (2*h)
    print(f"    cosh'(x) = {dcosh:.10f}, sinh(x) = {sinh_scratch(x):.10f} ✓")


# ======================================================================
#  4. Fonction erreur
# ======================================================================

def erf_scratch(x: float, n_terms: int = 50) -> float:
    """
    erf(x) = (2/√π) ∫₀^x e^{-t²} dt
           = (2/√π) Σ (-1)^n x^{2n+1} / (n!(2n+1)).
    """
    result = 0.0
    for n in range(n_terms):
        result += (-1)**n * x**(2*n+1) / (factorial(n) * (2*n+1))
    return 2 / np.sqrt(np.pi) * result


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_gamma(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(0.1, 5.5, 300)
    y = [math_gamma(xi) for xi in x]
    ax.plot(x, y, "b-", linewidth=2, label="$\\Gamma(x)$")

    # Points entiers
    ns = np.arange(1, 6)
    ax.plot(ns, [factorial(n-1) for n in ns], "ro", markersize=8,
            label="$\\Gamma(n) = (n-1)!$")

    # Stirling
    x_st = np.linspace(1, 5.5, 100)
    ax.plot(x_st, [gamma_stirling(xi-1) for xi in x_st], "g--", linewidth=1.5,
            label="Stirling")

    ax.set_xlabel("$x$"); ax.set_ylabel("$\\Gamma(x)$")
    ax.set_title("Fonction Gamma")
    ax.set_ylim(0, 30)
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_hyperboliques(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(-3, 3, 300)
    ax.plot(x, np.sinh(x), "b-", linewidth=2, label="$\\sinh(x)$")
    ax.plot(x, np.cosh(x), "r-", linewidth=2, label="$\\cosh(x)$")
    ax.plot(x, np.tanh(x), "g-", linewidth=2, label="$\\tanh(x)$")
    ax.axhline(1, color="green", linestyle=":", alpha=0.3)
    ax.axhline(-1, color="green", linestyle=":", alpha=0.3)

    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Fonctions hyperboliques")
    ax.set_ylim(-5, 5)
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_erf(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(-3, 3, 300)
    ax.plot(x, [erf_scratch(xi) for xi in x], "b-", linewidth=2, label="erf (série)")
    ax.plot(x, special.erf(x), "r--", linewidth=1.5, label="erf (scipy)")
    ax.axhline(1, color="grey", linestyle=":", alpha=0.3)
    ax.axhline(-1, color="grey", linestyle=":", alpha=0.3)

    ax.set_xlabel("$x$"); ax.set_ylabel("erf($x$)")
    ax.set_title("Fonction erreur : erf($x$) = $\\frac{2}{\\sqrt{\\pi}} \\int_0^x e^{-t^2} dt$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Fonction Gamma ===\n")
    verifier_gamma_proprietes()

    print(f"\n=== Stirling ===\n")
    for n in [5, 10, 20, 50]:
        exact = factorial(n)
        stirling = gamma_stirling(n)
        err = abs(stirling - exact) / exact
        print(f"  {n}! = {exact:.4e}, Stirling = {stirling:.4e}, err_rel = {err:.4f}")

    print(f"\n=== Fonction Beta ===\n")
    for a, b in [(2, 3), (0.5, 0.5), (1, 1)]:
        bi = beta_integrale(a, b)
        bg = beta_par_gamma(a, b)
        print(f"  B({a},{b}) : intégrale = {bi:.8f}, Γ = {bg:.8f}, err = {abs(bi-bg):.2e}")
    print(f"  B(1/2,1/2) = π = {np.pi:.6f} ✓")

    print(f"\n=== Fonctions hyperboliques ===\n")
    verifier_identites_hyperboliques(1.5)

    print(f"\n=== Fonction erreur ===\n")
    for x in [0.5, 1.0, 2.0, 3.0]:
        mine = erf_scratch(x)
        ref = float(special.erf(x))
        print(f"  erf({x}) = {mine:.10f} (scipy: {ref:.10f}, err = {abs(mine-ref):.2e})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_gamma(ax=axes[0])
    tracer_hyperboliques(ax=axes[1])
    tracer_erf(ax=axes[2])
    plt.tight_layout()
    plt.savefig("special_functions_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
