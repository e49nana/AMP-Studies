"""
derivatives.py
==============

Dérivées : définition, calcul numérique, règles de dérivation.

Couvre :
    - Définition : f'(x) = lim (f(x+h) - f(x)) / h
    - Différences finies : progressive, rétrograde, centrée
    - Dérivées d'ordre supérieur
    - Règles : somme, produit, quotient, chaîne
    - Vérification numérique des règles
    - Comparaison des ordres d'erreur : O(h) vs O(h²)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Différences finies
# ======================================================================

def derivee_progressive(f: Callable, x: float, h: float = 1e-7) -> float:
    """f'(x) ≈ (f(x+h) - f(x)) / h. Erreur O(h)."""
    return (f(x + h) - f(x)) / h


def derivee_retrograde(f: Callable, x: float, h: float = 1e-7) -> float:
    """f'(x) ≈ (f(x) - f(x-h)) / h. Erreur O(h)."""
    return (f(x) - f(x - h)) / h


def derivee_centree(f: Callable, x: float, h: float = 1e-5) -> float:
    """f'(x) ≈ (f(x+h) - f(x-h)) / (2h). Erreur O(h²)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def derivee_seconde(f: Callable, x: float, h: float = 1e-4) -> float:
    """f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h². Erreur O(h²)."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def derivee_nieme(f: Callable, x: float, n: int, h: float = 1e-3) -> float:
    """f^(n)(x) par différences finies récursives."""
    if n == 0:
        return f(x)
    if n == 1:
        return derivee_centree(f, x, h)
    # Récurrence : f^(n) ≈ (f^(n-1)(x+h) - f^(n-1)(x-h)) / (2h)
    return (derivee_nieme(f, x + h, n - 1, h) -
            derivee_nieme(f, x - h, n - 1, h)) / (2 * h)


# ======================================================================
#  2. Vérification des règles de dérivation
# ======================================================================

def verifier_regle_produit(f, g, df, dg, x: float) -> dict:
    """(fg)' = f'g + fg'."""
    fg_prime_num = derivee_centree(lambda t: f(t)*g(t), x)
    fg_prime_exact = df(x)*g(x) + f(x)*dg(x)
    return {
        "numérique": fg_prime_num,
        "f'g + fg'": fg_prime_exact,
        "erreur": abs(fg_prime_num - fg_prime_exact),
    }


def verifier_regle_chaine(f, g, df, dg, x: float) -> dict:
    """(f∘g)' = f'(g(x)) · g'(x)."""
    fog_prime_num = derivee_centree(lambda t: f(g(t)), x)
    fog_prime_exact = df(g(x)) * dg(x)
    return {
        "numérique": fog_prime_num,
        "f'(g(x))·g'(x)": fog_prime_exact,
        "erreur": abs(fog_prime_num - fog_prime_exact),
    }


def verifier_regle_quotient(f, g, df, dg, x: float) -> dict:
    """(f/g)' = (f'g - fg') / g²."""
    quot_prime_num = derivee_centree(lambda t: f(t)/g(t), x)
    quot_prime_exact = (df(x)*g(x) - f(x)*dg(x)) / g(x)**2
    return {
        "numérique": quot_prime_num,
        "(f'g - fg')/g²": quot_prime_exact,
        "erreur": abs(quot_prime_num - quot_prime_exact),
    }


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_derivee(f: Callable, df_exact: Callable | None, intervalle: tuple,
                    nom: str = "f", ax: plt.Axes | None = None) -> plt.Axes:
    """Trace f et f' (numérique et exacte si disponible)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(*intervalle, 300)
    ax.plot(x, [f(xi) for xi in x], "b-", linewidth=2, label=f"${nom}(x)$")

    df_num = [derivee_centree(f, xi) for xi in x]
    ax.plot(x, df_num, "r--", linewidth=2, label=f"${nom}'(x)$ (numérique)")

    if df_exact is not None:
        ax.plot(x, [df_exact(xi) for xi in x], "g:", linewidth=2,
                label=f"${nom}'(x)$ (exacte)")

    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"Dérivée de ${nom}$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_erreur_ordre(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare l'erreur O(h) vs O(h²) pour sin'(1) = cos(1)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    exact = np.cos(1.0)
    hs = np.logspace(-12, 0, 50)
    err_prog = [abs(derivee_progressive(np.sin, 1.0, h) - exact) for h in hs]
    err_cent = [abs(derivee_centree(np.sin, 1.0, h) - exact) for h in hs]

    ax.loglog(hs, err_prog, "r-", linewidth=2, label="Progressive $O(h)$")
    ax.loglog(hs, err_cent, "b-", linewidth=2, label="Centrée $O(h^2)$")
    ax.loglog(hs, hs, "r:", alpha=0.3, label="$h$")
    ax.loglog(hs, hs**2, "b:", alpha=0.3, label="$h^2$")
    ax.set_xlabel("$h$"); ax.set_ylabel("erreur")
    ax.set_title("Progressive $O(h)$ vs centrée $O(h^2)$")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Différences finies sur sin'(1) = cos(1) ===")
    exact = np.cos(1.0)
    print(f"  Exacte     : {exact:.15f}")
    for nom, fn in [("progressive", derivee_progressive),
                     ("rétrograde", derivee_retrograde),
                     ("centrée", derivee_centree)]:
        val = fn(np.sin, 1.0)
        print(f"  {nom:12s} : {val:.15f}  (err = {abs(val-exact):.2e})")

    print(f"\n=== Dérivées d'ordre supérieur de sin ===")
    for n in range(5):
        val = derivee_nieme(np.sin, 0.0, n)
        attendu = [0, 1, 0, -1, 0][n]
        print(f"  sin^({n})(0) = {val:>10.4f}  (attendu: {attendu})")

    print(f"\n=== Vérification des règles ===")
    f, g = np.sin, np.cos
    df, dg = np.cos, lambda x: -np.sin(x)
    x = 1.0
    print(f"  Produit  : {verifier_regle_produit(f, g, df, dg, x)}")
    print(f"  Chaîne   : {verifier_regle_chaine(np.exp, np.sin, np.exp, np.cos, x)}")
    print(f"  Quotient : {verifier_regle_quotient(f, g, df, dg, x)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_derivee(np.sin, np.cos, (-2*np.pi, 2*np.pi), "\\sin", ax=axes[0])
    tracer_erreur_ordre(ax=axes[1])
    plt.tight_layout()
    plt.savefig("derivatives_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
