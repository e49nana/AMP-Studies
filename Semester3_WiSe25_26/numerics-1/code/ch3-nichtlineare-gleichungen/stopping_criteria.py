"""
stopping_criteria.py
====================

Comparaison des critères d'arrêt pour les méthodes itératives.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 3.1.5.

Couvre :
    - Critère 1 : |f(x)| ≤ τ₁ (résidu)
    - Critère 2 : |x⁺ - x| / |x⁺| ≤ τ₂ (changement relatif)
    - Critère 3 : |x⁺ - x| ≤ τ₃ (changement absolu)
    - Piège : résidu petit ≠ solution précise (et inversement)
    - Recommandation du script : combiner les 3

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class TraceConvergence:
    methode: str
    historique_x: list[float] = field(default_factory=list)
    historique_fx: list[float] = field(default_factory=list)
    historique_dx: list[float] = field(default_factory=list)
    arret_residu: int | None = None
    arret_relatif: int | None = None
    arret_absolu: int | None = None


def newton_avec_criteres(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tau1: float = 1e-10,
    tau2: float = 1e-8,
    tau3: float = 1e-14,
    n_max: int = 100,
) -> TraceConvergence:
    """Newton en traçant quand chaque critère d'arrêt est satisfait."""
    trace = TraceConvergence(methode="Newton")
    x = x0

    for k in range(n_max):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            break
        x_new = x - fx / dfx

        trace.historique_x.append(x_new)
        trace.historique_fx.append(abs(f(x_new)))
        dx = abs(x_new - x)
        trace.historique_dx.append(dx)

        # Critère 1 : résidu
        if trace.arret_residu is None and abs(f(x_new)) <= tau1:
            trace.arret_residu = k
        # Critère 2 : changement relatif
        if trace.arret_relatif is None and abs(x_new) > 0 and dx / abs(x_new) <= tau2:
            trace.arret_relatif = k
        # Critère 3 : changement absolu
        if trace.arret_absolu is None and dx <= tau3:
            trace.arret_absolu = k

        x = x_new

    return trace


def demo_piege_residu():
    """
    Montre que |f(x)| petit ne signifie pas x ≈ x*.

    Exemple : f(x) = (x - 1)^3. Près de x* = 1 :
    - |f(0.99)| = 10⁻⁶ (résidu minuscule)
    - |x - x*| = 0.01 (erreur de 1%)
    """
    f = lambda x: (x - 1)**3
    x_test = 0.99
    print(f"  f(x) = (x-1)³, x* = 1")
    print(f"  x = {x_test} : |f(x)| = {abs(f(x_test)):.2e}, |x - x*| = {abs(x_test - 1):.2e}")
    print(f"  → Résidu minuscule mais erreur de 1% !")

    x_test2 = 1.0 + 1e-5
    print(f"\n  x = {x_test2} : |f(x)| = {abs(f(x_test2)):.2e}, |x - x*| = {abs(x_test2 - 1):.2e}")
    print(f"  → Erreur minuscule mais résidu encore plus petit.")


def tracer_criteres(
    traces: list[TraceConvergence],
    x_star: float,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    for t in traces:
        erreurs = [abs(xi - x_star) for xi in t.historique_x]
        line, = ax.semilogy(erreurs, "o-", markersize=4, label=t.methode)
        c = line.get_color()
        if t.arret_residu is not None:
            ax.axvline(t.arret_residu, color=c, linestyle=":", alpha=0.5)
            ax.annotate("τ₁", (t.arret_residu, erreurs[t.arret_residu]),
                        fontsize=8, color=c)
        if t.arret_relatif is not None:
            ax.axvline(t.arret_relatif, color=c, linestyle="--", alpha=0.5)
            ax.annotate("τ₂", (t.arret_relatif, erreurs[min(t.arret_relatif, len(erreurs)-1)]),
                        fontsize=8, color=c)

    ax.set_xlabel("itération")
    ax.set_ylabel("$|x_k - x^*|$")
    ax.set_title("§3.1.5 — quand chaque critère se déclenche")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Piège du résidu petit ===")
    demo_piege_residu()

    print("\n=== Newton : 3 critères comparés ===")
    f = lambda x: x**2 - 3
    df = lambda x: 2 * x
    trace = newton_avec_criteres(f, df, 4.5, tau1=1e-10, tau2=1e-8, tau3=1e-14)
    print(f"  τ₁ (résidu)   déclenché à l'itération {trace.arret_residu}")
    print(f"  τ₂ (relatif)  déclenché à l'itération {trace.arret_relatif}")
    print(f"  τ₃ (absolu)   déclenché à l'itération {trace.arret_absolu}")

    tracer_criteres([trace], np.sqrt(3))
    plt.tight_layout()
    plt.savefig("stopping_criteria_demo.png", dpi=120)
    print("Figure sauvegardée.")
