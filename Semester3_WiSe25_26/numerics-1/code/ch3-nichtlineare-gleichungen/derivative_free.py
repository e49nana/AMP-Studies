"""
derivative_free.py
==================

Méthodes sans dérivée pour les équations non-linéaires scalaires.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", sections 3.1.6, 3.1.7.

Couvre :
    - Regula Falsi (section 3.1.6)
    - Méthode de l'Illinois (amélioration de Regula Falsi)
    - Méthode hybride : bissection puis Newton/sécante
    - Comparaison des taux de convergence

Les méthodes de la section 3.1 (Newton, sécante, bissection) sont dans
newton_scalar.py. Ce module complète avec les variantes non couvertes.

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ResultatRootFinding:
    x: float
    iterations: int
    converge: bool
    methode: str
    historique_x: list[float] = field(default_factory=list)
    historique_fx: list[float] = field(default_factory=list)
    eval_count: int = 0

    def __repr__(self) -> str:
        statut = "convergé" if self.converge else "non convergé"
        return (
            f"ResultatRootFinding({self.methode}, {statut} en {self.iterations} it., "
            f"x ≈ {self.x:.15g}, {self.eval_count} évaluations de f)"
        )


# ======================================================================
#  1. Regula Falsi (section 3.1.6)
# ======================================================================

def regula_falsi(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-14,
    n_max: int = 200,
) -> ResultatRootFinding:
    """
    Regula Falsi : comme la bissection, mais au lieu de couper au milieu,
    on interpole linéairement entre (a, f(a)) et (b, f(b)) et on prend
    le zéro de cette droite.

    Avantage : converge plus vite que la bissection.
    Défaut : un des bords peut rester « collé » et la convergence
    dégénère en linéaire (avec un facteur asymptotique mauvais).
    """
    fa, fb = f(a), f(b)
    evals = 2
    if fa * fb > 0:
        raise ValueError("f(a) et f(b) doivent être de signes opposés.")

    hist_x, hist_fx = [], []
    converge = False

    for k in range(1, n_max + 1):
        # Interpolation linéaire : zéro de la sécante
        x = a - fa * (b - a) / (fb - fa)
        fx = f(x)
        evals += 1
        hist_x.append(x)
        hist_fx.append(fx)

        if abs(fx) < tol or abs(b - a) < tol:
            converge = True
            break

        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx

    return ResultatRootFinding(
        x=x, iterations=k, converge=converge, methode="Regula Falsi",
        historique_x=hist_x, historique_fx=hist_fx, eval_count=evals,
    )


# ======================================================================
#  2. Illinois (amélioration de Regula Falsi)
# ======================================================================

def illinois(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-14,
    n_max: int = 200,
) -> ResultatRootFinding:
    """
    Méthode de l'Illinois : variante de Regula Falsi qui évite le
    « stuck endpoint » en divisant par 2 la valeur du bord qui
    ne bouge pas.

    Convergence super-linéaire (≈ 1.442).
    """
    fa, fb = f(a), f(b)
    evals = 2
    if fa * fb > 0:
        raise ValueError("f(a) et f(b) doivent être de signes opposés.")

    hist_x, hist_fx = [], []
    converge = False
    side = 0  # 0 = aucun, -1 = a collé, 1 = b collé

    for k in range(1, n_max + 1):
        x = a - fa * (b - a) / (fb - fa)
        fx = f(x)
        evals += 1
        hist_x.append(x)
        hist_fx.append(fx)

        if abs(fx) < tol or abs(b - a) < tol:
            converge = True
            break

        if fa * fx < 0:
            b, fb = x, fx
            if side == -1:
                fa /= 2  # « Illinois trick » : décoller le bord a
            side = -1
        else:
            a, fa = x, fx
            if side == 1:
                fb /= 2
            side = 1

    return ResultatRootFinding(
        x=x, iterations=k, converge=converge, methode="Illinois",
        historique_x=hist_x, historique_fx=hist_fx, eval_count=evals,
    )


# ======================================================================
#  3. Hybride bissection → sécante (section 3.1.6 stratégie)
# ======================================================================

def hybride_bisection_secante(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-14,
    n_bisect: int = 10,
    n_max: int = 100,
) -> ResultatRootFinding:
    """
    Stratégie hybride (section 3.1.6) :
        1. Quelques pas de bissection pour réduire l'intervalle (convergence
           globale garantie).
        2. Basculer vers la sécante pour la convergence rapide (α ≈ 1.618).
    """
    fa, fb = f(a), f(b)
    evals = 2
    if fa * fb > 0:
        raise ValueError("f(a) et f(b) doivent être de signes opposés.")

    hist_x, hist_fx = [], []

    # Phase 1 : bissection
    for k in range(n_bisect):
        m = (a + b) / 2
        fm = f(m)
        evals += 1
        hist_x.append(m)
        hist_fx.append(fm)
        if abs(fm) < tol:
            return ResultatRootFinding(
                x=m, iterations=k + 1, converge=True,
                methode="Hybride (bisect→séc.)",
                historique_x=hist_x, historique_fx=hist_fx, eval_count=evals,
            )
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    # Phase 2 : sécante
    x_prev, x = a, b
    for k in range(n_bisect + 1, n_max + 1):
        fx = f(x)
        f_prev = f(x_prev)
        evals += 2
        denom = fx - f_prev
        if abs(denom) < 1e-300:
            break
        x_new = x - fx * (x - x_prev) / denom
        hist_x.append(x_new)
        hist_fx.append(f(x_new))
        evals += 1

        if abs(f(x_new)) < tol:
            return ResultatRootFinding(
                x=x_new, iterations=k, converge=True,
                methode="Hybride (bisect→séc.)",
                historique_x=hist_x, historique_fx=hist_fx, eval_count=evals,
            )
        x_prev, x = x, x_new

    return ResultatRootFinding(
        x=x, iterations=k, converge=False, methode="Hybride (bisect→séc.)",
        historique_x=hist_x, historique_fx=hist_fx, eval_count=evals,
    )


# ======================================================================
#  4. Tracé
# ======================================================================

def tracer_comparaison(
    resultats: list[ResultatRootFinding],
    x_star: float,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for r in resultats:
        erreurs = [abs(xi - x_star) for xi in r.historique_x]
        ax.semilogy(erreurs, "o-", markersize=4,
                    label=f"{r.methode} ({r.eval_count} évals)")
    ax.set_xlabel("itération")
    ax.set_ylabel("$|x_k - x^*|$")
    ax.set_title("Comparaison des méthodes sans dérivée")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    f = lambda x: x**3 - 2 * x - 5
    x_star = 2.0945514815423265

    print("=== f(x) = x³ - 2x - 5 sur [2, 3] ===")
    res_rf = regula_falsi(f, 2.0, 3.0)
    res_il = illinois(f, 2.0, 3.0)
    res_hy = hybride_bisection_secante(f, 2.0, 3.0, n_bisect=5)

    for r in [res_rf, res_il, res_hy]:
        print(f"  {r}")

    print("\n=== Tracé ===")
    tracer_comparaison([res_rf, res_il, res_hy], x_star)
    plt.tight_layout()
    plt.savefig("derivative_free_demo.png", dpi=120)
    print("Figure sauvegardée : derivative_free_demo.png")
