"""
newton_scalar.py
================

Méthode de Newton pour les équations non-linéaires scalaires f(x) = 0,
avec variantes et mesure de la convergence.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", chapitre 3.1.

Couvre :
    - Newton classique (formule 3.3)
    - Sécante (section 3.1.7)
    - Bissection (section 3.1.6)
    - Convergence expérimentale (section 3.1.4)
    - Trois critères d'arrêt combinés (section 3.1.5)
    - Conditionnement du problème de recherche de zéros (section 3.1.2)
    - Newton modifié pour nullstelles multiples (section 3.1.8)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Résultat d'itération
# ======================================================================

@dataclass
class ResultatNewton:
    """Stocke la solution, l'historique et les diagnostics de convergence."""
    x: float
    iterations: int
    converge: bool
    methode: str
    historique_x: list[float] = field(default_factory=list)
    historique_fx: list[float] = field(default_factory=list)
    ordres_experimentaux: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        statut = "convergé" if self.converge else "non convergé"
        return (
            f"ResultatNewton({self.methode}, {statut} "
            f"en {self.iterations} itérations, x ≈ {self.x:.15g})"
        )


# ======================================================================
#  2. Critères d'arrêt combinés (section 3.1.5)
# ======================================================================

def criteres_arrêt(
    x_new: float,
    x_old: float,
    fx: float,
    tau1: float = 1e-14,
    tau2: float | None = None,
    tau3: float = 1e-15,
) -> bool:
    """
    Combinaison des 3 critères de la section 3.1.5 :
        |f(x⁺)| ≤ τ₁              (résidu petit)
        |x⁺ - x| / |x⁺| ≤ τ₂     (changement relatif petit)
        |x⁺ - x| ≤ τ₃             (changement absolu petit)

    Avec τ₂ = √(ε_mach) par défaut, comme recommandé dans le script.
    """
    if tau2 is None:
        tau2 = np.sqrt(np.finfo(float).eps)

    if abs(fx) <= tau1:
        return True
    dx = abs(x_new - x_old)
    if dx <= tau3:
        return True
    if abs(x_new) > 0 and dx / abs(x_new) <= tau2:
        return True
    return False


# ======================================================================
#  3. Ordre de convergence expérimental (section 3.1.4)
# ======================================================================

def ordre_experimental_exact(
    historique_x: list[float], x_star: float,
) -> list[float]:
    """
    Calcul avec la solution exacte connue (section 3.1.4, cas 1) :
        α ≈ ln(e_k / e_{k+1}) / ln(e_{k-1} / e_k)
    """
    e = [abs(xi - x_star) for xi in historique_x]
    ordres = []
    for k in range(1, len(e) - 1):
        if e[k] > 0 and e[k - 1] > 0 and e[k + 1] > 0:
            num = np.log(e[k] / e[k + 1])
            den = np.log(e[k - 1] / e[k])
            if den != 0:
                ordres.append(num / den)
    return ordres


def ordre_experimental_sans_solution(historique_x: list[float]) -> list[float]:
    """
    Calcul sans connaître x* (section 3.1.4, cas 2, formule 3.10) :
        α ≈ ln(d_k / d_{k+1}) / ln(d_{k-1} / d_k)
    avec d_k = |x_k - x_{k+1}|.
    """
    d = [abs(historique_x[k] - historique_x[k + 1])
         for k in range(len(historique_x) - 1)]
    ordres = []
    for k in range(1, len(d) - 1):
        if d[k] > 0 and d[k - 1] > 0 and d[k + 1] > 0:
            num = np.log(d[k] / d[k + 1])
            den = np.log(d[k - 1] / d[k])
            if abs(den) > 1e-30:
                ordres.append(num / den)
    return ordres


# ======================================================================
#  4. Newton classique (formule 3.3 — Satz 3.5)
# ======================================================================

def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-14,
    n_max: int = 100,
) -> ResultatNewton:
    """
    Méthode de Newton : x⁺ = x - f(x) / f'(x).

    Convergence quadratique (Satz 3.5) si :
        - f'(x*) ≠ 0 (nullstelle simple)
        - f' Lipschitz-continue (Lemma 3.4)
        - x₀ suffisamment proche de x*
    """
    hist_x = [x0]
    hist_fx = [f(x0)]
    x = x0

    converge = False
    for k in range(1, n_max + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            break
        x_new = x - fx / dfx
        hist_x.append(x_new)
        hist_fx.append(f(x_new))

        if criteres_arrêt(x_new, x, f(x_new), tau1=tol):
            x = x_new
            converge = True
            break
        x = x_new

    return ResultatNewton(
        x=x, iterations=k, converge=converge, methode="Newton",
        historique_x=hist_x, historique_fx=hist_fx,
    )


# ======================================================================
#  5. Sécante (section 3.1.7)
# ======================================================================

def secante(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-14,
    n_max: int = 100,
) -> ResultatNewton:
    """
    Méthode de la sécante : remplace f'(x) par le quotient différence
        f'(x) ≈ (f(x) - f(x_{k-1})) / (x - x_{k-1}).

    Convergence d'ordre φ = (1+√5)/2 ≈ 1.618 (nombre d'or).
    Avantage : une seule évaluation de f par itération (pas de f').
    """
    hist_x = [x0, x1]
    hist_fx = [f(x0), f(x1)]
    x_prev, x = x0, x1

    converge = False
    for k in range(2, n_max + 2):
        fx = f(x)
        f_prev = f(x_prev)
        denom = fx - f_prev
        if denom == 0:
            break
        x_new = x - fx * (x - x_prev) / denom
        hist_x.append(x_new)
        hist_fx.append(f(x_new))

        if criteres_arrêt(x_new, x, f(x_new), tau1=tol):
            x_prev, x = x, x_new
            converge = True
            break
        x_prev, x = x, x_new

    return ResultatNewton(
        x=x, iterations=k - 1, converge=converge, methode="Sécante",
        historique_x=hist_x, historique_fx=hist_fx,
    )


# ======================================================================
#  6. Bissection (section 3.1.6)
# ======================================================================

def bissection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-14,
    n_max: int = 200,
) -> ResultatNewton:
    """
    Méthode de bissection : convergence linéaire garantie si f(a)·f(b) < 0.

    À chaque pas on réduit l'intervalle de moitié. Lent mais sûr :
    convergence globale, pas besoin de dérivée.
    Rate de convergence : log₁₀(2) ≈ 0.301 chiffres par itération.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) et f(b) doivent être de signes opposés.")

    hist_x = [(a + b) / 2]
    hist_fx = [f(hist_x[0])]

    converge = False
    for k in range(1, n_max + 1):
        m = (a + b) / 2
        fm = f(m)
        hist_x.append(m)
        hist_fx.append(fm)

        if abs(b - a) < tol or abs(fm) < tol:
            converge = True
            break
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return ResultatNewton(
        x=(a + b) / 2, iterations=k, converge=converge, methode="Bissection",
        historique_x=hist_x, historique_fx=hist_fx,
    )


# ======================================================================
#  7. Condition du problème nullstelle (section 3.1.2)
# ======================================================================

def condition_nullstelle(df_at_xstar: float) -> float:
    """
    Condition absolue du problème de recherche de zéros :
        cond_abs = 1 / |f'(x*)|.

    Si f'(x*) ≈ 0 (nullstelle multiple) → mauvais conditionnement.
    """
    if df_at_xstar == 0:
        return float("inf")
    return 1.0 / abs(df_at_xstar)


# ======================================================================
#  8. Newton modifié pour nullstelles multiples (section 3.1.8)
# ======================================================================

def newton_modifie(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    multiplicite: int = 2,
    tol: float = 1e-14,
    n_max: int = 100,
) -> ResultatNewton:
    """
    Newton modifié : x⁺ = x - m·f(x)/f'(x) (section 3.1.8).

    Retrouve la convergence quadratique si m est la vraie multiplicité.
    """
    hist_x = [x0]
    hist_fx = [f(x0)]
    x = x0

    converge = False
    for k in range(1, n_max + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            break
        x_new = x - multiplicite * fx / dfx
        hist_x.append(x_new)
        hist_fx.append(f(x_new))

        if criteres_arrêt(x_new, x, f(x_new), tau1=tol):
            x = x_new
            converge = True
            break
        x = x_new

    return ResultatNewton(
        x=x, iterations=k, converge=converge, methode=f"Newton modifié (m={multiplicite})",
        historique_x=hist_x, historique_fx=hist_fx,
    )


# ======================================================================
#  9. Tracés
# ======================================================================

def tracer_convergence(
    resultats: list[ResultatNewton],
    x_star: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trace |x_k - x*| ou |f(x_k)| en semi-log."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for r in resultats:
        if x_star is not None:
            erreurs = [abs(xi - x_star) for xi in r.historique_x]
            ylabel = "$|x_k - x^*|$"
        else:
            erreurs = [abs(fxi) for fxi in r.historique_fx]
            ylabel = "$|f(x_k)|$"
        ax.semilogy(erreurs, "o-", label=r.methode, markersize=4)

    ax.set_xlabel("itération $k$")
    ax.set_ylabel(ylabel)
    ax.set_title("Convergence comparée")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_newton_graphique(
    f: Callable[[float], float],
    historique_x: list[float],
    intervalle: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    n_steps: int = 4,
) -> plt.Axes:
    """
    Illustration graphique du Newton (comme la Figure 3.2 du script) :
    tracé de la tangente à chaque pas.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if intervalle is None:
        xmin = min(historique_x[:n_steps + 1]) - 0.5
        xmax = max(historique_x[:n_steps + 1]) + 0.5
    else:
        xmin, xmax = intervalle

    xs = np.linspace(xmin, xmax, 300)
    ax.plot(xs, [f(x) for x in xs], "k-", linewidth=2, label="$f(x)$")
    ax.axhline(0, color="grey", linewidth=0.5)

    couleurs = plt.cm.viridis(np.linspace(0.1, 0.9, n_steps))
    for i in range(min(n_steps, len(historique_x) - 1)):
        xi = historique_x[i]
        fi = f(xi)
        xi1 = historique_x[i + 1]
        # Tangente : t(z) = f(xi) + f'(xi)·(z - xi), passe par (xi1, 0)
        ax.plot([xi, xi1], [fi, 0], "o--", color=couleurs[i],
                label=f"$x_{i}={xi:.4f}$", markersize=6)
        ax.plot(xi, fi, "o", color=couleurs[i], markersize=8)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("Illustration du Newton (Figure 3.2 du script)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def tracer_ordres_experimentaux(
    resultats: list[ResultatNewton],
    x_star: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trace l'ordre de convergence expérimental à chaque itération."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for r in resultats:
        if x_star is not None:
            ordres = ordre_experimental_exact(r.historique_x, x_star)
        else:
            ordres = ordre_experimental_sans_solution(r.historique_x)
        if ordres:
            ax.plot(range(len(ordres)), ordres, "o-", label=r.methode, markersize=5)

    ax.axhline(2, color="red", linestyle="--", alpha=0.5, label="quadratique (α=2)")
    ax.axhline(1.618, color="green", linestyle=":", alpha=0.5, label="sécante (α≈1.618)")
    ax.axhline(1, color="blue", linestyle=":", alpha=0.5, label="linéaire (α=1)")
    ax.set_xlabel("indice $k$")
    ax.set_ylabel("ordre expérimental $\\alpha$")
    ax.set_title("Section 3.1.4 — ordre de convergence expérimental")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3.5)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    # -- Übung 3.1 : f(x) = x² - 3, x0 = 4.5 --
    f = lambda x: x**2 - 3
    df = lambda x: 2 * x
    x_star = np.sqrt(3)

    print("=== Übung 3.1 : f(x) = x² - 3, x₀ = 4.5 ===")
    res = newton(f, df, 4.5)
    print(res)
    print(f"  x* exact = {x_star:.15f}")
    ordres = ordre_experimental_exact(res.historique_x, x_star)
    print(f"  ordres expérimentaux : {[f'{o:.3f}' for o in ordres]}")

    # -- Comparaison Newton / Sécante / Bissection --
    print("\n=== Comparaison sur f(x) = x³ - 2x - 5 ===")
    f2 = lambda x: x**3 - 2 * x - 5
    df2 = lambda x: 3 * x**2 - 2
    x_star2 = 2.0945514815423265  # racine par Newton haute précision

    res_n = newton(f2, df2, 3.0)
    res_s = secante(f2, 2.0, 3.0)
    res_b = bissection(f2, 2.0, 3.0)

    for r in [res_n, res_s, res_b]:
        print(f"  {r}")

    # -- Nullstelle multiple : f(x) = x³ --
    print("\n=== Nullstelle multiple : f(x) = x³ ===")
    f3 = lambda x: x**3
    df3 = lambda x: 3 * x**2
    res_std = newton(f3, df3, 1.0, tol=1e-10)
    res_mod = newton_modifie(f3, df3, 1.0, multiplicite=3, tol=1e-10)
    print(f"  Newton standard : {res_std.iterations} itérations")
    print(f"  Newton modifié  : {res_mod.iterations} itérations (retrouve α ≈ 2)")

    # -- Tracés --
    print("\n=== Tracés ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tracer_newton_graphique(f, res.historique_x, ax=axes[0])
    tracer_convergence([res_n, res_s, res_b], x_star=x_star2, ax=axes[1])
    tracer_ordres_experimentaux([res_n, res_s], x_star=x_star2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("newton_scalar_demo.png", dpi=120)
    print("Figure sauvegardée : newton_scalar_demo.png")
