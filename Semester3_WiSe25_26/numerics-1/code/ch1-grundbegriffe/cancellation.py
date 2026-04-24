"""
cancellation.py
===============

Étude expérimentale du phénomène d'Auslöschung (cancellation catastrophique)
et reformulations stables.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", sections 1.3.3 et 1.4.

Le phénomène : quand on soustrait deux nombres flottants très proches,
les chiffres significatifs identiques s'annulent et l'erreur relative
explose. La condition du problème "addition" devient :

    cond_rel,∞ = (|x_1| + |x_2|) / |x_1 + x_2|

qui tend vers l'infini quand x_2 ≈ -x_1.

Quatre exemples canoniques sont implémentés :
    1. f(x) = sqrt(x^2 + 1) - x          (Beispiel 1.13 du script)
    2. f(x) = (1 - cos(x)) / x^2         (limite vers 1/2 en x=0)
    3. dérivée numérique par différence finie progressive
    4. variance par la formule "naïve" E[X^2] - E[X]^2

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
#  Constantes
# ----------------------------------------------------------------------

#: Précision machine pour float64 (IEEE 754 double, voir Beispiel 1.12).
EPS_MACH: float = float(np.finfo(np.float64).eps)


# ----------------------------------------------------------------------
#  1. Outil générique : conditionnement de l'addition
# ----------------------------------------------------------------------

def condition_addition(x1: float, x2: float) -> float:
    """
    Condition relative de l'addition x1 + x2, formule (1.2) du script.

    cond_rel,∞ = (|x1| + |x2|) / |x1 + x2|.

    - Si x1 et x2 ont le même signe : cond = 1 (problème bien conditionné).
    - Si x2 ≈ -x1 : cond → ∞ (Auslöschung).
    """
    s = x1 + x2
    if s == 0:
        return float("inf")
    return (abs(x1) + abs(x2)) / abs(s)


# ----------------------------------------------------------------------
#  2. Beispiel 1.13 du script : f(x) = sqrt(x^2 + 1) - x
# ----------------------------------------------------------------------

def f_naive(x: float) -> float:
    """
    Algorithme naïf : f(x) = sqrt(x^2 + 1) - x.

    Instable pour x >> 1 : la dernière soustraction est une Auslöschung
    car sqrt(x^2 + 1) ≈ x dans ce régime.
    """
    return np.sqrt(x * x + 1.0) - x


def f_stable(x: float) -> float:
    """
    Reformulation stable :
        sqrt(x^2 + 1) - x  =  1 / (sqrt(x^2 + 1) + x).

    On a multiplié haut et bas par le conjugué. Plus de soustraction —
    pas d'Auslöschung. Mathématiquement équivalent à `f_naive`.
    """
    return 1.0 / (np.sqrt(x * x + 1.0) + x)


# ----------------------------------------------------------------------
#  3. Autre exemple : f(x) = (1 - cos(x)) / x^2
# ----------------------------------------------------------------------

def g_naive(x: float) -> float:
    """
    Algorithme naïf : g(x) = (1 - cos(x)) / x^2.

    Instable près de 0 : cos(x) ≈ 1 donc 1 - cos(x) souffre d'Auslöschung.
    Limite mathématique en 0 : 1/2.
    """
    return (1.0 - np.cos(x)) / (x * x)


def g_stable(x: float) -> float:
    """
    Reformulation stable via l'identité trigonométrique :
        1 - cos(x) = 2 sin^2(x/2),
    donc g(x) = 2 sin^2(x/2) / x^2 = (sin(x/2) / (x/2))^2 / 2.

    Plus aucune soustraction de termes proches.
    """
    s = np.sin(x / 2.0)
    return 2.0 * (s * s) / (x * x)


# ----------------------------------------------------------------------
#  4. Dérivée numérique : différence finie progressive
# ----------------------------------------------------------------------

def derivee_progressive(f: Callable[[float], float], x: float, h: float) -> float:
    """
    Approximation D_+ f(x) ≈ (f(x+h) - f(x)) / h.

    Compromis classique :
    - h trop grand : erreur de troncature O(h)
    - h trop petit : Auslöschung dans (f(x+h) - f(x))

    Le minimum de l'erreur totale est atteint vers h ≈ sqrt(EPS_MACH).
    """
    return (f(x + h) - f(x)) / h


# ----------------------------------------------------------------------
#  5. Variance : formule naïve vs formule à deux passes
# ----------------------------------------------------------------------

def variance_naive(x: np.ndarray) -> float:
    """
    Formule "naïve" :  Var = E[X^2] - E[X]^2.

    Mathématiquement correcte mais numériquement catastrophique quand
    E[X^2] ≈ E[X]^2 (données fortement décalées par rapport à 0).
    """
    n = len(x)
    moy_carres = float(np.sum(x * x)) / n
    moy = float(np.sum(x)) / n
    return moy_carres - moy * moy


def variance_deux_passes(x: np.ndarray) -> float:
    """
    Formule à deux passes : Var = (1/n) * Σ (x_i - moyenne)^2.

    Stable : on soustrait la moyenne avant d'élever au carré, donc
    pas d'Auslöschung de grandes valeurs.
    """
    moy = float(np.mean(x))
    return float(np.mean((x - moy) ** 2))


# ----------------------------------------------------------------------
#  6. Outils d'analyse d'erreur
# ----------------------------------------------------------------------

@dataclass
class ErreurRelative:
    """Petit récap d'une erreur relative à un point donné."""
    x: float
    valeur_calculee: float
    valeur_reference: float

    @property
    def erreur(self) -> float:
        if self.valeur_reference == 0:
            return abs(self.valeur_calculee)
        return abs(self.valeur_calculee - self.valeur_reference) / abs(
            self.valeur_reference
        )


def erreur_relative_array(
    f_test: Callable[[float], float],
    f_ref: Callable[[float], float],
    xs: np.ndarray,
) -> np.ndarray:
    """
    Erreur relative point par point entre `f_test` et `f_ref`,
    en évitant la division par zéro.
    """
    out = np.empty_like(xs, dtype=float)
    for i, x in enumerate(xs):
        ref = f_ref(x)
        test = f_test(x)
        if ref == 0:
            out[i] = abs(test)
        else:
            out[i] = abs(test - ref) / abs(ref)
    return out


# ----------------------------------------------------------------------
#  7. Tracés
# ----------------------------------------------------------------------

def tracer_beispiel_1_13(ax: plt.Axes | None = None) -> plt.Axes:
    """
    Trace l'erreur relative pour f(x) = sqrt(x^2+1) - x sur [10, 10^8],
    formule naïve vs formule stable.

    La référence est calculée en haute précision (numpy float128 si
    disponible, sinon la formule stable elle-même qui est essentiellement
    exacte).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    xs = np.logspace(1, 8, 200)

    # Référence : formule stable (essentiellement exacte ici)
    ref = np.array([f_stable(x) for x in xs])
    naive = np.array([f_naive(x) for x in xs])

    err_naive = np.abs(naive - ref) / np.abs(ref)
    # L'erreur de la formule stable contre elle-même = 0, on prend
    # plutôt epsilon machine comme floor pour comparaison visuelle.
    err_stable = np.full_like(err_naive, EPS_MACH)

    ax.loglog(xs, err_naive, "r-", linewidth=2, label="Naïf : √(x²+1) − x")
    ax.loglog(xs, err_stable, "g--", linewidth=2, label="Stable : 1 / (√(x²+1) + x)")
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5, label="100 % d'erreur")
    ax.set_xlabel("x")
    ax.set_ylabel("erreur relative")
    ax.set_title("Beispiel 1.13 — Auslöschung dans √(x²+1) − x")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_derivee_numerique(ax: plt.Axes | None = None) -> plt.Axes:
    """
    Trace l'erreur de la dérivée numérique pour f(x) = sin(x) en x = 1
    en fonction de h, sur 16 ordres de grandeur.

    On voit la fameuse forme en V :
        - branche descendante : erreur de troncature O(h)
        - branche montante : Auslöschung qui domine
        - minimum vers h ≈ sqrt(EPS_MACH) ≈ 1.5e-8.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x0 = 1.0
    derivee_exacte = np.cos(x0)  # car d/dx sin(x) = cos(x)
    hs = np.logspace(-16, 0, 100)

    erreurs = np.array(
        [abs(derivee_progressive(np.sin, x0, h) - derivee_exacte) for h in hs]
    )

    ax.loglog(hs, erreurs, "b-", linewidth=2, label="erreur observée")
    h_opt = np.sqrt(EPS_MACH)
    ax.axvline(
        h_opt, color="red", linestyle="--", alpha=0.7,
        label=f"h optimal ≈ √ε_mach ≈ {h_opt:.1e}",
    )
    ax.set_xlabel("h (pas de discrétisation)")
    ax.set_ylabel("erreur absolue |D₊f(x) − f'(x)|")
    ax.set_title("Dérivée numérique de sin en x=1 : compromis troncature / Auslöschung")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_variance(ax: plt.Axes | None = None) -> plt.Axes:
    """
    Compare la variance "naïve" et "deux passes" sur des données décalées
    de plus en plus loin de zéro.

    Données : x_i = decalage + bruit gaussien d'écart-type 1.
    Variance théorique : 1.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(0)
    base = rng.standard_normal(10_000)  # variance ≈ 1
    decalages = np.logspace(0, 9, 40)

    err_naive = []
    err_2pass = []
    for d in decalages:
        x = base + d
        err_naive.append(abs(variance_naive(x) - 1.0))
        err_2pass.append(abs(variance_deux_passes(x) - 1.0))

    ax.loglog(decalages, err_naive, "r-", linewidth=2, label="Naïf : E[X²] − E[X]²")
    ax.loglog(decalages, err_2pass, "g--", linewidth=2, label="Deux passes")
    ax.set_xlabel("décalage des données")
    ax.set_ylabel("erreur absolue sur la variance (≈ 1 attendue)")
    ax.set_title("Variance : Auslöschung dans E[X²] − E[X]²")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


# ----------------------------------------------------------------------
#  Démo si exécuté directement
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== Précision machine ε_mach = {EPS_MACH:.3e} (cf. Beispiel 1.12) ===\n")

    print("--- Conditionnement de l'addition (Eq. 1.2) ---")
    cas = [
        ("même signe", 1.0, 2.0),
        ("opposés très différents", 1.0, -0.5),
        ("Auslöschung légère", 1.0, -0.99),
        ("Auslöschung forte", 1.0, -0.999_999),
        ("Auslöschung extrême", 1.0, -0.999_999_999_999),
    ]
    for nom, a, b in cas:
        print(f"  {nom:30s} cond = {condition_addition(a, b):.3e}")

    print("\n--- Beispiel 1.13 : √(x²+1) − x pour x = 10^k ---")
    print(f"{'x':>10} | {'naïf':>22} | {'stable':>22} | {'err. rel. naïf':>14}")
    print("-" * 78)
    for k in range(1, 9):
        x = 10.0**k
        a = f_naive(x)
        b = f_stable(x)
        err = abs(a - b) / abs(b) if b != 0 else float("nan")
        print(f"{x:>10.0e} | {a:>22.15e} | {b:>22.15e} | {err:>14.3e}")

    print("\n--- Variance : données décalées (vraie variance = 1) ---")
    rng = np.random.default_rng(0)
    base = rng.standard_normal(10_000)
    print(f"{'décalage':>12} | {'naïf':>14} | {'deux passes':>14}")
    print("-" * 50)
    for d in [1, 1e3, 1e6, 1e9]:
        x = base + d
        print(f"{d:>12.0e} | {variance_naive(x):>14.6f} | {variance_deux_passes(x):>14.6f}")

    print("\n--- Tracés ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_beispiel_1_13(axes[0])
    tracer_derivee_numerique(axes[1])
    tracer_variance(axes[2])
    plt.tight_layout()
    plt.savefig("cancellation_demo.png", dpi=120)
    print("Figure sauvegardée : cancellation_demo.png")
