"""
probability_basics.py
=====================

Fondements de la théorie des probabilités.

Couvre :
    - Espace probabilisé (Ω, A, P)
    - Axiomes de Kolmogorov
    - Modèle de Laplace : P(A) = |A|/|Ω|
    - Théorème de Bayes : P(A|B) = P(B|A)·P(A)/P(B)
    - Formule des probabilités totales
    - Exemples : dés, cartes, urnes

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from math import comb
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Axiomes et calcul
# ======================================================================

def laplace(favorables: int, total: int) -> float:
    """P(A) = |A| / |Ω| (équiprobabilité)."""
    return favorables / total


def complement(p: float) -> float:
    """P(Ā) = 1 - P(A)."""
    return 1 - p


def union(pA: float, pB: float, pAB: float) -> float:
    """P(A ∪ B) = P(A) + P(B) - P(A ∩ B)."""
    return pA + pB - pAB


def conditionnelle(pAB: float, pB: float) -> float:
    """P(A|B) = P(A ∩ B) / P(B)."""
    return pAB / pB if pB > 0 else 0


def bayes(pBA: float, pA: float, pB: float) -> float:
    """P(A|B) = P(B|A)·P(A) / P(B)."""
    return pBA * pA / pB if pB > 0 else 0


def probabilites_totales(pB_given_Ai: list[float], pAi: list[float]) -> float:
    """P(B) = Σ P(B|Aᵢ)·P(Aᵢ)."""
    return sum(pb * pa for pb, pa in zip(pB_given_Ai, pAi))


def sont_independants(pA: float, pB: float, pAB: float, tol: float = 1e-10) -> bool:
    """A et B indépendants ssi P(A ∩ B) = P(A)·P(B)."""
    return abs(pAB - pA * pB) < tol


# ======================================================================
#  2. Exemples classiques
# ======================================================================

def exemple_des():
    """Probabilités avec deux dés."""
    print("=== Deux dés (6 faces) ===\n")
    omega = [(i, j) for i in range(1, 7) for j in range(1, 7)]
    n = len(omega)

    # Somme = 7
    A = [(i, j) for i, j in omega if i + j == 7]
    print(f"  P(somme = 7) = {len(A)}/{n} = {laplace(len(A), n):.4f}")

    # Au moins un 6
    B = [(i, j) for i, j in omega if i == 6 or j == 6]
    print(f"  P(au moins un 6) = {len(B)}/{n} = {laplace(len(B), n):.4f}")
    print(f"  P(aucun 6) = 1 - P(≥1 six) = {complement(laplace(len(B), n)):.4f} = (5/6)² = {(5/6)**2:.4f}")

    # Double
    C = [(i, j) for i, j in omega if i == j]
    print(f"  P(double) = {len(C)}/{n} = {laplace(len(C), n):.4f}")


def exemple_urne():
    """Tirage d'une urne — avec et sans remise."""
    print("\n=== Urne : 5 rouges, 3 bleues ===\n")
    R, B, total = 5, 3, 8

    # Sans remise : 2 tirages
    p_RR = (R/total) * ((R-1)/(total-1))
    p_RB = (R/total) * (B/(total-1))
    print(f"  Sans remise :")
    print(f"    P(R,R) = {p_RR:.4f}")
    print(f"    P(R,B) = {p_RB:.4f}")

    # Avec remise
    p_RR_r = (R/total)**2
    print(f"  Avec remise :")
    print(f"    P(R,R) = {p_RR_r:.4f}")


def exemple_bayes():
    """Problème classique de Bayes : test médical."""
    print("\n=== Bayes : test médical ===\n")
    # Prévalence, sensibilité, spécificité
    p_malade = 0.001
    sensibilite = 0.99   # P(+|malade)
    specificite = 0.95    # P(-|sain)

    p_sain = 1 - p_malade
    p_pos_sain = 1 - specificite  # faux positif
    p_pos = probabilites_totales([sensibilite, p_pos_sain], [p_malade, p_sain])
    p_malade_pos = bayes(sensibilite, p_malade, p_pos)

    print(f"  Prévalence     = {p_malade*100:.1f}%")
    print(f"  Sensibilité    = {sensibilite*100:.0f}%")
    print(f"  Spécificité    = {specificite*100:.0f}%")
    print(f"  P(+ | malade)  = {sensibilite}")
    print(f"  P(+ | sain)    = {p_pos_sain}")
    print(f"  P(+)           = {p_pos:.5f}")
    print(f"  P(malade | +)  = {p_malade_pos:.4f} = {p_malade_pos*100:.1f}%")
    print(f"  → Même avec un bon test, la plupart des positifs sont des faux positifs !")


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_somme_des(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    sommes = range(2, 13)
    omega = [(i, j) for i in range(1, 7) for j in range(1, 7)]
    probs = [sum(1 for i, j in omega if i+j == s) / 36 for s in sommes]

    ax.bar(sommes, probs, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("somme $S$"); ax.set_ylabel("$P(S)$")
    ax.set_title("Distribution de la somme de deux dés")
    ax.set_xticks(list(sommes))
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def tracer_bayes_prevalence(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre que P(malade|+) dépend fortement de la prévalence."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    prevalences = np.logspace(-4, -1, 200)
    sens, spec = 0.99, 0.95

    ppv = []
    for prev in prevalences:
        p_pos = sens * prev + (1-spec) * (1-prev)
        ppv.append(sens * prev / p_pos)

    ax.semilogx(prevalences * 100, [p*100 for p in ppv], "b-", linewidth=2)
    ax.set_xlabel("prévalence (%)"); ax.set_ylabel("P(malade | test +) (%)")
    ax.set_title("Paradoxe de Bayes : le test dépend de la prévalence")
    ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    exemple_des()
    exemple_urne()
    exemple_bayes()

    print(f"\n=== Indépendance ===\n")
    # Dé : A = {pair}, B = {≤3}
    pA = 3/6; pB = 3/6; pAB = 1/6  # {2}... non: {2} seulement? Non: {2} a prob 1/6
    # A∩B = {2} → P = 1/6, P(A)P(B) = 1/4 → pas indépendants
    # Correction: A = {2,4,6}, B = {1,2,3}, A∩B = {2} → P(A∩B) = 1/6, P(A)P(B) = 9/36 = 1/4
    print(f"  Dé : A={{pair}}, B={{≤3}}")
    print(f"  P(A) = 1/2, P(B) = 1/2, P(A∩B) = P({{2}}) = 1/6")
    print(f"  P(A)·P(B) = 1/4 ≠ 1/6 → pas indépendants")
    print(f"  Indépendants ? {sont_independants(1/2, 1/2, 1/6)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_somme_des(ax=axes[0])
    tracer_bayes_prevalence(ax=axes[1])
    plt.tight_layout()
    plt.savefig("probability_basics_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
