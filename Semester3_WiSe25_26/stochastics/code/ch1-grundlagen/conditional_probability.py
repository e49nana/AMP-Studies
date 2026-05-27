"""
conditional_probability.py
==========================

Probabilités conditionnelles, indépendance, arbres.

Couvre :
    - P(A|B) = P(A∩B)/P(B)
    - Indépendance : P(A∩B) = P(A)·P(B)
    - Arbres de probabilités (computation et affichage)
    - Problème de Monty Hall (simulation)
    - Paradoxe des anniversaires
    - Chaîne de Bayes itéré (mise à jour bayésienne)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from math import comb, factorial

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Monty Hall
# ======================================================================

def monty_hall_simulation(n_trials: int = 100_000) -> dict:
    """
    Problème de Monty Hall :
        3 portes, 1 voiture, 2 chèvres.
        Vous choisissez, l'animateur ouvre une chèvre, vous pouvez changer.
    Résultat : changer gagne 2/3 du temps.
    """
    rng = np.random.default_rng(42)
    voitures = rng.integers(0, 3, n_trials)
    choix = rng.integers(0, 3, n_trials)

    victoires_garder = np.sum(choix == voitures)
    victoires_changer = n_trials - victoires_garder  # changer gagne ssi le choix initial était faux

    return {
        "garder": victoires_garder / n_trials,
        "changer": victoires_changer / n_trials,
        "n_trials": n_trials,
    }


# ======================================================================
#  2. Paradoxe des anniversaires
# ======================================================================

def prob_anniversaire_unique(n: int) -> float:
    """P(tous différents) = 365/365 · 364/365 · ... · (366-n)/365."""
    if n > 365:
        return 0
    p = 1.0
    for k in range(n):
        p *= (365 - k) / 365
    return p


def prob_anniversaire_commun(n: int) -> float:
    """P(au moins 2 même anniversaire) = 1 - P(tous différents)."""
    return 1 - prob_anniversaire_unique(n)


def seuil_anniversaire(p_cible: float = 0.5) -> int:
    """Plus petit n tel que P(collision) ≥ p_cible."""
    for n in range(1, 366):
        if prob_anniversaire_commun(n) >= p_cible:
            return n
    return 365


# ======================================================================
#  3. Mise à jour bayésienne
# ======================================================================

def mise_a_jour_bayes(
    prior: float, likelihood: float, likelihood_complement: float,
) -> float:
    """
    posterior = P(H|D) = P(D|H)·P(H) / P(D).
    P(D) = P(D|H)·P(H) + P(D|¬H)·P(¬H).
    """
    p_data = likelihood * prior + likelihood_complement * (1 - prior)
    return likelihood * prior / p_data


def bayes_iteratif(
    prior: float, observations: list[tuple[float, float]],
) -> list[float]:
    """
    Met à jour le prior avec une série d'observations.
    Chaque observation = (P(D|H), P(D|¬H)).
    """
    posteriors = [prior]
    current = prior
    for lik_H, lik_notH in observations:
        current = mise_a_jour_bayes(current, lik_H, lik_notH)
        posteriors.append(current)
    return posteriors


# ======================================================================
#  4. Arbres de probabilités
# ======================================================================

def arbre_2_niveaux(
    p_A: float, p_B_given_A: float, p_B_given_notA: float,
) -> dict:
    """Calcule toutes les probabilités d'un arbre à 2 niveaux."""
    p_notA = 1 - p_A
    p_B = p_B_given_A * p_A + p_B_given_notA * p_notA
    p_A_given_B = p_B_given_A * p_A / p_B if p_B > 0 else 0

    return {
        "P(A)": p_A,
        "P(¬A)": p_notA,
        "P(B|A)": p_B_given_A,
        "P(B|¬A)": p_B_given_notA,
        "P(A∩B)": p_A * p_B_given_A,
        "P(¬A∩B)": p_notA * p_B_given_notA,
        "P(B)": p_B,
        "P(A|B)": p_A_given_B,
    }


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_anniversaire(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ns = range(1, 80)
    probs = [prob_anniversaire_commun(n) for n in ns]
    n50 = seuil_anniversaire(0.5)

    ax.plot(ns, probs, "b-", linewidth=2)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.axvline(n50, color="green", linestyle=":", alpha=0.5,
                label=f"$n = {n50}$ pour $P \\geq 50\\%$")
    ax.set_xlabel("nombre de personnes $n$")
    ax.set_ylabel("$P$(au moins 2 même anniversaire)")
    ax.set_title("Paradoxe des anniversaires")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_monty_hall(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ns = [100, 500, 1000, 5000, 10000, 50000, 100000]
    garder = []
    changer = []
    for n in ns:
        r = monty_hall_simulation(n)
        garder.append(r["garder"])
        changer.append(r["changer"])

    ax.semilogx(ns, garder, "rs-", markersize=5, label="garder")
    ax.semilogx(ns, changer, "go-", markersize=5, label="changer")
    ax.axhline(1/3, color="red", linestyle=":", alpha=0.3)
    ax.axhline(2/3, color="green", linestyle=":", alpha=0.3)
    ax.set_xlabel("nombre de simulations"); ax.set_ylabel("taux de victoire")
    ax.set_title("Monty Hall : changer gagne 2/3 du temps")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_bayes_iteratif(ax: plt.Axes | None = None) -> plt.Axes:
    """Mise à jour bayésienne avec des observations successives."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Pièce biaisée ? Prior = 50%, chaque lancer = observation
    prior = 0.5
    # H = pièce biaisée (P(face) = 0.8), ¬H = pièce juste (P(face) = 0.5)
    observations = [(0.8, 0.5)] * 7 + [(0.2, 0.5)] * 3  # 7 face, 3 pile
    posteriors = bayes_iteratif(prior, observations)

    ax.plot(posteriors, "bo-", markersize=6, linewidth=2)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.3)
    ax.set_xlabel("observation #"); ax.set_ylabel("$P$(biaisée | données)")
    ax.set_title("Mise à jour bayésienne (7 faces, 3 piles)")

    # Annoter
    for i, (lik, _) in enumerate(observations):
        ax.annotate("F" if lik > 0.5 else "P", (i+1, posteriors[i+1]),
                    textcoords="offset points", xytext=(0, 10), fontsize=8,
                    ha="center", color="green" if lik > 0.5 else "red")

    ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Monty Hall ===\n")
    r = monty_hall_simulation()
    print(f"  Garder  : {r['garder']:.4f} (théo: 1/3 = {1/3:.4f})")
    print(f"  Changer : {r['changer']:.4f} (théo: 2/3 = {2/3:.4f})")

    print(f"\n=== Paradoxe des anniversaires ===\n")
    for n in [10, 23, 30, 50, 70]:
        print(f"  n={n:>2} : P(collision) = {prob_anniversaire_commun(n):.4f}")
    print(f"  Seuil 50% : n = {seuil_anniversaire(0.5)}")
    print(f"  Seuil 99% : n = {seuil_anniversaire(0.99)}")

    print(f"\n=== Arbre de probabilités ===\n")
    # Exemple : test COVID, prévalence 5%, sensibilité 95%, spécificité 90%
    arbre = arbre_2_niveaux(0.05, 0.95, 0.10)
    for k, v in arbre.items():
        print(f"  {k:12s} = {v:.4f}")

    print(f"\n=== Mise à jour bayésienne ===\n")
    posteriors = bayes_iteratif(0.5, [(0.8, 0.5)]*7 + [(0.2, 0.5)]*3)
    for i, p in enumerate(posteriors):
        print(f"  Après {i} obs : P(biaisée) = {p:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_anniversaire(ax=axes[0])
    tracer_monty_hall(ax=axes[1])
    tracer_bayes_iteratif(ax=axes[2])
    plt.tight_layout()
    plt.savefig("conditional_probability_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
