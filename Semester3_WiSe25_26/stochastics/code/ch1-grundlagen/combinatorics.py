"""
combinatorics.py
================

Combinatoire : dénombrement et coefficients binomiaux.

Couvre :
    - Factorielle, permutations, arrangements
    - Combinaisons C(n,k) = n! / (k!(n-k)!)
    - Triangle de Pascal
    - Formule du binôme : (a+b)^n = Σ C(n,k) a^k b^{n-k}
    - Principe d'inclusion-exclusion
    - Problème du tiroir (Schubfachprinzip)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from math import factorial, comb, perm

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. From-scratch
# ======================================================================

def factorielle(n: int) -> int:
    """n! = 1·2·3·...·n. 0! = 1."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def permutations_count(n: int) -> int:
    """Nombre de permutations de n éléments : n!"""
    return factorielle(n)


def arrangements_count(n: int, k: int) -> int:
    """A(n,k) = n! / (n-k)! = nombre de k-uplets ordonnés parmi n."""
    return factorielle(n) // factorielle(n - k)


def combinaisons_count(n: int, k: int) -> int:
    """C(n,k) = n! / (k!(n-k)!) = nombre de sous-ensembles de taille k."""
    if k < 0 or k > n:
        return 0
    return factorielle(n) // (factorielle(k) * factorielle(n - k))


def combinaisons_repetition(n: int, k: int) -> int:
    """C_rep(n,k) = C(n+k-1, k) = tirages avec remise, ordre sans importance."""
    return combinaisons_count(n + k - 1, k)


# ======================================================================
#  2. Triangle de Pascal
# ======================================================================

def triangle_pascal(n_rows: int) -> list[list[int]]:
    """Construit le triangle de Pascal jusqu'à la ligne n."""
    triangle = [[1]]
    for i in range(1, n_rows):
        prev = triangle[-1]
        row = [1]
        for j in range(1, i):
            row.append(prev[j-1] + prev[j])
        row.append(1)
        triangle.append(row)
    return triangle


def afficher_pascal(n: int) -> None:
    """Affiche le triangle de Pascal formaté."""
    tri = triangle_pascal(n)
    width = len(str(tri[-1][len(tri[-1])//2])) + 1
    for i, row in enumerate(tri):
        padding = " " * ((n - i) * width // 2)
        print(padding + " ".join(f"{x:>{width}}" for x in row))


# ======================================================================
#  3. Formule du binôme
# ======================================================================

def binome_expansion(a: float, b: float, n: int) -> float:
    """(a+b)^n = Σ C(n,k) a^k b^{n-k}."""
    return sum(combinaisons_count(n, k) * a**k * b**(n-k) for k in range(n+1))


def binome_coefficients(n: int) -> list[int]:
    """Renvoie [C(n,0), C(n,1), ..., C(n,n)]."""
    return [combinaisons_count(n, k) for k in range(n+1)]


# ======================================================================
#  4. Inclusion-Exclusion
# ======================================================================

def inclusion_exclusion_2(A: int, B: int, A_inter_B: int) -> int:
    """|A ∪ B| = |A| + |B| - |A ∩ B|."""
    return A + B - A_inter_B


def inclusion_exclusion_3(A: int, B: int, C: int,
                           AB: int, AC: int, BC: int, ABC: int) -> int:
    """|A ∪ B ∪ C| = |A|+|B|+|C| - |AB| - |AC| - |BC| + |ABC|."""
    return A + B + C - AB - AC - BC + ABC


def derangements(n: int) -> int:
    """
    D_n = nombre de permutations sans point fixe (Subfakultät).
    D_n = n! · Σ_{k=0}^n (-1)^k / k!
    """
    return round(factorielle(n) * sum((-1)**k / factorielle(k) for k in range(n+1)))


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_binomial_coefficients(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for n in [5, 10, 15, 20]:
        coeffs = binome_coefficients(n)
        k = range(n + 1)
        ax.plot(k, coeffs, "o-", markersize=3, linewidth=1.5, label=f"$n = {n}$")

    ax.set_xlabel("$k$"); ax.set_ylabel("$C(n, k)$")
    ax.set_title("Coefficients binomiaux $C(n, k)$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_pascal_heatmap(n: int = 15, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    tri = triangle_pascal(n)
    # Matrice
    M = np.zeros((n, n))
    for i, row in enumerate(tri):
        for j, val in enumerate(row):
            M[i, j] = val

    ax.imshow(np.log1p(M), cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("$k$"); ax.set_ylabel("$n$")
    ax.set_title(f"Triangle de Pascal (log, $n = {n}$)")
    return ax


def tracer_derangements(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ns = range(1, 15)
    D = [derangements(n) for n in ns]
    N = [factorielle(n) for n in ns]
    ratio = [d/n for d, n in zip(D, N)]

    ax.plot(list(ns), ratio, "bo-", markersize=5, linewidth=2)
    ax.axhline(1/np.e, color="red", linestyle="--", label=f"$1/e \\approx {1/np.e:.4f}$")
    ax.set_xlabel("$n$"); ax.set_ylabel("$D_n / n!$")
    ax.set_title("Ratio dérangements / permutations → $1/e$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Dénombrement ===\n")
    print(f"  10! = {factorielle(10)}")
    print(f"  A(10,3) = {arrangements_count(10, 3)}")
    print(f"  C(10,3) = {combinaisons_count(10, 3)}")
    print(f"  C_rep(5,3) = {combinaisons_repetition(5, 3)}")

    print(f"\n  Vérif : C(10,3) = {comb(10,3)} (math.comb) ✓")

    print(f"\n=== Triangle de Pascal (8 lignes) ===\n")
    afficher_pascal(8)

    print(f"\n=== Formule du binôme ===\n")
    for n in [2, 5, 10]:
        direct = (3 + 2)**n
        binome = binome_expansion(3, 2, n)
        print(f"  (3+2)^{n} = {direct}, binôme = {int(binome)} ✓")

    print(f"\n=== Inclusion-Exclusion ===\n")
    print(f"  |A|=30, |B|=20, |A∩B|=5 → |A∪B| = {inclusion_exclusion_2(30, 20, 5)}")

    print(f"\n=== Dérangements ===\n")
    for n in range(1, 8):
        print(f"  D_{n} = {derangements(n):>6} / {factorielle(n):>6} = {derangements(n)/factorielle(n):.4f}")
    print(f"  → D_n/n! → 1/e = {1/np.e:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_binomial_coefficients(ax=axes[0])
    tracer_pascal_heatmap(ax=axes[1])
    tracer_derangements(ax=axes[2])
    plt.tight_layout()
    plt.savefig("combinatorics_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
