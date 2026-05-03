"""
determinant.py
==============

Calcul du déterminant par différentes méthodes.

Couvre :
    - Formule de Leibniz (somme sur les permutations, O(n!))
    - Règle de Sarrus (3×3 uniquement)
    - Développement de Laplace par cofacteurs (récursif, O(n!))
    - Par échelonnement (Gauss, O(n³)) — la méthode efficace
    - Comparaison des coûts et résultats

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np


# ======================================================================
#  1. Leibniz (O(n!))
# ======================================================================

def signe_permutation(perm: tuple[int, ...]) -> int:
    """Signe d'une permutation : (-1)^{nombre d'inversions}."""
    n = len(perm)
    inversions = sum(1 for i in range(n) for j in range(i+1, n) if perm[i] > perm[j])
    return 1 if inversions % 2 == 0 else -1


def det_leibniz(A: np.ndarray) -> float:
    """
    Formule de Leibniz :
        det(A) = Σ_{σ ∈ Sₙ} sign(σ) · Π_{i=1}^n a_{i,σ(i)}.

    Coût : n! termes → inutilisable pour n > 10.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    total = 0.0
    for perm in permutations(range(n)):
        produit = signe_permutation(perm)
        for i in range(n):
            produit *= A[i, perm[i]]
        total += produit
    return total


# ======================================================================
#  2. Sarrus (3×3 uniquement)
# ======================================================================

def det_sarrus(A: np.ndarray) -> float:
    """
    Règle de Sarrus pour matrices 3×3 :
        det = a₁₁a₂₂a₃₃ + a₁₂a₂₃a₃₁ + a₁₃a₂₁a₃₂
            - a₁₃a₂₂a₃₁ - a₁₁a₂₃a₃₂ - a₁₂a₂₁a₃₃.

    Mnémotechnique : diagonales descendantes (+) - diagonales montantes (-).
    """
    if A.shape != (3, 3):
        raise ValueError("Sarrus ne fonctionne que pour 3×3.")
    a = A
    return (a[0,0]*a[1,1]*a[2,2] + a[0,1]*a[1,2]*a[2,0] + a[0,2]*a[1,0]*a[2,1]
          - a[0,2]*a[1,1]*a[2,0] - a[0,0]*a[1,2]*a[2,1] - a[0,1]*a[1,0]*a[2,2])


# ======================================================================
#  3. Laplace (cofacteurs, récursif)
# ======================================================================

def cofacteur(A: np.ndarray, i: int, j: int) -> float:
    """Cofacteur C_ij = (-1)^{i+j} · det(A sans ligne i, colonne j)."""
    sous_matrice = np.delete(np.delete(A, i, axis=0), j, axis=1)
    return (-1)**(i + j) * det_laplace(sous_matrice)


def det_laplace(A: np.ndarray, ligne: int = 0) -> float:
    """
    Développement de Laplace selon une ligne :
        det(A) = Σ_j a_{i,j} · C_{i,j}.

    Récursif : cas de base pour 1×1 et 2×2.
    Coût : O(n!) — même que Leibniz, mais plus lisible.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 1:
        return float(A[0, 0])
    if n == 2:
        return float(A[0,0]*A[1,1] - A[0,1]*A[1,0])

    total = 0.0
    for j in range(n):
        if A[ligne, j] != 0:  # petite optimisation
            total += A[ligne, j] * cofacteur(A, ligne, j)
    return total


# ======================================================================
#  4. Par échelonnement (Gauss, O(n³))
# ======================================================================

def det_gauss(A: np.ndarray) -> float:
    """
    Déterminant par échelonnement :
        det(A) = (-1)^p · Π r_ii
    où p = nombre de permutations de lignes.

    C'est la méthode utilisée en pratique (O(n³/3)).
    """
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    n_perm = 0

    for k in range(n - 1):
        # Pivot partiel
        i_max = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[i_max, k]) < 1e-300:
            return 0.0  # singulière
        if i_max != k:
            A[[k, i_max]] = A[[i_max, k]]
            n_perm += 1
        for i in range(k + 1, n):
            A[i, k+1:] -= (A[i, k] / A[k, k]) * A[k, k+1:]

    signe = (-1) ** n_perm
    return signe * float(np.prod(np.diag(A)))


# ======================================================================
#  5. Comparaison
# ======================================================================

def comparer_methodes(A: np.ndarray) -> None:
    """Compare toutes les méthodes sur la même matrice."""
    n = A.shape[0]
    print(f"  Matrice {n}×{n} :")

    results = {"Gauss": det_gauss(A), "NumPy": float(np.linalg.det(A))}

    if n <= 8:
        results["Leibniz"] = det_leibniz(A)
    if n == 3:
        results["Sarrus"] = det_sarrus(A)
    if n <= 8:
        results["Laplace"] = det_laplace(A)

    for name, val in results.items():
        print(f"    {name:>10} : {val:>16.8f}")

    # Comptage des opérations
    if n <= 10:
        print(f"    Leibniz/Laplace : {factorial(n)} termes")
    print(f"    Gauss : ~{n**3 // 3} opérations")


if __name__ == "__main__":
    print("=== Matrice 2×2 ===")
    A2 = np.array([[3, 7], [1, 4]], dtype=float)
    print(f"  det = ad - bc = {3*4 - 7*1}")
    comparer_methodes(A2)

    print(f"\n=== Matrice 3×3 (avec Sarrus) ===")
    A3 = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    comparer_methodes(A3)

    print(f"\n=== Matrice 5×5 ===")
    rng = np.random.default_rng(42)
    A5 = rng.standard_normal((5, 5))
    comparer_methodes(A5)

    print(f"\n=== Matrice singulière ===")
    S = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    print(f"  det (Gauss) = {det_gauss(S):.6f}")
    print(f"  → Singulière (rang 2)")

    print(f"\n=== Coût comparatif ===")
    for n in [3, 5, 8, 10, 20]:
        print(f"  n={n:>2} : Leibniz {factorial(n):>15,} termes, Gauss ~{n**3//3:>6} ops")
