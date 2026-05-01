"""
elementary_matrices.py
======================

Matrices élémentaires et opérations sur les lignes.

Couvre :
    - Permutation (Vertauschungsmatrix) : Pij
    - Dilatation (Skalierungsmatrix) : Di(λ)
    - Transvection (Additionsmatrix) : Lij(λ)
    - Décomposition d'une matrice inversible en produit de matrices élémentaires
    - Lien avec l'élimination de Gauss

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np


def permutation(n: int, i: int, j: int) -> np.ndarray:
    """
    Matrice de permutation P_ij : échange les lignes i et j.
    P_ij = I avec lignes i et j échangées.
    Propriété : P_ij² = I, det(P_ij) = -1.
    """
    P = np.eye(n)
    P[[i, j]] = P[[j, i]]
    return P


def dilatation(n: int, i: int, lam: float) -> np.ndarray:
    """
    Matrice de dilatation D_i(λ) : multiplie la ligne i par λ.
    D = I avec D[i,i] = λ.
    Propriété : det(D) = λ.
    """
    D = np.eye(n)
    D[i, i] = lam
    return D


def transvection(n: int, i: int, j: int, lam: float) -> np.ndarray:
    """
    Matrice de transvection L_ij(λ) : ajoute λ fois la ligne j à la ligne i.
    L = I avec L[i,j] = λ (i ≠ j).
    Propriété : det(L) = 1, L_ij(λ)⁻¹ = L_ij(-λ).
    """
    L = np.eye(n)
    L[i, j] = lam
    return L


def verifier_proprietes() -> None:
    """Vérifie les propriétés des matrices élémentaires."""
    n = 3
    print("=== Propriétés des matrices élémentaires ===\n")

    # Permutation
    P = permutation(n, 0, 2)
    print(f"P₀₂ =\n{P}")
    print(f"P² = I ? {np.allclose(P @ P, np.eye(n))} ✓")
    print(f"det(P) = {np.linalg.det(P):.0f} = -1 ✓")
    print(f"P⁻¹ = Pᵀ ? {np.allclose(np.linalg.inv(P), P.T)} ✓\n")

    # Dilatation
    D = dilatation(n, 1, 3.0)
    print(f"D₁(3) =\n{D}")
    print(f"det(D) = {np.linalg.det(D):.0f} = 3 ✓")
    D_inv = dilatation(n, 1, 1/3)
    print(f"D⁻¹ = D₁(1/3) ? {np.allclose(np.linalg.inv(D), D_inv)} ✓\n")

    # Transvection
    L = transvection(n, 2, 0, -5.0)
    print(f"L₂₀(-5) =\n{L}")
    print(f"det(L) = {np.linalg.det(L):.0f} = 1 ✓")
    L_inv = transvection(n, 2, 0, 5.0)
    print(f"L⁻¹ = L₂₀(5) ? {np.allclose(np.linalg.inv(L), L_inv)} ✓\n")


def demo_gauss_elementaire() -> None:
    """
    Montre que l'élimination de Gauss revient à multiplier par des
    matrices élémentaires à gauche.
    """
    print("=== Gauss = produit de matrices élémentaires ===\n")
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    print(f"A =\n{A}\n")

    # Étape 1 : L₁₀(-2) · A → annule a₂₁
    E1 = transvection(3, 1, 0, -2)
    A1 = E1 @ A
    print(f"E₁ = L₁₀(-2) :\n{E1}")
    print(f"E₁·A =\n{A1}\n")

    # Étape 2 : L₂₀(-4) → annule a₃₁
    E2 = transvection(3, 2, 0, -4)
    A2 = E2 @ A1
    print(f"E₂ = L₂₀(-4) :\n{E2}")
    print(f"E₂·E₁·A =\n{A2}\n")

    # Étape 3 : L₂₁(-3) → annule a₃₂
    E3 = transvection(3, 2, 1, -3)
    R = E3 @ A2
    print(f"E₃ = L₂₁(-3) :\n{E3}")
    print(f"R = E₃·E₂·E₁·A =\n{R}\n")

    # Vérification : L = (E₃E₂E₁)⁻¹
    L = np.linalg.inv(E3 @ E2 @ E1)
    print(f"L = (E₃E₂E₁)⁻¹ =\n{L}")
    print(f"||LR - A|| = {np.linalg.norm(L @ R - A):.2e} ✓")


def appliquer_a_vecteur() -> None:
    """Montre l'effet des matrices élémentaires sur un vecteur."""
    print("\n=== Effet sur un vecteur ===\n")
    v = np.array([1, 2, 3], dtype=float)
    print(f"v = {v}")

    P = permutation(3, 0, 2)
    print(f"P₀₂ · v = {P @ v}  (échange composantes 1 et 3)")

    D = dilatation(3, 1, -1)
    print(f"D₁(-1) · v = {D @ v}  (change le signe de la composante 2)")

    L = transvection(3, 2, 0, 5)
    print(f"L₂₀(5) · v = {L @ v}  (ajoute 5× comp. 1 à comp. 3)")


if __name__ == "__main__":
    verifier_proprietes()
    demo_gauss_elementaire()
    appliquer_a_vecteur()
