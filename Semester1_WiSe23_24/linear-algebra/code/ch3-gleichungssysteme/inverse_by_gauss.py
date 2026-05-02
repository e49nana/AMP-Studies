"""
inverse_by_gauss.py
===================

Calcul de A⁻¹ par Gauss-Jordan : [A|I] → [I|A⁻¹].

Couvre :
    - Algorithme pas-à-pas avec affichage intermédiaire
    - Détection de matrice singulière
    - Vérification A·A⁻¹ = I
    - Formule explicite pour matrices 2×2 et 3×3
    - Comparaison from-scratch vs numpy.linalg.inv

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np


def inverse_gauss_jordan(A: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Calcul de A⁻¹ par Gauss-Jordan.

    On forme la matrice augmentée [A | I] et on la réduit en RREF.
    Si la partie gauche devient I, la partie droite est A⁻¹.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrice non carrée.")

    aug = np.hstack([A.copy(), np.eye(n)])

    if verbose:
        print(f"  [A | I] =\n{np.round(aug, 4)}\n")

    for col in range(n):
        # Pivot partiel
        i_max = col + np.argmax(np.abs(aug[col:, col]))
        if abs(aug[i_max, col]) < 1e-12:
            raise np.linalg.LinAlgError(f"Matrice singulière (étape {col}).")
        aug[[col, i_max]] = aug[[i_max, col]]

        # Normalisation
        aug[col] /= aug[col, col]

        # Élimination
        for i in range(n):
            if i != col:
                aug[i] -= aug[i, col] * aug[col]

        if verbose:
            print(f"  Étape {col+1} (pivot colonne {col+1}) :")
            print(f"  {np.round(aug, 4)}\n")

    return aug[:, n:]


def inverse_2x2(A: np.ndarray) -> np.ndarray:
    """
    Formule directe pour 2×2 :
        A⁻¹ = (1/det) · [[d, -b], [-c, a]]
    avec A = [[a, b], [c, d]].
    """
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    det = a*d - b*c
    if abs(det) < 1e-12:
        raise np.linalg.LinAlgError("det = 0.")
    return np.array([[d, -b], [-c, a]]) / det


def inverse_3x3(A: np.ndarray) -> np.ndarray:
    """
    Formule par cofacteurs pour 3×3 :
        A⁻¹ = (1/det) · cof(A)ᵀ.
    """
    a = A
    det = (a[0,0]*(a[1,1]*a[2,2]-a[1,2]*a[2,1])
          -a[0,1]*(a[1,0]*a[2,2]-a[1,2]*a[2,0])
          +a[0,2]*(a[1,0]*a[2,1]-a[1,1]*a[2,0]))
    if abs(det) < 1e-12:
        raise np.linalg.LinAlgError("det = 0.")

    cof = np.array([
        [a[1,1]*a[2,2]-a[1,2]*a[2,1], -(a[1,0]*a[2,2]-a[1,2]*a[2,0]), a[1,0]*a[2,1]-a[1,1]*a[2,0]],
        [-(a[0,1]*a[2,2]-a[0,2]*a[2,1]), a[0,0]*a[2,2]-a[0,2]*a[2,0], -(a[0,0]*a[2,1]-a[0,1]*a[2,0])],
        [a[0,1]*a[1,2]-a[0,2]*a[1,1], -(a[0,0]*a[1,2]-a[0,2]*a[1,0]), a[0,0]*a[1,1]-a[0,1]*a[1,0]],
    ])
    return cof / det


if __name__ == "__main__":
    print("=== Inverse par Gauss-Jordan (pas-à-pas) ===\n")
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    print(f"A =\n{A}\n")
    A_inv = inverse_gauss_jordan(A, verbose=True)
    print(f"A⁻¹ =\n{np.round(A_inv, 6)}")
    print(f"Vérif A·A⁻¹ =\n{np.round(A @ A_inv, 10)}\n")

    print("=== Formule 2×2 ===")
    B = np.array([[3, 7], [1, 4]], dtype=float)
    B_inv = inverse_2x2(B)
    print(f"B⁻¹ = {B_inv.tolist()}")
    print(f"det(B) = {3*4 - 7*1}")
    print(f"B·B⁻¹ = {np.round(B @ B_inv, 10).tolist()}\n")

    print("=== Formule 3×3 (cofacteurs) ===")
    C_inv_cof = inverse_3x3(A)
    print(f"||Gauss-Jordan - cofacteurs|| = {np.linalg.norm(A_inv - C_inv_cof):.2e}")
    print(f"||mine - numpy|| = {np.linalg.norm(A_inv - np.linalg.inv(A)):.2e}")

    print("\n=== Matrice singulière ===")
    S = np.array([[1, 2], [2, 4]], dtype=float)
    try:
        inverse_gauss_jordan(S)
    except np.linalg.LinAlgError as e:
        print(f"  Détecté : {e} ✓")

    print("\n=== Benchmark from-scratch vs NumPy ===")
    rng = np.random.default_rng(42)
    for n in [5, 10, 50, 100]:
        A = rng.standard_normal((n, n))
        inv_mine = inverse_gauss_jordan(A)
        inv_np = np.linalg.inv(A)
        err = np.linalg.norm(inv_mine - inv_np, np.inf)
        print(f"  n={n:>3} : ||mine - numpy||_∞ = {err:.2e}")
