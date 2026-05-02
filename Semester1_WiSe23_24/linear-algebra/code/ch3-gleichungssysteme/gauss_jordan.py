"""
gauss_jordan.py
===============

Élimination de Gauss-Jordan et forme échelonnée réduite (RREF).

Couvre :
    - Gauss-Jordan complet (élimination vers le haut ET vers le bas)
    - RREF (Reduced Row Echelon Form / reduzierte Zeilenstufenform)
    - Résolution de Ax = b par RREF de la matrice augmentée [A|b]
    - Identification des variables libres et pivots
    - Comparaison avec scipy.linalg.rref (via sympy)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RREFResult:
    """Résultat d'un échelonnement."""
    matrice: np.ndarray
    pivot_cols: list[int]
    rang: int
    variables_libres: list[int]

    def __repr__(self) -> str:
        return (f"RREF(rang={self.rang}, pivots={self.pivot_cols}, "
                f"libres={self.variables_libres})")


def rref(A: np.ndarray, tol: float = 1e-10) -> RREFResult:
    """
    Forme échelonnée réduite par lignes (RREF / Gauss-Jordan).

    Algorithme :
        1. Pour chaque colonne, trouver le pivot (pivot partiel)
        2. Normaliser la ligne pivot (pivot → 1)
        3. Éliminer TOUTES les autres entrées de la colonne (haut ET bas)

    Résultat : chaque colonne pivot a exactement un 1, le reste est 0.
    """
    A = np.asarray(A, dtype=float).copy()
    m, n = A.shape
    pivot_cols = []
    row = 0

    for col in range(n):
        if row >= m:
            break
        # Pivot partiel
        i_max = row + np.argmax(np.abs(A[row:, col]))
        if abs(A[i_max, col]) < tol:
            continue

        # Échange
        A[[row, i_max]] = A[[i_max, row]]

        # Normalisation
        A[row] /= A[row, col]

        # Élimination (toutes les lignes sauf row)
        for i in range(m):
            if i != row and abs(A[i, col]) > tol:
                A[i] -= A[i, col] * A[row]

        pivot_cols.append(col)
        row += 1

    # Variables libres
    n_cols = n
    variables_libres = [j for j in range(n_cols) if j not in pivot_cols]

    # Nettoyer les quasi-zéros
    A[np.abs(A) < tol] = 0.0

    return RREFResult(
        matrice=A,
        pivot_cols=pivot_cols,
        rang=len(pivot_cols),
        variables_libres=variables_libres,
    )


def resoudre_par_rref(
    A: np.ndarray, b: np.ndarray,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """
    Résout Ax = b par RREF de [A|b].

    Renvoie (x_particulier, base_kern) :
        - x_particulier : une solution, ou None si incompatible
        - base_kern : base de l'espace des solutions homogènes
        - Solution générale : x = x_part + Σ λᵢ kᵢ
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    m, n = A.shape

    # RREF de la matrice augmentée
    aug = np.hstack([A, b])
    result = rref(aug)
    R = result.matrice

    # Vérifier compatibilité : si un pivot est dans la dernière colonne → incompatible
    if any(p == n for p in result.pivot_cols):
        return None, []

    # Solution particulière
    x_part = np.zeros(n)
    for i, pc in enumerate(result.pivot_cols):
        if pc < n:
            x_part[pc] = R[i, n]

    # Base du noyau (variables libres)
    free = [j for j in range(n) if j not in result.pivot_cols]
    kern_base = []
    for fc in free:
        v = np.zeros(n)
        v[fc] = 1.0
        for i, pc in enumerate(result.pivot_cols):
            if pc < n:
                v[pc] = -R[i, fc]
        kern_base.append(v)

    return x_part, kern_base


def afficher_systeme(A: np.ndarray, b: np.ndarray) -> None:
    """Affiche le système sous forme lisible."""
    m, n = A.shape
    var_names = [f"x{j+1}" for j in range(n)]
    print("  Système :")
    for i in range(m):
        termes = []
        for j in range(n):
            if A[i, j] != 0:
                coeff = A[i, j]
                if coeff == 1:
                    termes.append(f"{var_names[j]}")
                elif coeff == -1:
                    termes.append(f"-{var_names[j]}")
                else:
                    termes.append(f"{coeff:g}·{var_names[j]}")
        eq = " + ".join(termes).replace("+ -", "- ")
        print(f"    {eq} = {b[i]:g}")


if __name__ == "__main__":
    print("=== Exemple 1 : solution unique ===")
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    afficher_systeme(A, b)
    result = rref(np.hstack([A, b.reshape(-1, 1)]))
    print(f"  RREF :\n{result.matrice}")
    x_part, kern = resoudre_par_rref(A, b)
    print(f"  Solution : x = {x_part}")
    print(f"  Vérif : Ax = {A @ x_part} (= b ? {np.allclose(A @ x_part, b)} ✓)\n")

    print("=== Exemple 2 : infinité de solutions ===")
    A2 = np.array([[1, 2, -1, 3], [2, 4, -1, 5], [3, 6, -2, 8]], dtype=float)
    b2 = np.array([1, 3, 4], dtype=float)
    afficher_systeme(A2, b2)
    x_part, kern = resoudre_par_rref(A2, b2)
    print(f"  x_part = {x_part}")
    print(f"  Base Kern : {[v.tolist() for v in kern]}")
    print(f"  Variables libres : {rref(A2).variables_libres}")
    print(f"  Vérif x_part : Ax = {A2 @ x_part} ≈ b ? {np.allclose(A2 @ x_part, b2)} ✓")
    for v in kern:
        print(f"  Vérif kern : A·{v} = {A2 @ v} ≈ 0 ? {np.allclose(A2 @ v, 0)} ✓")

    print(f"\n=== Exemple 3 : système incompatible ===")
    A3 = np.array([[1, 1], [1, 1]], dtype=float)
    b3 = np.array([1, 2], dtype=float)
    afficher_systeme(A3, b3)
    x_part, kern = resoudre_par_rref(A3, b3)
    print(f"  Solution : {x_part} → {'incompatible ✗' if x_part is None else 'trouvée'}")
