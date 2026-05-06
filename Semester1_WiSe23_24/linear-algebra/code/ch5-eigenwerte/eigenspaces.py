"""
eigenspaces.py
==============

Eigenräume, multiplicités algébrique et géométrique.

Couvre :
    - Eigenraum E_λ = Kern(A - λI)
    - Multiplicité algébrique mₐ (ordre de la racine du poly. car.)
    - Multiplicité géométrique m_g = dim Kern(A - λI)
    - Toujours m_g ≤ mₐ
    - Cas m_g < mₐ : matrice non diagonalisable

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EigenInfo:
    """Information complète sur une valeur propre."""
    valeur: complex
    mult_algebrique: int
    mult_geometrique: int
    eigenraum: list[np.ndarray]
    diagonalisable: bool


def eigenraum(A: np.ndarray, lam: float, tol: float = 1e-8) -> list[np.ndarray]:
    """
    Base de l'espace propre E_λ = Kern(A - λI).
    """
    n = A.shape[0]
    B = A - lam * np.eye(n)

    # RREF de B
    B = B.copy()
    pivot_cols = []
    row = 0
    for col in range(n):
        if row >= n:
            break
        i_max = row + np.argmax(np.abs(B[row:, col]))
        if abs(B[i_max, col]) < tol:
            continue
        B[[row, i_max]] = B[[i_max, row]]
        B[row] /= B[row, col]
        for i in range(n):
            if i != row:
                B[i] -= B[i, col] * B[row]
        pivot_cols.append(col)
        row += 1

    free = [j for j in range(n) if j not in pivot_cols]
    base = []
    for fc in free:
        v = np.zeros(n)
        v[fc] = 1.0
        for i, pc in enumerate(pivot_cols):
            v[pc] = -B[i, fc]
        # Normaliser
        norm = np.linalg.norm(v)
        if norm > tol:
            base.append(v / norm)
    return base


def multiplicite_algebrique(eigvals: np.ndarray, lam: float, tol: float = 1e-6) -> int:
    """Compte combien de fois λ apparaît dans le spectre."""
    return int(np.sum(np.abs(eigvals - lam) < tol))


def analyser_spectre(A: np.ndarray, tol: float = 1e-6) -> list[EigenInfo]:
    """Analyse complète du spectre d'une matrice."""
    A = np.asarray(A, dtype=float)
    eigvals = np.linalg.eigvals(A)

    # Regrouper les valeurs propres distinctes
    distincts = []
    used = np.zeros(len(eigvals), dtype=bool)
    for i, lam in enumerate(eigvals):
        if used[i]:
            continue
        lam_real = lam.real if abs(lam.imag) < tol else lam
        if isinstance(lam_real, complex):
            continue  # skip complexes pour simplifier
        distincts.append(float(lam_real))
        for j in range(i, len(eigvals)):
            if abs(eigvals[j] - lam) < tol:
                used[j] = True

    infos = []
    for lam in sorted(distincts):
        ma = multiplicite_algebrique(eigvals.real, lam, tol)
        er = eigenraum(A, lam, tol)
        mg = len(er)
        infos.append(EigenInfo(
            valeur=lam,
            mult_algebrique=ma,
            mult_geometrique=mg,
            eigenraum=er,
            diagonalisable=(mg == ma),
        ))

    return infos


def est_diagonalisable(A: np.ndarray) -> bool:
    """A est diagonalisable ssi m_g = m_a pour toute valeur propre."""
    infos = analyser_spectre(A)
    return all(info.diagonalisable for info in infos)


if __name__ == "__main__":
    print("=== Exemple 1 : matrice diagonalisable ===")
    A1 = np.array([[2, 1, 0], [0, 3, 0], [0, 0, 2]], dtype=float)
    print(f"A =\n{A1}\n")
    for info in analyser_spectre(A1):
        print(f"  λ = {info.valeur:.1f} : mₐ = {info.mult_algebrique}, "
              f"m_g = {info.mult_geometrique}, "
              f"E_λ = {[np.round(v, 4).tolist() for v in info.eigenraum]}")
    print(f"  Diagonalisable ? {est_diagonalisable(A1)}")

    print(f"\n=== Exemple 2 : matrice NON diagonalisable ===")
    A2 = np.array([[2, 1], [0, 2]], dtype=float)  # Jordan block
    print(f"A =\n{A2}\n")
    for info in analyser_spectre(A2):
        print(f"  λ = {info.valeur:.1f} : mₐ = {info.mult_algebrique}, "
              f"m_g = {info.mult_geometrique}")
        print(f"    E_λ = {[np.round(v, 4).tolist() for v in info.eigenraum]}")
    print(f"  Diagonalisable ? {est_diagonalisable(A2)}")
    print(f"  → m_g = 1 < m_a = 2 : pas assez de vecteurs propres indépendants.")

    print(f"\n=== Exemple 3 : matrice symétrique (toujours diagonalisable) ===")
    A3 = np.array([[4, 2, 0], [2, 5, 3], [0, 3, 6]], dtype=float)
    print(f"A =\n{A3}\n")
    for info in analyser_spectre(A3):
        print(f"  λ = {info.valeur:.4f} : mₐ = {info.mult_algebrique}, "
              f"m_g = {info.mult_geometrique}")
    print(f"  Diagonalisable ? {est_diagonalisable(A3)} "
          f"(symétrique → toujours ✓)")

    print(f"\n=== Exemple 4 : identité (triple valeur propre) ===")
    I3 = np.eye(3)
    for info in analyser_spectre(I3):
        print(f"  λ = {info.valeur:.0f} : mₐ = {info.mult_algebrique}, "
              f"m_g = {info.mult_geometrique}")
    print(f"  Diagonalisable ? {est_diagonalisable(I3)} "
          f"(m_g = m_a = 3 ✓)")
