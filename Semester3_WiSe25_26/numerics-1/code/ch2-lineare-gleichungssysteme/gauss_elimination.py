"""
gauss_elimination.py
====================

Élimination de Gauss et décomposition LR avec stratégies de pivot.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 2.3.

Couvre :
    - Élimination de Gauss "naïve" (sans pivot)
    - Pivot partiel par colonne (Spaltenpivotsuche)  -- standard en pratique
    - Pivot total (Totale Pivotsuche)                 -- plus stable, plus cher
    - Décomposition PA = LR avec vecteur de permutation (Merkvektor)
    - Substitutions avant et arrière
    - Reproduction de l'Übung 2.21 du script (instabilité sans pivot)

Convention : on utilise la lettre R (rechte Dreiecksmatrix) plutôt que U
comme dans le script allemand. La littérature anglaise écrit LU.

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt


class StrategiePivot(str, Enum):
    """Trois stratégies de pivot couvertes par le script (section 2.3.5)."""
    AUCUN = "aucun"
    PARTIEL = "partiel"   # Spaltenpivotsuche (par colonne)
    TOTAL = "total"       # Totale Pivotsuche


# ----------------------------------------------------------------------
#  1. Substitutions avant et arrière (formule 2.10 du script)
# ----------------------------------------------------------------------

def substitution_avant(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Résout L y = b avec L triangulaire inférieure à diagonale unitaire.

    Vorwärtseinsetzen : y_i = b_i - Σ_{j<i} L_ij y_j.
    Coût : ~ n²/2 opérations (Satz 2.20).
    """
    n = len(b)
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def substitution_arriere(R: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Résout R x = c avec R triangulaire supérieure (formule 2.10) :
        x_i = (c_i - Σ_{j>i} R_ij x_j) / R_ii,   i = n, ..., 1.
    Coût : ~ n²/2 opérations.
    """
    n = len(c)
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    return x


# ----------------------------------------------------------------------
#  2. Décomposition LR avec stratégie de pivot
# ----------------------------------------------------------------------

@dataclass
class DecompositionLR:
    """
    Résultat d'une décomposition PA = LR (avec éventuellement aussi
    permutation de colonnes pour le pivot total : PAQ = LR).

    Attributs
    ---------
    L : matrice triangulaire inférieure à diagonale unitaire.
    R : matrice triangulaire supérieure.
    p : Merkvektor des permutations de lignes.
        p[i] = indice de la ligne d'origine qui se retrouve en position i.
    q : Merkvektor des permutations de colonnes (pivot total uniquement).
        Vaut None si pivot ≠ total.
    strategie : stratégie utilisée.
    """
    L: np.ndarray
    R: np.ndarray
    p: np.ndarray
    q: np.ndarray | None
    strategie: StrategiePivot

    def matrice_P(self) -> np.ndarray:
        """Reconstruit la matrice de permutation P à partir du Merkvektor p."""
        n = len(self.p)
        P = np.zeros((n, n))
        for i, pi in enumerate(self.p):
            P[i, pi] = 1.0
        return P

    def matrice_Q(self) -> np.ndarray | None:
        """Matrice de permutation des colonnes, ou None."""
        if self.q is None:
            return None
        n = len(self.q)
        Q = np.zeros((n, n))
        for j, qj in enumerate(self.q):
            Q[qj, j] = 1.0
        return Q


def decomposition_lr(
    A: np.ndarray,
    strategie: StrategiePivot | str = StrategiePivot.PARTIEL,
) -> DecompositionLR:
    """
    Calcule la décomposition LR de A avec la stratégie de pivot demandée.

    Implémentation in-place sur une copie de A : L et R partagent la même
    matrice (L sous la diagonale, R sur et au-dessus), conformément à la
    description « höchst speichersparend » du script p. 28-29.

    Coût : O(n³/3) pour la décomposition (Satz 2.20).

    Lève
    ----
    np.linalg.LinAlgError
        Si un pivot vaut 0 (matrice singulière, ou pivot indispensable
        en stratégie AUCUN).
    """
    strategie = StrategiePivot(strategie)
    A = np.asarray(A, dtype=float).copy()
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrice non carrée : {A.shape}.")

    p = np.arange(n)  # Merkvektor des lignes
    q = np.arange(n) if strategie == StrategiePivot.TOTAL else None

    for k in range(n - 1):
        # --- Choix du pivot selon la stratégie ---
        if strategie == StrategiePivot.AUCUN:
            pivot = A[k, k]
        elif strategie == StrategiePivot.PARTIEL:
            # Plus grand |a_{ik}| pour i = k, ..., n-1
            i_max = k + int(np.argmax(np.abs(A[k:, k])))
            if i_max != k:
                A[[k, i_max]] = A[[i_max, k]]
                p[[k, i_max]] = p[[i_max, k]]
            pivot = A[k, k]
        else:  # TOTAL
            sub = np.abs(A[k:, k:])
            di, dj = np.unravel_index(np.argmax(sub), sub.shape)
            i_max, j_max = k + di, k + dj
            if i_max != k:
                A[[k, i_max]] = A[[i_max, k]]
                p[[k, i_max]] = p[[i_max, k]]
            if j_max != k:
                A[:, [k, j_max]] = A[:, [j_max, k]]
                q[[k, j_max]] = q[[j_max, k]]
            pivot = A[k, k]

        if abs(pivot) < 1e-300:
            raise np.linalg.LinAlgError(
                f"Pivot nul à l'étape k={k} (strategie={strategie.value})."
            )

        # --- Élimination (formules 2.5 - 2.6 du script) ---
        for i in range(k + 1, n):
            A[i, k] /= A[k, k]                    # ℓ_ik
            A[i, k + 1:] -= A[i, k] * A[k, k + 1:]  # mise à jour de la sous-matrice

    # Extraction de L et R depuis la matrice mémoire combinée
    L = np.tril(A, k=-1) + np.eye(n)
    R = np.triu(A)
    return DecompositionLR(L=L, R=R, p=p, q=q, strategie=strategie)


# ----------------------------------------------------------------------
#  3. Résolution d'un système Ax = b
# ----------------------------------------------------------------------

def resoudre(
    A: np.ndarray,
    b: np.ndarray,
    strategie: StrategiePivot | str = StrategiePivot.PARTIEL,
) -> np.ndarray:
    """
    Résout Ax = b par décomposition LR + substitutions.

    Étapes :
        1. PA = LR (PAQ = LR pour pivot total)
        2. Permuter b : b' = P b
        3. L y = b'  (substitution avant)
        4. R z = y   (substitution arrière)
        5. x = Q z   (réordonner si pivot total)
    """
    decomp = decomposition_lr(A, strategie)
    b_perm = np.asarray(b, dtype=float)[decomp.p]
    y = substitution_avant(decomp.L, b_perm)
    z = substitution_arriere(decomp.R, y)

    if decomp.q is None:
        return z
    # Inverser la permutation des colonnes : x[q[j]] = z[j]
    x = np.empty_like(z)
    x[decomp.q] = z
    return x


def determinant(A: np.ndarray) -> float:
    """
    Déterminant via décomposition LR (section 2.3.3 : « Abfallprodukt »).

    det(A) = (-1)^{nb_perm} · ∏ R_ii.
    Coût : O(n³/3), bien moins cher que la formule de Leibniz.
    """
    decomp = decomposition_lr(A, StrategiePivot.PARTIEL)
    # Compter le nombre d'inversions dans p (= nombre de transpositions)
    p = decomp.p.copy()
    n_perm = 0
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            n_perm += 1
    signe = -1.0 if n_perm % 2 else 1.0
    return signe * float(np.prod(np.diag(decomp.R)))


# ----------------------------------------------------------------------
#  4. Outils d'analyse d'erreur
# ----------------------------------------------------------------------

def residu_relatif(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """||Ax - b|| / ||b|| (norme infini)."""
    r = A @ x - b
    nb = np.linalg.norm(b, np.inf)
    return float(np.linalg.norm(r, np.inf) / nb) if nb > 0 else float("inf")


def erreur_relative(x_calc: np.ndarray, x_exact: np.ndarray) -> float:
    """||x_calc - x_exact|| / ||x_exact|| (norme infini)."""
    nx = np.linalg.norm(x_exact, np.inf)
    return float(np.linalg.norm(x_calc - x_exact, np.inf) / nx)


# ----------------------------------------------------------------------
#  5. Reproduction de l'Übung 2.21 — instabilité sans pivot
# ----------------------------------------------------------------------

def systeme_uebung_2_21() -> tuple[np.ndarray, np.ndarray]:
    """
    Système de l'Übung 2.21 :
        0.00035 x1 + 1.2654 x2 = 3.5267
        1.2547  x1 + 1.3182 x2 = 6.8541

    Le pivot a_11 = 0.00035 est minuscule : sans pivot, la division
    par cette valeur fait exploser les ℓ_i1, et la combinaison qui suit
    perd toute la précision sur a_22 (Auslöschung classique).
    """
    A = np.array([[0.00035, 1.2654], [1.2547, 1.3182]])
    b = np.array([3.5267, 6.8541])
    return A, b


def comparer_strategies_uebung_2_21() -> dict[str, dict]:
    """Compare les 3 stratégies sur l'exemple de l'Übung 2.21."""
    A, b = systeme_uebung_2_21()
    x_ref = np.linalg.solve(A, b)  # référence

    resultats = {}
    for strat in StrategiePivot:
        try:
            x = resoudre(A, b, strat)
            resultats[strat.value] = {
                "x": x,
                "residu_rel": residu_relatif(A, x, b),
                "erreur_rel": erreur_relative(x, x_ref),
            }
        except np.linalg.LinAlgError as e:
            resultats[strat.value] = {"erreur": str(e)}
    return resultats


# ----------------------------------------------------------------------
#  6. Tracé : croissance de l'erreur en fonction du conditionnement
# ----------------------------------------------------------------------

def matrice_hilbert(n: int) -> np.ndarray:
    """
    Matrice de Hilbert n×n : H_ij = 1 / (i + j - 1).
    Exemple classique de matrice extrêmement mal conditionnée :
    cond_2(H_n) croît exponentiellement avec n.
    """
    i = np.arange(1, n + 1)
    return 1.0 / (i[:, None] + i[None, :] - 1)


def tracer_erreur_vs_conditionnement(
    tailles: tuple[int, ...] = tuple(range(2, 14)),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Sur la suite des matrices de Hilbert, trace l'erreur relative
    en fonction du nombre de conditionnement.

    Borne théorique (section 2.3.6) :
        ||x_calc - x|| / ||x||  ≤  κ(A) · ε_mach  (ordre de grandeur)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    eps = np.finfo(float).eps
    conds, errs = [], []
    for n in tailles:
        H = matrice_hilbert(n)
        x_exact = np.ones(n)
        b = H @ x_exact
        try:
            x_calc = resoudre(H, b, StrategiePivot.PARTIEL)
            conds.append(np.linalg.cond(H, np.inf))
            errs.append(erreur_relative(x_calc, x_exact))
        except np.linalg.LinAlgError:
            pass

    conds, errs = np.array(conds), np.array(errs)
    ax.loglog(conds, errs, "bo-", label="erreur observée (Hilbert)")
    ax.loglog(conds, conds * eps, "r--", label=r"borne théorique : $\kappa(A) \cdot \varepsilon_{mach}$")
    ax.set_xlabel(r"nombre de conditionnement $\kappa_\infty(A)$")
    ax.set_ylabel("erreur relative sur la solution")
    ax.set_title("Section 2.3.6 — l'erreur croît avec le conditionnement")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


# ----------------------------------------------------------------------
#  Démo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Test élémentaire 3×3 ===")
    A = np.array([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]])
    b = np.array([5.0, 12.0, 28.0])
    x = resoudre(A, b)
    print(f"x = {x}")
    print(f"vérification A x = {A @ x} (attendu {b})")
    print(f"det(A) = {determinant(A)}  (numpy : {np.linalg.det(A):.6f})")

    print("\n=== Übung 2.21 — instabilité sans pivot ===")
    res = comparer_strategies_uebung_2_21()
    A, b = systeme_uebung_2_21()
    x_ref = np.linalg.solve(A, b)
    print(f"Solution exacte (numpy) : {x_ref}")
    for strat, r in res.items():
        if "erreur" in r:
            print(f"  {strat:>8}  ÉCHEC : {r['erreur']}")
        else:
            print(
                f"  {strat:>8}  x = {r['x']}, "
                f"err_rel = {r['erreur_rel']:.2e}, "
                f"residu_rel = {r['residu_rel']:.2e}"
            )

    print("\n=== Comparaison from-scratch vs numpy.linalg ===")
    rng = np.random.default_rng(42)
    A = rng.standard_normal((50, 50))
    b = rng.standard_normal(50)
    x_mine = resoudre(A, b)
    x_np = np.linalg.solve(A, b)
    print(f"||x_mine - x_numpy||_∞ = {np.linalg.norm(x_mine - x_np, np.inf):.2e}")

    print("\n=== Tracé : erreur vs conditionnement (matrices de Hilbert) ===")
    tracer_erreur_vs_conditionnement()
    plt.tight_layout()
    plt.savefig("gauss_conditionnement.png", dpi=120)
    print("Figure sauvegardée : gauss_conditionnement.png")
