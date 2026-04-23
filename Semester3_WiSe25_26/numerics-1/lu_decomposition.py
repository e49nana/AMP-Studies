"""
lu_decomposition.py
===================

Décomposition LR comme objet réutilisable, Nachiteration, et Cholesky.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", sections 2.3 et 2.4.

Complète le programme 3 (gauss_elimination.py) avec :
    - Classe LR réutilisable pour multiples seconds membres (section 2.3.2)
    - Nachiteration (section 2.3.7)
    - Calcul du déterminant et de l'inverse
    - Décomposition de Cholesky pour matrices SPD (section 2.4)
    - Comparaison with scipy.linalg

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Classe LR réutilisable
# ======================================================================

class FactorisationLR:
    """
    Encapsule PA = LR et permet de résoudre Ax = b pour n'importe quel b
    en O(n²) (alors que la factorisation coûte O(n³/3) — une seule fois).

    C'est le point clé de la section 2.3.2 : « Wenn man mehrere lineare
    Gleichungssysteme mit derselben Matrix lösen muss, ist der Schritt 1
    nur einmal durchzuführen. »
    """

    def __init__(self, A: np.ndarray) -> None:
        A = np.asarray(A, dtype=float).copy()
        n = A.shape[0]
        self.n = n
        self.p = np.arange(n)  # Merkvektor

        # Gauss avec pivot partiel, stockage in-place (L sous diag, R sur diag)
        for k in range(n - 1):
            i_max = k + int(np.argmax(np.abs(A[k:, k])))
            if i_max != k:
                A[[k, i_max]] = A[[i_max, k]]
                self.p[[k, i_max]] = self.p[[i_max, k]]

            pivot = A[k, k]
            if abs(pivot) < 1e-300:
                raise np.linalg.LinAlgError(f"Pivot nul à l'étape k={k}.")

            for i in range(k + 1, n):
                A[i, k] /= pivot
                A[i, k + 1:] -= A[i, k] * A[k, k + 1:]

        self._LR = A  # matrice combinée L\R

    @property
    def L(self) -> np.ndarray:
        return np.tril(self._LR, -1) + np.eye(self.n)

    @property
    def R(self) -> np.ndarray:
        return np.triu(self._LR)

    def resoudre(self, b: np.ndarray) -> np.ndarray:
        """Résout Ax = b en O(n²) via substitutions avant/arrière."""
        b = np.asarray(b, dtype=float)[self.p]
        n = self.n
        A = self._LR

        # Substitution avant (Ly = Pb)
        y = np.zeros(n)
        for i in range(n):
            y[i] = b[i] - A[i, :i] @ y[:i]

        # Substitution arrière (Rx = y)
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

        return x

    def resoudre_multiple(self, B: np.ndarray) -> np.ndarray:
        """Résout AX = B pour plusieurs seconds membres (colonnes de B)."""
        B = np.asarray(B, dtype=float)
        if B.ndim == 1:
            return self.resoudre(B)
        return np.column_stack([self.resoudre(B[:, j]) for j in range(B.shape[1])])

    def determinant(self) -> float:
        """det(A) = (-1)^{nb_perm} · Π R_ii (section 2.3.3)."""
        p = self.p.copy()
        n_perm = 0
        for i in range(len(p)):
            while p[i] != i:
                j = p[i]
                p[i], p[j] = p[j], p[i]
                n_perm += 1
        signe = -1.0 if n_perm % 2 else 1.0
        return signe * float(np.prod(np.diag(self._LR)))

    def inverse(self) -> np.ndarray:
        """A⁻¹ par résolution de AX = I (n résolutions de systèmes)."""
        return self.resoudre_multiple(np.eye(self.n))


# ======================================================================
#  2. Nachiteration (section 2.3.7)
# ======================================================================

def nachiteration(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    n_iter: int = 3,
    lr: FactorisationLR | None = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Nachiteration (section 2.3.7) : améliore une solution approchée x̃.

    Algorithme :
        1. r = Ax̃ - b  (résidu)
        2. Résoudre Ay = r  (coût : O(n²) si LR déjà calculée)
        3. x̃₂ = x̃ - y

    Renvoie (x_amélioré, historique_résidus).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if lr is None:
        lr = FactorisationLR(A)
    if x0 is None:
        x0 = lr.resoudre(b)

    x = x0.copy()
    residus = [float(np.linalg.norm(A @ x - b, np.inf))]

    for _ in range(n_iter):
        r = A @ x - b
        y = lr.resoudre(r)
        x = x - y
        residus.append(float(np.linalg.norm(A @ x - b, np.inf)))

    return x, residus


# ======================================================================
#  3. Décomposition de Cholesky (section 2.4)
# ======================================================================

def cholesky(A: np.ndarray) -> np.ndarray:
    """
    Décomposition de Cholesky : A = LLᵀ (section 2.4).

    Valide uniquement pour matrices symétriques définies positives.
    Coût : n³/6 (moitié du Gauss, car on exploite la symétrie).

    Renvoie L (triangulaire inférieure).
    """
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    L = np.zeros((n, n))

    for j in range(n):
        # Diagonale
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s <= 0:
            raise np.linalg.LinAlgError(
                f"Matrice non définie positive (étape j={j}, s={s:.2e})."
            )
        L[j, j] = np.sqrt(s)

        # Sous-diagonale
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]

    return L


def resoudre_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Résout Ax = b via A = LLᵀ + substitutions."""
    L = cholesky(A)
    n = len(b)
    # Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    # Lᵀx = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L[i + 1:, i] @ x[i + 1:]) / L[i, i]
    return x


# ======================================================================
#  4. Tracé
# ======================================================================

def tracer_nachiteration(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre comment la Nachiteration améliore la solution."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(42)
    tailles = [50, 100, 200]

    for n in tailles:
        # Matrice modérément mal conditionnée
        A = rng.standard_normal((n, n))
        A = A + n * np.eye(n)  # rend diag dominant mais κ non trivial
        x_exact = np.ones(n)
        b = A @ x_exact

        lr = FactorisationLR(A)
        _, residus = nachiteration(A, b, n_iter=5, lr=lr)
        ax.semilogy(residus, "o-", label=f"$n = {n}$", markersize=5)

    ax.set_xlabel("itération de Nachiteration")
    ax.set_ylabel("$\\|Ax - b\\|_\\infty$")
    ax.set_title("Section 2.3.7 — Nachiteration")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    print("=== LR réutilisable pour multiples b ===")
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    lr = FactorisationLR(A)

    for i in range(3):
        b = np.zeros(3)
        b[i] = 1.0
        x = lr.resoudre(b)
        print(f"  e_{i+1} → x = {x}  (colonne {i+1} de A⁻¹)")

    print(f"\n  det(A) = {lr.determinant():.6f}  (numpy: {np.linalg.det(A):.6f})")
    print(f"  ||A⁻¹ - numpy||_∞ = {np.linalg.norm(lr.inverse() - np.linalg.inv(A), np.inf):.2e}")

    print("\n=== Nachiteration (section 2.3.7) ===")
    rng = np.random.default_rng(0)
    n = 100
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    x_exact = np.ones(n)
    b = A @ x_exact
    x_nach, residus = nachiteration(A, b, n_iter=3)
    print(f"  Résidus après chaque Nachiteration : {[f'{r:.2e}' for r in residus]}")
    print(f"  Erreur finale : {np.linalg.norm(x_nach - x_exact, np.inf):.2e}")

    print("\n=== Cholesky (section 2.4) ===")
    B = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=float)
    L = cholesky(B)
    print(f"  L =\n{L}")
    print(f"  ||LLᵀ - B||_∞ = {np.linalg.norm(L @ L.T - B, np.inf):.2e}")
    x = resoudre_cholesky(B, np.array([1.0, 2.0, 3.0]))
    print(f"  x = {x}")
    print(f"  numpy: {np.linalg.solve(B, [1, 2, 3])}")

    print("\n=== Tracé ===")
    tracer_nachiteration()
    plt.tight_layout()
    plt.savefig("lu_decomposition_demo.png", dpi=120)
    print("Figure sauvegardée : lu_decomposition_demo.png")
