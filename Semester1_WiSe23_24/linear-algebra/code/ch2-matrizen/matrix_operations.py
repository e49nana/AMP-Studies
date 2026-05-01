"""
matrix_operations.py
====================

Opérations matricielles fondamentales from-scratch.

Couvre :
    - Addition, multiplication scalaire
    - Multiplication matricielle (ligne × colonne)
    - Transposée
    - Inverse par Gauss-Jordan
    - Trace
    - Puissances de matrices
    - Comparaison from-scratch vs NumPy

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np


class Matrice:
    """Matrice de R^{m×n} avec opérations from-scratch."""

    def __init__(self, data: list[list[float]] | np.ndarray) -> None:
        self.data = np.array(data, dtype=float)
        self.m, self.n = self.data.shape

    def __repr__(self) -> str:
        return f"Matrice({self.m}×{self.n})"

    def __str__(self) -> str:
        return str(self.data)

    # --- Opérations élémentaires ---

    def __add__(self, other: Matrice) -> Matrice:
        return Matrice(self.data + other.data)

    def __sub__(self, other: Matrice) -> Matrice:
        return Matrice(self.data - other.data)

    def __mul__(self, scalar: float) -> Matrice:
        return Matrice(scalar * self.data)

    def __rmul__(self, scalar: float) -> Matrice:
        return self.__mul__(scalar)

    def __neg__(self) -> Matrice:
        return Matrice(-self.data)

    def __matmul__(self, other: Matrice) -> Matrice:
        """Multiplication matricielle from-scratch : C_ij = Σ_k A_ik B_kj."""
        if self.n != other.m:
            raise ValueError(f"Dimensions incompatibles : {self.m}×{self.n} @ {other.m}×{other.n}")
        C = np.zeros((self.m, other.n))
        for i in range(self.m):
            for j in range(other.n):
                for k in range(self.n):
                    C[i, j] += self.data[i, k] * other.data[k, j]
        return Matrice(C)

    # --- Propriétés ---

    def transposee(self) -> Matrice:
        """Aᵀ : échange lignes et colonnes."""
        return Matrice(self.data.T.copy())

    def trace(self) -> float:
        """tr(A) = Σ a_ii."""
        return float(np.sum(np.diag(self.data)))

    def est_carree(self) -> bool:
        return self.m == self.n

    def est_symetrique(self, tol: float = 1e-10) -> bool:
        if not self.est_carree():
            return False
        return bool(np.allclose(self.data, self.data.T, atol=tol))

    # --- Inverse par Gauss-Jordan ---

    def inverse(self) -> Matrice:
        """
        A⁻¹ par Gauss-Jordan : on échelonne [A | I] → [I | A⁻¹].
        """
        if not self.est_carree():
            raise ValueError("Matrice non carrée.")
        n = self.n
        aug = np.hstack([self.data.copy(), np.eye(n)])

        for col in range(n):
            # Pivot partiel
            i_max = col + np.argmax(np.abs(aug[col:, col]))
            aug[[col, i_max]] = aug[[i_max, col]]

            if abs(aug[col, col]) < 1e-12:
                raise np.linalg.LinAlgError("Matrice singulière.")

            aug[col] /= aug[col, col]
            for i in range(n):
                if i != col:
                    aug[i] -= aug[i, col] * aug[col]

        return Matrice(aug[:, n:])

    # --- Puissances ---

    def puissance(self, k: int) -> Matrice:
        """Aᵏ par multiplications successives."""
        if not self.est_carree():
            raise ValueError("Puissance définie pour matrices carrées seulement.")
        result = Matrice(np.eye(self.n))
        for _ in range(k):
            result = result @ self
        return result

    @staticmethod
    def identite(n: int) -> Matrice:
        return Matrice(np.eye(n))

    @staticmethod
    def zeros(m: int, n: int) -> Matrice:
        return Matrice(np.zeros((m, n)))


def comparer_avec_numpy(A: Matrice, B: Matrice) -> None:
    """Vérifie que nos opérations coincident avec NumPy."""
    print(f"  {'opération':>15} | {'||mine - numpy||':>16}")
    print("  " + "-" * 38)
    produit = (A @ B).data
    ref = A.data @ B.data
    print(f"  {'A @ B':>15} | {np.linalg.norm(produit - ref):.2e}")

    trans = A.transposee().data
    print(f"  {'Aᵀ':>15} | {np.linalg.norm(trans - A.data.T):.2e}")

    if A.est_carree():
        inv_mine = A.inverse().data
        inv_np = np.linalg.inv(A.data)
        print(f"  {'A⁻¹':>15} | {np.linalg.norm(inv_mine - inv_np):.2e}")


if __name__ == "__main__":
    A = Matrice([[1, 2], [3, 4]])
    B = Matrice([[5, 6], [7, 8]])

    print("=== Opérations de base ===")
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    print(f"A + B =\n{A + B}")
    print(f"A @ B =\n{A @ B}")
    print(f"Aᵀ =\n{A.transposee()}")
    print(f"tr(A) = {A.trace()}")

    print(f"\n=== Inverse ===")
    print(f"A⁻¹ =\n{A.inverse()}")
    print(f"A @ A⁻¹ =\n{A @ A.inverse()}")

    print(f"\n=== Puissances ===")
    print(f"A² =\n{A.puissance(2)}")
    print(f"A³ =\n{A.puissance(3)}")

    print(f"\n=== Comparaison avec NumPy ===")
    comparer_avec_numpy(A, B)

    print(f"\n=== Propriétés ===")
    S = Matrice([[1, 2, 3], [2, 5, 4], [3, 4, 6]])
    print(f"S symétrique ? {S.est_symetrique()}")
    print(f"A symétrique ? {A.est_symetrique()}")
