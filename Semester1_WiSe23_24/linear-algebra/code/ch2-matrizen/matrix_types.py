"""
matrix_types.py
===============

Types spéciaux de matrices et leurs propriétés.

Couvre :
    - Symétrique (A = Aᵀ) et antisymétrique (A = -Aᵀ)
    - Orthogonale (QᵀQ = I, det = ±1)
    - Diagonale, triangulaire (inf/sup)
    - Idempotente (A² = A, projections)
    - Nilpotente (Aᵏ = 0)
    - Positiv definit (xᵀAx > 0)
    - Décomposition A = S + K (sym + antisym)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def est_symetrique(A: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(A, A.T, atol=tol)


def est_antisymetrique(A: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(A, -A.T, atol=tol)


def est_orthogonale(A: np.ndarray, tol: float = 1e-10) -> bool:
    """QᵀQ = I et QQᵀ = I."""
    n = A.shape[0]
    return np.allclose(A.T @ A, np.eye(n), atol=tol)


def est_diagonale(A: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(A, np.diag(np.diag(A)), atol=tol)


def est_triangulaire_inf(A: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(A, np.tril(A), atol=tol)


def est_triangulaire_sup(A: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(A, np.triu(A), atol=tol)


def est_idempotente(A: np.ndarray, tol: float = 1e-10) -> bool:
    """A² = A (matrice de projection)."""
    return np.allclose(A @ A, A, atol=tol)


def est_nilpotente(A: np.ndarray, tol: float = 1e-10) -> int | None:
    """Renvoie l'indice de nilpotence k (plus petit k tq Aᵏ = 0), ou None."""
    n = A.shape[0]
    Ak = A.copy()
    for k in range(1, n + 1):
        if np.allclose(Ak, 0, atol=tol):
            return k
        Ak = Ak @ A
    return None


def est_positiv_definit(A: np.ndarray) -> bool:
    """Toutes les valeurs propres sont > 0 (pour A symétrique)."""
    if not est_symetrique(A):
        return False
    return bool(np.all(np.linalg.eigvalsh(A) > 0))


def decomposition_sym_antisym(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    A = S + K avec S = (A + Aᵀ)/2 (symétrique), K = (A - Aᵀ)/2 (antisymétrique).
    Toute matrice carrée se décompose ainsi de manière unique.
    """
    S = (A + A.T) / 2
    K = (A - A.T) / 2
    return S, K


def classifier(A: np.ndarray) -> list[str]:
    """Renvoie la liste des types détectés pour A."""
    types = []
    if est_diagonale(A): types.append("diagonale")
    if est_triangulaire_inf(A) and not est_diagonale(A): types.append("triangulaire inf.")
    if est_triangulaire_sup(A) and not est_diagonale(A): types.append("triangulaire sup.")
    if est_symetrique(A): types.append("symétrique")
    if est_antisymetrique(A): types.append("antisymétrique")
    if est_orthogonale(A): types.append("orthogonale")
    if est_idempotente(A): types.append("idempotente")
    nilp = est_nilpotente(A)
    if nilp is not None: types.append(f"nilpotente (k={nilp})")
    if est_symetrique(A) and est_positiv_definit(A): types.append("positiv definit")
    return types if types else ["aucun type spécial"]


def tracer_effet_orthogonale(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre qu'une matrice orthogonale conserve les longueurs et angles."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    theta = np.pi / 6  # rotation de 30°
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])

    # Carré unité
    pts = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], dtype=float).T
    pts_rot = Q @ pts

    ax.plot(pts[0], pts[1], "b-o", markersize=5, label="original")
    ax.plot(pts_rot[0], pts_rot[1], "r-o", markersize=5, label=f"Q·x (rotation {np.degrees(theta):.0f}°)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(); ax.set_title("Matrice orthogonale : conserve longueurs et angles")
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    return ax


if __name__ == "__main__":
    print("=== Classification de matrices ===\n")

    matrices = {
        "Identité": np.eye(3),
        "Diagonale": np.diag([2, -1, 3]),
        "Symétrique": np.array([[4,2,1],[2,5,3],[1,3,6]], dtype=float),
        "Antisymétrique": np.array([[0,1,-2],[-1,0,3],[2,-3,0]], dtype=float),
        "Rotation 45°": np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                   [np.sin(np.pi/4), np.cos(np.pi/4)]]),
        "Projection": np.array([[1,0],[0,0]], dtype=float),
        "Nilpotente": np.array([[0,1,0],[0,0,1],[0,0,0]], dtype=float),
        "Triang. sup.": np.array([[1,2,3],[0,4,5],[0,0,6]], dtype=float),
    }

    for nom, A in matrices.items():
        types = classifier(A)
        print(f"  {nom:20s} : {', '.join(types)}")

    print(f"\n=== Décomposition A = S + K ===")
    A = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
    S, K = decomposition_sym_antisym(A)
    print(f"A =\n{A}")
    print(f"S = (A+Aᵀ)/2 =\n{S}")
    print(f"K = (A-Aᵀ)/2 =\n{K}")
    print(f"S symétrique ? {est_symetrique(S)}")
    print(f"K antisymétrique ? {est_antisymetrique(K)}")
    print(f"||S + K - A|| = {np.linalg.norm(S + K - A):.2e}")

    tracer_effet_orthogonale()
    plt.savefig("matrix_types_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
