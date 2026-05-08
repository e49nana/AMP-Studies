"""
least_squares_intro.py
======================

Introduction aux moindres carrés du point de vue de l'algèbre linéaire.

Couvre :
    - Projection orthogonale dans R^n : le point de vue géométrique
    - Ax = b n'a pas de solution → on cherche x* tel que Ax* est la
      projection de b sur Im(A)
    - Équations normales AᵀAx = Aᵀb comme conséquence de la projection
    - Droite de régression comme application
    - Pont vers Numerik I (programme householder_qr.py)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def projection_sur_image(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Projection orthogonale de b sur Im(A) :
        p = A(AᵀA)⁻¹Aᵀb = A x*.

    C'est le point de Im(A) le plus proche de b.
    """
    x_star = np.linalg.solve(A.T @ A, A.T @ b)
    return A @ x_star


def residu(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """r = b - Ax (le résidu est orthogonal à Im(A))."""
    return b - A @ x


def moindres_carres(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Résout ||Ax - b||₂ → min par équations normales."""
    return np.linalg.solve(A.T @ A, A.T @ b)


def regression_lineaire(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Droite y = a + bx au sens des moindres carrés.

    Matrice de design : A = [[1, x₁], [1, x₂], ...].
    """
    A = np.column_stack([np.ones_like(x), x])
    c = moindres_carres(A, y)
    return c[0], c[1]  # (intercept, pente)


def r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient de détermination R²."""
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot


def tracer_projection_3d(ax=None):
    """Visualise la projection de b sur le plan Im(A) en R³."""
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    # Im(A) = plan engendré par les colonnes de A
    A = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)

    x_star = moindres_carres(A, b)
    p = A @ x_star
    r = b - p

    # Plan
    xx, yy = np.meshgrid(np.linspace(-1, 3, 10), np.linspace(-1, 3, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.15, color="cyan")

    # Vecteurs
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color="blue", arrow_length_ratio=0.1,
              linewidth=2, label="b")
    ax.quiver(0, 0, 0, p[0], p[1], p[2], color="green", arrow_length_ratio=0.1,
              linewidth=2, label="p = Ax*")
    ax.quiver(p[0], p[1], p[2], r[0], r[1], r[2], color="red",
              arrow_length_ratio=0.15, linewidth=2, label="r = b - p")

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("b = p + r, avec r ⊥ Im(A)")
    ax.legend()
    return ax


def tracer_regression(x, y, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    a, b_coeff = regression_lineaire(x, y)
    y_pred = a + b_coeff * x
    R2 = r_squared(y, y_pred)

    ax.plot(x, y, "ko", markersize=5, label="données")
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, a + b_coeff * x_line, "r-", linewidth=2,
            label=f"y = {a:.2f} + {b_coeff:.2f}x (R² = {R2:.4f})")

    # Résidus
    for xi, yi, yp in zip(x, y, y_pred):
        ax.plot([xi, xi], [yi, yp], "g-", alpha=0.4, linewidth=1)

    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Régression linéaire — moindres carrés")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Projection orthogonale ===")
    A = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)
    x_star = moindres_carres(A, b)
    p = A @ x_star
    r = residu(A, x_star, b)
    print(f"  b = {b}, x* = {x_star}")
    print(f"  p = Ax* = {p} (projection sur le plan xy)")
    print(f"  r = b - p = {r} (composante perpendiculaire)")
    print(f"  ⟨r, col₁(A)⟩ = {np.dot(r, A[:, 0]):.2e} ≈ 0 ✓")
    print(f"  ⟨r, col₂(A)⟩ = {np.dot(r, A[:, 1]):.2e} ≈ 0 ✓")
    print(f"  ||r|| = {np.linalg.norm(r):.4f} = distance de b à Im(A)")

    print(f"\n=== Régression linéaire ===")
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 30)
    y = 2.5 * x + 3 + rng.normal(0, 2, 30)
    a, b_coeff = regression_lineaire(x, y)
    print(f"  y ≈ {a:.3f} + {b_coeff:.3f}x  (vrai : 3 + 2.5x)")
    print(f"  R² = {r_squared(y, a + b_coeff * x):.4f}")

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    tracer_projection_3d(ax1)
    ax2 = fig.add_subplot(122)
    tracer_regression(x, y, ax2)
    plt.tight_layout()
    plt.savefig("least_squares_intro_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
