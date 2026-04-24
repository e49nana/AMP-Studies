"""
least_squares_fitting.py
========================

Applications concrètes des moindres carrés : ajustement de modèles
paramétriques avec linéarisation.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", sections 5.5.1-5.5.3.

Couvre :
    - Régression linéaire (section 5.5.1)
    - Ajustement polynomial (section 5.5.2)
    - Linéarisation de modèles non-linéaires (section 5.5.3) :
        * y = a · e^{bx}          → ln(y) = ln(a) + bx
        * y = a · x^b             → ln(y) = ln(a) + b·ln(x)
        * y = a / (b + x)         → 1/y = b/a + x/a
    - Coefficient de détermination R²
    - Comparaison QR vs équations normales

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Résultat d'ajustement
# ======================================================================

@dataclass
class ResultatFit:
    """Résultat d'un ajustement aux moindres carrés."""
    parametres: dict[str, float]
    residu: float
    r_squared: float
    modele: str

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v:.6g}" for k, v in self.parametres.items())
        return f"Fit({self.modele}: {params}, R²={self.r_squared:.6f})"


def r_squared(y_data: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient de détermination R² = 1 - SS_res / SS_tot."""
    ss_res = np.sum((y_data - y_pred)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _solve_ls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Moindres carrés via QR (from-scratch via numpy pour simplicité)."""
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R[:A.shape[1]], (Q.T @ b)[:A.shape[1]])


# ======================================================================
#  2. Modèles linéaires (sections 5.5.1, 5.5.2)
# ======================================================================

def fit_lineaire(x: np.ndarray, y: np.ndarray) -> ResultatFit:
    """y = a + bx (section 5.5.1)."""
    A = np.column_stack([np.ones_like(x), x])
    c = _solve_ls(A, y)
    y_pred = c[0] + c[1] * x
    return ResultatFit(
        parametres={"a": c[0], "b": c[1]},
        residu=float(np.linalg.norm(y - y_pred)),
        r_squared=r_squared(y, y_pred),
        modele="y = a + bx",
    )


def fit_polynomial(x: np.ndarray, y: np.ndarray, degre: int) -> ResultatFit:
    """y = a₀ + a₁x + ... + a_d x^d (section 5.5.2)."""
    A = np.column_stack([x**k for k in range(degre + 1)])
    c = _solve_ls(A, y)
    y_pred = A @ c
    params = {f"a{k}": c[k] for k in range(degre + 1)}
    return ResultatFit(
        parametres=params,
        residu=float(np.linalg.norm(y - y_pred)),
        r_squared=r_squared(y, y_pred),
        modele=f"polynôme deg. {degre}",
    )


# ======================================================================
#  3. Modèles non-linéaires linéarisés (section 5.5.3)
# ======================================================================

def fit_exponentiel(x: np.ndarray, y: np.ndarray) -> ResultatFit:
    """
    y = a · e^{bx} → ln(y) = ln(a) + bx.

    Précondition : y > 0.
    """
    log_y = np.log(y)
    A = np.column_stack([np.ones_like(x), x])
    c = _solve_ls(A, log_y)
    a, b = np.exp(c[0]), c[1]
    y_pred = a * np.exp(b * x)
    return ResultatFit(
        parametres={"a": a, "b": b},
        residu=float(np.linalg.norm(y - y_pred)),
        r_squared=r_squared(y, y_pred),
        modele="y = a·exp(bx)",
    )


def fit_puissance(x: np.ndarray, y: np.ndarray) -> ResultatFit:
    """
    y = a · x^b → ln(y) = ln(a) + b·ln(x).

    Précondition : x > 0, y > 0.
    """
    log_x, log_y = np.log(x), np.log(y)
    A = np.column_stack([np.ones_like(log_x), log_x])
    c = _solve_ls(A, log_y)
    a, b = np.exp(c[0]), c[1]
    y_pred = a * x**b
    return ResultatFit(
        parametres={"a": a, "b": b},
        residu=float(np.linalg.norm(y - y_pred)),
        r_squared=r_squared(y, y_pred),
        modele="y = a·x^b",
    )


def fit_hyperbolique(x: np.ndarray, y: np.ndarray) -> ResultatFit:
    """
    y = a / (b + x) → 1/y = b/a + (1/a)x.

    Précondition : y ≠ 0.
    """
    inv_y = 1.0 / y
    A = np.column_stack([np.ones_like(x), x])
    c = _solve_ls(A, inv_y)
    a = 1.0 / c[1]
    b = c[0] * a
    y_pred = a / (b + x)
    return ResultatFit(
        parametres={"a": a, "b": b},
        residu=float(np.linalg.norm(y - y_pred)),
        r_squared=r_squared(y, y_pred),
        modele="y = a/(b+x)",
    )


# ======================================================================
#  4. Sélection automatique du meilleur modèle
# ======================================================================

def meilleur_modele(
    x: np.ndarray, y: np.ndarray,
) -> ResultatFit:
    """Teste plusieurs modèles et renvoie celui avec le meilleur R²."""
    candidats = [fit_lineaire(x, y)]
    candidats.append(fit_polynomial(x, y, 2))

    if np.all(y > 0):
        candidats.append(fit_exponentiel(x, y))
    if np.all(x > 0) and np.all(y > 0):
        candidats.append(fit_puissance(x, y))
    if np.all(y != 0):
        candidats.append(fit_hyperbolique(x, y))

    return max(candidats, key=lambda r: r.r_squared)


# ======================================================================
#  5. Tracé
# ======================================================================

def tracer_fits(
    x: np.ndarray, y: np.ndarray,
    resultats: list[ResultatFit],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, y, "ko", markersize=5, label="données")
    x_fine = np.linspace(x.min(), x.max(), 200)

    for r in resultats:
        p = r.parametres
        if "y = a + bx" in r.modele:
            y_f = p["a"] + p["b"] * x_fine
        elif "exp" in r.modele:
            y_f = p["a"] * np.exp(p["b"] * x_fine)
        elif "x^b" in r.modele:
            y_f = p["a"] * x_fine**p["b"]
        elif "/(b+x)" in r.modele:
            y_f = p["a"] / (p["b"] + x_fine)
        else:
            # Polynôme
            y_f = sum(p[f"a{k}"] * x_fine**k for k in range(len(p)))
        ax.plot(x_fine, y_f, "-", linewidth=1.5,
                label=f"{r.modele} (R²={r.r_squared:.4f})")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Ajustement aux moindres carrés (section 5.5)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== Exemple 1 : données linéaires bruitées ===")
    x1 = np.linspace(0, 5, 30)
    y1 = 2.5 * x1 + 1.0 + rng.normal(0, 0.5, 30)
    res1 = fit_lineaire(x1, y1)
    print(f"  {res1}")

    print("\n=== Exemple 2 : données exponentielles ===")
    x2 = np.linspace(0, 3, 20)
    y2 = 2.0 * np.exp(0.8 * x2) + rng.normal(0, 0.5, 20)
    y2 = np.maximum(y2, 0.1)  # assurer y > 0
    res_lin = fit_lineaire(x2, y2)
    res_exp = fit_exponentiel(x2, y2)
    res_poly = fit_polynomial(x2, y2, 2)
    print(f"  Linéaire    : {res_lin}")
    print(f"  Exponentiel : {res_exp}")
    print(f"  Quadratique : {res_poly}")
    print(f"  Meilleur    : {meilleur_modele(x2, y2)}")

    print("\n=== Exemple 3 : loi de puissance y = 3·x^0.5 ===")
    x3 = np.linspace(0.5, 10, 25)
    y3 = 3.0 * x3**0.5 + rng.normal(0, 0.3, 25)
    y3 = np.maximum(y3, 0.1)
    res_pow = fit_puissance(x3, y3)
    print(f"  {res_pow}")

    print("\n=== Tracés ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_fits(x1, y1, [res1], ax=axes[0])
    axes[0].set_title("Régression linéaire")
    tracer_fits(x2, y2, [res_lin, res_exp, res_poly], ax=axes[1])
    axes[1].set_title("Exp vs Lin vs Poly")
    tracer_fits(x3, y3, [res_pow, fit_lineaire(x3, y3)], ax=axes[2])
    axes[2].set_title("Loi de puissance")
    plt.tight_layout()
    plt.savefig("least_squares_fitting_demo.png", dpi=120)
    print("Figure sauvegardée : least_squares_fitting_demo.png")
