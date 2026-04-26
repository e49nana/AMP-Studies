"""
nonlinear_to_linear.py
======================

Linéarisation de modèles non-linéaires pour les moindres carrés.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 5.5.3.

Module standalone qui détaille les transformations avec exemples
physiques concrets : loi de refroidissement, croissance bactérienne,
décroissance radioactive, loi de puissance.

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ModeleLinearise:
    nom: str
    formule: str
    transformation: str
    a: float
    b: float
    r_squared: float


def _r2(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def fit_exponentiel(x, y) -> ModeleLinearise:
    """y = a·exp(bx) → ln(y) = ln(a) + bx."""
    A = np.column_stack([np.ones_like(x), x])
    c, _, _, _ = np.linalg.lstsq(A, np.log(y), rcond=None)
    a, b = np.exp(c[0]), c[1]
    return ModeleLinearise("Exponentiel", "y = a·exp(bx)", "ln(y) = ln(a) + bx",
                           a, b, _r2(y, a * np.exp(b * x)))


def fit_puissance(x, y) -> ModeleLinearise:
    """y = a·x^b → ln(y) = ln(a) + b·ln(x)."""
    A = np.column_stack([np.ones_like(x), np.log(x)])
    c, _, _, _ = np.linalg.lstsq(A, np.log(y), rcond=None)
    a, b = np.exp(c[0]), c[1]
    return ModeleLinearise("Puissance", "y = a·x^b", "ln(y) = ln(a) + b·ln(x)",
                           a, b, _r2(y, a * x**b))


def fit_hyperbolique(x, y) -> ModeleLinearise:
    """y = a/(b+x) → 1/y = b/a + (1/a)x."""
    A = np.column_stack([np.ones_like(x), x])
    c, _, _, _ = np.linalg.lstsq(A, 1.0 / y, rcond=None)
    a = 1.0 / c[1]
    b = c[0] * a
    return ModeleLinearise("Hyperbolique", "y = a/(b+x)", "1/y = b/a + x/a",
                           a, b, _r2(y, a / (b + x)))


def fit_logarithmique(x, y) -> ModeleLinearise:
    """y = a + b·ln(x) → linéaire en (1, ln(x))."""
    A = np.column_stack([np.ones_like(x), np.log(x)])
    c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return ModeleLinearise("Logarithmique", "y = a + b·ln(x)", "déjà linéaire",
                           c[0], c[1], _r2(y, c[0] + c[1] * np.log(x)))


def fit_saturation(x, y) -> ModeleLinearise:
    """y = a·x/(b+x) → 1/y = 1/a + (b/a)·(1/x)."""
    A = np.column_stack([np.ones_like(x), 1.0 / x])
    c, _, _, _ = np.linalg.lstsq(A, 1.0 / y, rcond=None)
    a = 1.0 / c[0]
    b = c[1] * a
    return ModeleLinearise("Saturation", "y = ax/(b+x)", "1/y = 1/a + (b/a)/x",
                           a, b, _r2(y, a * x / (b + x)))


def tracer_linearisation(
    x, y, modele: ModeleLinearise,
    ax_orig: plt.Axes, ax_lin: plt.Axes,
) -> None:
    """Trace l'ajustement dans l'espace original et linéarisé."""
    x_fine = np.linspace(x.min(), x.max(), 200)

    # Espace original
    if "exp" in modele.formule:
        y_pred = modele.a * np.exp(modele.b * x_fine)
    elif "x^b" in modele.formule:
        y_pred = modele.a * x_fine**modele.b
    elif "/(b+x)" in modele.formule and "ax" not in modele.formule:
        y_pred = modele.a / (modele.b + x_fine)
    elif "ln(x)" in modele.formule:
        y_pred = modele.a + modele.b * np.log(x_fine)
    else:
        y_pred = modele.a * x_fine / (modele.b + x_fine)

    ax_orig.plot(x, y, "ko", markersize=4)
    ax_orig.plot(x_fine, y_pred, "r-", linewidth=2)
    ax_orig.set_title(f"{modele.formule} (R²={modele.r_squared:.4f})")
    ax_orig.grid(True, alpha=0.3)

    # Espace linéarisé
    if "exp" in modele.formule:
        ax_lin.plot(x, np.log(y), "ko", markersize=4)
        ax_lin.plot(x_fine, np.log(modele.a) + modele.b * x_fine, "r-")
        ax_lin.set_ylabel("ln(y)")
    elif "x^b" in modele.formule:
        ax_lin.plot(np.log(x), np.log(y), "ko", markersize=4)
        ax_lin.plot(np.log(x_fine), np.log(modele.a) + modele.b * np.log(x_fine), "r-")
        ax_lin.set_xlabel("ln(x)"); ax_lin.set_ylabel("ln(y)")
    else:
        ax_lin.plot(x, 1.0 / y, "ko", markersize=4)
        ax_lin.set_ylabel("1/y")
    ax_lin.set_title(f"Linéarisé : {modele.transformation}")
    ax_lin.grid(True, alpha=0.3)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Exemple 1 : croissance exponentielle
    x1 = np.linspace(0, 4, 25)
    y1 = 3.0 * np.exp(0.5 * x1) + rng.normal(0, 0.3, 25)
    y1 = np.maximum(y1, 0.1)
    m1 = fit_exponentiel(x1, y1)
    print(f"Exponentiel : a={m1.a:.4f} (vrai 3), b={m1.b:.4f} (vrai 0.5), R²={m1.r_squared:.4f}")

    # Exemple 2 : loi de puissance
    x2 = np.linspace(0.5, 10, 30)
    y2 = 2.0 * x2**1.5 + rng.normal(0, 1, 30)
    y2 = np.maximum(y2, 0.1)
    m2 = fit_puissance(x2, y2)
    print(f"Puissance   : a={m2.a:.4f} (vrai 2), b={m2.b:.4f} (vrai 1.5), R²={m2.r_squared:.4f}")

    # Exemple 3 : saturation (Michaelis-Menten)
    x3 = np.linspace(0.5, 20, 20)
    y3 = 10 * x3 / (5 + x3) + rng.normal(0, 0.3, 20)
    y3 = np.maximum(y3, 0.1)
    m3 = fit_saturation(x3, y3)
    print(f"Saturation  : a={m3.a:.4f} (vrai 10), b={m3.b:.4f} (vrai 5), R²={m3.r_squared:.4f}")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    tracer_linearisation(x1, y1, m1, axes[0, 0], axes[0, 1])
    tracer_linearisation(x2, y2, m2, axes[1, 0], axes[1, 1])
    tracer_linearisation(x3, y3, m3, axes[2, 0], axes[2, 1])
    plt.tight_layout()
    plt.savefig("nonlinear_to_linear_demo.png", dpi=120)
    print("Figure sauvegardée.")
