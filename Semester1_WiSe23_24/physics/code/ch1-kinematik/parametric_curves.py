"""
parametric_curves.py
====================

Trajectoires paramétriques classiques en physique.

Couvre :
    - Cycloïde : point sur une roue qui roule
    - Spirale d'Archimède : r = a + bθ
    - Figures de Lissajous : x = sin(aθ), y = sin(bθ + δ)
    - Épicycloïde et hypocycloïde (spirographe)
    - Vitesse et accélération le long de courbes paramétriques

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Courbes classiques
# ======================================================================

def cycloide(R: float, n_tours: int = 3, n_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Cycloïde : trajectoire d'un point sur une roue de rayon R.
        x(t) = R(t - sin t), y(t) = R(1 - cos t).
    """
    t = np.linspace(0, 2*np.pi*n_tours, n_points)
    return R * (t - np.sin(t)), R * (1 - np.cos(t))


def spirale_archimede(a: float, b: float, n_tours: int = 5, n_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """r(θ) = a + bθ → x = r cos θ, y = r sin θ."""
    theta = np.linspace(0, 2*np.pi*n_tours, n_points)
    r = a + b * theta
    return r * np.cos(theta), r * np.sin(theta)


def lissajous(a: int, b: int, delta: float, n_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Figures de Lissajous : x = sin(aθ), y = sin(bθ + δ).
    a/b = rapport de fréquences, δ = déphasage.
    """
    t = np.linspace(0, 2*np.pi, n_points)
    return np.sin(a * t), np.sin(b * t + delta)


def epicycloide(R: float, r: float, n_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """
    Épicycloïde : petit cercle (rayon r) roule à l'extérieur du grand (rayon R).
        x = (R+r)cos θ - r cos((R+r)θ/r)
        y = (R+r)sin θ - r sin((R+r)θ/r)
    """
    # Nombre de tours pour fermer la courbe
    from math import gcd
    n = int(r / gcd(int(R), int(r))) if R == int(R) and r == int(r) else 10
    theta = np.linspace(0, 2*np.pi*n, n_points)
    x = (R + r) * np.cos(theta) - r * np.cos((R + r) / r * theta)
    y = (R + r) * np.sin(theta) - r * np.sin((R + r) / r * theta)
    return x, y


def hypocycloide(R: float, r: float, n_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """
    Hypocycloïde : petit cercle roule à l'intérieur du grand.
        x = (R-r)cos θ + r cos((R-r)θ/r)
        y = (R-r)sin θ - r sin((R-r)θ/r)
    R/r = 3 → deltoïde, R/r = 4 → astroïde.
    """
    from math import gcd
    n = int(r / gcd(int(R), int(r))) if R == int(R) and r == int(r) else 10
    theta = np.linspace(0, 2*np.pi*n, n_points)
    x = (R - r) * np.cos(theta) + r * np.cos((R - r) / r * theta)
    y = (R - r) * np.sin(theta) - r * np.sin((R - r) / r * theta)
    return x, y


# ======================================================================
#  2. Vitesse et accélération le long d'une courbe
# ======================================================================

def vitesse_courbe(x: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
    """|v(t)| = √(ẋ² + ẏ²)."""
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    return np.sqrt(vx**2 + vy**2)


def courbure(x: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
    """κ = |ẍẏ - ẋÿ| / (ẋ² + ẏ²)^{3/2}."""
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    num = np.abs(ax * vy - vx * ay)
    den = (vx**2 + vy**2)**1.5
    return num / np.maximum(den, 1e-15)


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_galerie(fig: plt.Figure | None = None) -> plt.Figure:
    """Galerie de courbes paramétriques."""
    if fig is None:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    else:
        axes = fig.subplots(2, 3)

    courbes = [
        ("Cycloïde", *cycloide(1, 3)),
        ("Spirale d'Archimède", *spirale_archimede(0, 0.2, 5)),
        ("Lissajous (3:2, δ=π/4)", *lissajous(3, 2, np.pi/4)),
        ("Épicycloïde (5:2)", *epicycloide(5, 2)),
        ("Hypocycloïde (astroïde)", *hypocycloide(4, 1)),
        ("Lissajous (5:4, δ=π/2)", *lissajous(5, 4, np.pi/2)),
    ]

    for ax, (nom, x, y) in zip(axes.flat, courbes):
        ax.plot(x, y, linewidth=1.5, color="tab:blue")
        ax.set_title(nom)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Galerie de courbes paramétriques", fontsize=14)
    fig.tight_layout()
    return fig


def tracer_cycloide_detaillee(ax: plt.Axes | None = None) -> plt.Axes:
    """Cycloïde avec la roue qui roule."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    R = 1
    x, y = cycloide(R, 2, 500)
    ax.plot(x, y, "b-", linewidth=2, label="cycloïde")
    ax.axhline(0, color="brown", linewidth=2)

    # Positions de la roue
    for t in np.linspace(0, 4*np.pi, 9):
        cx = R * t
        cy = R
        theta = np.linspace(0, 2*np.pi, 50)
        ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), "grey", linewidth=0.5, alpha=0.4)
        # Point sur la roue
        px = R*(t - np.sin(t))
        py = R*(1 - np.cos(t))
        ax.plot(px, py, "ro", markersize=4)

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Cycloïde : point sur une roue qui roule")
    ax.set_ylim(-0.3, 2.5)
    return ax


if __name__ == "__main__":
    print("=== Cycloïde ===")
    R = 1
    x, y = cycloide(R)
    dt = 2*np.pi*3 / len(x)
    v = vitesse_courbe(x, y, dt)
    print(f"  R = {R}")
    print(f"  v_min = {v[10:-10].min():.4f} (en haut, quand le point est au sommet)")
    print(f"  v_max = {v[10:-10].max():.4f} (en bas, quand le point touche le sol)")

    print(f"\n=== Lissajous ===")
    for a, b, delta in [(1, 1, np.pi/2), (2, 3, 0), (3, 2, np.pi/4)]:
        x, y = lissajous(a, b, delta)
        print(f"  a={a}, b={b}, δ={np.degrees(delta):.0f}° → "
              f"courbe {'fermée' if a != b else 'ellipse'}")

    print(f"\n=== Spirographe ===")
    for R_big, r_small, nom in [(5, 2, "épi"), (4, 1, "astroïde"), (3, 1, "deltoïde")]:
        if nom == "épi":
            x, y = epicycloide(R_big, r_small)
        else:
            x, y = hypocycloide(R_big, r_small)
        print(f"  R={R_big}, r={r_small} ({nom}) → {int(R_big/r_small)} pointes")

    fig = tracer_galerie()
    fig.savefig("parametric_curves_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
