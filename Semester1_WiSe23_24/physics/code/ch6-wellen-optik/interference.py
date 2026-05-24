"""
interference.py
===============

Interférence et diffraction.

Couvre :
    - Fentes de Young : deux sources cohérentes
    - Position des maxima : d sin(θ) = mλ
    - Diffraction par une fente : I = I₀ (sin(β)/β)² avec β = πa sin(θ)/λ
    - Réseau de diffraction : N fentes
    - Figure d'interférence 2D

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def young_intensite(
    theta: np.ndarray, d: float, lam: float, I0: float = 1,
) -> np.ndarray:
    """
    Interférence de Young (2 fentes) :
        I = 4I₀ cos²(πd sin(θ)/λ).
    """
    delta = np.pi * d * np.sin(theta) / lam
    return 4 * I0 * np.cos(delta)**2


def young_maxima(d: float, lam: float, n_max: int = 5) -> list[float]:
    """Angles des maxima : d sin(θ) = mλ → θ = arcsin(mλ/d)."""
    maxima = []
    for m in range(-n_max, n_max + 1):
        arg = m * lam / d
        if abs(arg) <= 1:
            maxima.append(np.arcsin(arg))
    return maxima


def diffraction_fente(
    theta: np.ndarray, a: float, lam: float, I0: float = 1,
) -> np.ndarray:
    """
    Diffraction par une fente de largeur a :
        I = I₀ (sin β / β)² avec β = πa sin(θ)/λ.
    """
    beta = np.pi * a * np.sin(theta) / lam
    # sin(β)/β → 1 quand β → 0
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.where(np.abs(beta) < 1e-10, 1.0, np.sin(beta) / beta)
    return I0 * sinc**2


def young_avec_diffraction(
    theta: np.ndarray, d: float, a: float, lam: float, I0: float = 1,
) -> np.ndarray:
    """
    Réaliste : I = diffraction × interférence.
    L'enveloppe de diffraction module les franges d'interférence.
    """
    return diffraction_fente(theta, a, lam, I0) * young_intensite(theta, d, lam, 1)


def reseau_diffraction(
    theta: np.ndarray, d: float, lam: float, N: int, I0: float = 1,
) -> np.ndarray:
    """
    Réseau de N fentes :
        I = I₀ · (sin(Nδ/2) / sin(δ/2))² avec δ = 2πd sin(θ)/λ.
    Pics très fins aux mêmes positions que Young, mais N² × plus intenses.
    """
    delta = np.pi * d * np.sin(theta) / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(N * delta)
        den = np.sin(delta)
        ratio = np.where(np.abs(den) < 1e-12, float(N), num / den)
    return I0 * ratio**2 / N**2


# ======================================================================
#  Tracés
# ======================================================================

def tracer_young(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    lam = 500e-9  # 500 nm vert
    d = 0.1e-3    # 0.1 mm

    theta = np.linspace(-0.02, 0.02, 1000)
    I = young_intensite(theta, d, lam)

    ax.plot(np.degrees(theta)*1000, I, "b-", linewidth=2)
    ax.set_xlabel("angle (mrad)"); ax.set_ylabel("$I / I_0$")
    ax.set_title(f"Young : $d = {d*1e3:.1f}$ mm, $\\lambda = {lam*1e9:.0f}$ nm")
    ax.grid(True, alpha=0.3)

    # Maxima
    maxima = young_maxima(d, lam, 3)
    for m, th in enumerate(maxima):
        if abs(th) < 0.02:
            ax.axvline(np.degrees(th)*1000, color="red", linestyle=":", alpha=0.3)

    return ax


def tracer_diffraction(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    lam = 500e-9
    theta = np.linspace(-0.05, 0.05, 2000)

    for a_um in [10, 20, 50]:
        a = a_um * 1e-6
        I = diffraction_fente(theta, a, lam)
        ax.plot(np.degrees(theta)*1000, I, linewidth=2, label=f"$a = {a_um}$ μm")

    ax.set_xlabel("angle (mrad)"); ax.set_ylabel("$I / I_0$")
    ax.set_title(f"Diffraction par une fente ($\\lambda = {lam*1e9:.0f}$ nm)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_reseau(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    lam = 500e-9
    d = 2e-6  # 2 μm (500 traits/mm)
    theta = np.linspace(-0.3, 0.3, 5000)

    for N in [2, 5, 20]:
        I = reseau_diffraction(theta, d, lam, N)
        ax.plot(np.degrees(theta), I, linewidth=1.5, alpha=0.7, label=f"$N = {N}$")

    ax.set_xlabel("angle (°)"); ax.set_ylabel("$I / I_0$")
    ax.set_title(f"Réseau de diffraction ($d = {d*1e6:.0f}$ μm)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_young_2d(ax: plt.Axes | None = None) -> plt.Axes:
    """Figure d'interférence 2D (écran)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    lam = 500e-9
    d = 0.2e-3
    a = 20e-6
    L = 1.0  # distance à l'écran

    y = np.linspace(-0.02, 0.02, 500)
    theta = np.arctan(y / L)
    I = young_avec_diffraction(theta, d, a, lam)

    # Image 2D
    I_2d = np.outer(np.ones(50), I)
    ax.imshow(I_2d, extent=[y[0]*1e3, y[-1]*1e3, -1, 1], aspect="auto",
               cmap="hot", interpolation="bilinear")
    ax.set_xlabel("position sur l'écran (mm)")
    ax.set_yticks([])
    ax.set_title("Figure d'interférence (Young + diffraction)")
    return ax


if __name__ == "__main__":
    lam = 500e-9

    print("=== Fentes de Young ===\n")
    d = 0.1e-3
    maxima = young_maxima(d, lam, 3)
    print(f"  d = {d*1e3} mm, λ = {lam*1e9:.0f} nm")
    for i, th in enumerate(maxima):
        m = i - len(maxima)//2
        print(f"  m = {m:>2} : θ = {np.degrees(th)*1000:.2f} mrad = {np.degrees(th):.4f}°")

    print(f"\n  Interfrange sur écran à L=1m : Δy = λL/d = {lam*1/d*1e3:.2f} mm")

    print(f"\n=== Diffraction par une fente ===\n")
    a = 20e-6
    theta_1 = np.arcsin(lam / a)
    print(f"  a = {a*1e6:.0f} μm, 1er minimum à θ = {np.degrees(theta_1):.2f}°")
    print(f"  Largeur du pic central : 2λ/a = {2*lam/a*1e3:.2f} mrad")

    print(f"\n=== Réseau de diffraction ===\n")
    d_res = 2e-6  # 500 traits/mm
    for m in [1, 2, 3]:
        arg = m * lam / d_res
        if abs(arg) <= 1:
            th = np.degrees(np.arcsin(arg))
            print(f"  Ordre {m} : θ = {th:.2f}°")
    print(f"  Pouvoir de résolution (N=1000) : R = mN = {1000}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_young(ax=axes[0, 0])
    tracer_diffraction(ax=axes[0, 1])
    tracer_reseau(ax=axes[1, 0])
    tracer_young_2d(ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("interference_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
