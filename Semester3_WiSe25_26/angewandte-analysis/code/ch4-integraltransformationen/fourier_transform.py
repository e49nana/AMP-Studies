"""
fourier_transform.py
====================

Transformée de Fourier continue.

Couvre :
    - TF : F(ω) = ∫ f(t) e^{-iωt} dt
    - TF inverse : f(t) = (1/2π) ∫ F(ω) e^{iωt} dω
    - Propriétés : linéarité, décalage, convolution → produit
    - Spectre d'amplitude |F(ω)| et de phase arg(F(ω))
    - TF de signaux classiques : gaussienne, rectangle, sinus
    - Théorème de Parseval : ∫|f|² = (1/2π)∫|F|²

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# ======================================================================
#  1. TF numérique (approx. par quadrature)
# ======================================================================

def fourier_transform_num(
    f: Callable, omega: np.ndarray, t_range: tuple = (-20, 20),
) -> np.ndarray:
    """F(ω) = ∫ f(t) e^{-iωt} dt (approximation numérique)."""
    F = np.zeros(len(omega), dtype=complex)
    for k, w in enumerate(omega):
        re, _ = integrate.quad(lambda t: f(t) * np.cos(w*t), *t_range)
        im, _ = integrate.quad(lambda t: -f(t) * np.sin(w*t), *t_range)
        F[k] = re + 1j*im
    return F


def inverse_ft_num(
    F: Callable, t: np.ndarray, omega_range: tuple = (-50, 50),
) -> np.ndarray:
    """f(t) = (1/2π) ∫ F(ω) e^{iωt} dω."""
    result = np.zeros(len(t), dtype=complex)
    for k, ti in enumerate(t):
        re, _ = integrate.quad(lambda w: np.real(F(w) * np.exp(1j*w*ti)), *omega_range)
        im, _ = integrate.quad(lambda w: np.imag(F(w) * np.exp(1j*w*ti)), *omega_range)
        result[k] = (re + 1j*im) / (2*np.pi)
    return result


# ======================================================================
#  2. TF analytiques
# ======================================================================

def tf_gaussienne(omega: np.ndarray, sigma: float = 1) -> np.ndarray:
    """f(t) = e^{-t²/(2σ²)} → F(ω) = σ√(2π) e^{-σ²ω²/2}."""
    return sigma * np.sqrt(2*np.pi) * np.exp(-sigma**2 * omega**2 / 2)


def tf_rectangle(omega: np.ndarray, T: float = 1) -> np.ndarray:
    """f(t) = 1 pour |t| < T → F(ω) = 2sin(ωT)/ω = 2T sinc(ωT/π)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(omega) < 1e-15, 2*T, 2*np.sin(omega*T)/omega)


def tf_exponentielle(omega: np.ndarray, a: float = 1) -> np.ndarray:
    """f(t) = e^{-a|t|} → F(ω) = 2a/(a²+ω²)."""
    return 2*a / (a**2 + omega**2)


# ======================================================================
#  3. Propriétés
# ======================================================================

def convolution_num(f: np.ndarray, g: np.ndarray, dt: float) -> np.ndarray:
    """(f * g)(t) = ∫ f(τ)g(t-τ) dτ ≈ FFT⁻¹(FFT(f)·FFT(g))·dt."""
    F = np.fft.fft(f) * dt
    G = np.fft.fft(g) * dt
    return np.real(np.fft.ifft(F * G)) / dt


def parseval_check(f_vals: np.ndarray, F_vals: np.ndarray, dt: float, domega: float) -> dict:
    """∫|f|²dt = (1/2π)∫|F|²dω."""
    lhs = np.sum(np.abs(f_vals)**2) * dt
    rhs = np.sum(np.abs(F_vals)**2) * domega / (2*np.pi)
    return {"∫|f|²": lhs, "(1/2π)∫|F|²": rhs, "ratio": lhs/rhs if rhs > 0 else 0}


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_tf_gaussienne(ax1=None, ax2=None) -> None:
    if ax1 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    t = np.linspace(-5, 5, 300)
    omega = np.linspace(-10, 10, 300)

    for sigma in [0.5, 1, 2]:
        f_t = np.exp(-t**2 / (2*sigma**2))
        F_w = tf_gaussienne(omega, sigma)
        ax1.plot(t, f_t, linewidth=2, label=f"$\\sigma = {sigma}$")
        ax2.plot(omega, F_w, linewidth=2, label=f"$\\sigma = {sigma}$")

    ax1.set_xlabel("$t$"); ax1.set_ylabel("$f(t)$")
    ax1.set_title("Gaussienne : étroite en $t$ → large en $\\omega$")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("$\\omega$"); ax2.set_ylabel("$|F(\\omega)|$")
    ax2.set_title("TF de la gaussienne (aussi gaussienne)")
    ax2.legend(); ax2.grid(True, alpha=0.3)


def tracer_tf_rectangle(ax1=None, ax2=None) -> None:
    if ax1 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    omega = np.linspace(-20, 20, 500)
    for T in [0.5, 1, 2]:
        t = np.linspace(-3, 3, 300)
        f_t = np.where(np.abs(t) < T, 1, 0)
        F_w = tf_rectangle(omega, T)
        ax1.plot(t, f_t + T*0.1, linewidth=2, label=f"$T = {T}$")
        ax2.plot(omega, np.abs(F_w), linewidth=2, label=f"$T = {T}$")

    ax1.set_xlabel("$t$"); ax1.set_title("Rectangle")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("$\\omega$"); ax2.set_title("$|F(\\omega)| = |2\\sin(\\omega T)/\\omega|$")
    ax2.legend(); ax2.grid(True, alpha=0.3)


def tracer_convolution(ax=None) -> plt.Axes:
    """Montre que TF(f*g) = TF(f)·TF(g)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    n = 1024
    dt = 0.05
    t = np.arange(n) * dt - n*dt/2

    f = np.exp(-t**2 / 2)
    g = np.where(np.abs(t) < 1, 1, 0)
    fg_conv = convolution_num(f, g, dt)

    ax.plot(t, f, "b-", linewidth=1.5, alpha=0.5, label="$f$ (gaussienne)")
    ax.plot(t, g, "r-", linewidth=1.5, alpha=0.5, label="$g$ (rectangle)")
    ax.plot(t, fg_conv, "k-", linewidth=2, label="$f * g$ (convolution)")
    ax.set_xlabel("$t$"); ax.set_ylabel("amplitude")
    ax.set_title("Convolution : $f * g$ lisse le rectangle")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    return ax


if __name__ == "__main__":
    print("=== TF de la gaussienne ===\n")
    print("  f(t) = e^{-t²/2} → F(ω) = √(2π) e^{-ω²/2}")
    print("  → La TF d'une gaussienne est une gaussienne !")
    print("  → Principe d'incertitude : σ_t · σ_ω ≥ 1/2")

    print(f"\n=== Vérification numérique ===\n")
    omega = np.linspace(-10, 10, 100)
    F_num = fourier_transform_num(lambda t: np.exp(-t**2/2), omega)
    F_exact = tf_gaussienne(omega, 1)
    print(f"  ||F_num - F_exact||_∞ = {np.max(np.abs(F_num - F_exact)):.2e}")

    print(f"\n=== Parseval ===\n")
    n = 1024; dt = 0.05
    t = np.arange(n)*dt - n*dt/2
    f = np.exp(-t**2/2)
    F = np.fft.fftshift(np.fft.fft(f)) * dt
    domega = 2*np.pi / (n*dt)
    p = parseval_check(f, F, dt, domega)
    print(f"  ∫|f|² = {p['∫|f|²']:.6f}")
    print(f"  (1/2π)∫|F|² = {p['(1/2π)∫|F|²']:.6f}")
    print(f"  Ratio = {p['ratio']:.6f} ≈ 1 ✓")

    print(f"\n=== Convolution ===\n")
    print("  TF(f * g) = TF(f) · TF(g)")
    print("  → La convolution dans le domaine temporel = produit en fréquence")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_tf_gaussienne(axes[0, 0], axes[0, 1])
    tracer_tf_rectangle(axes[1, 0], axes[1, 1])
    plt.tight_layout()
    plt.savefig("fourier_transform_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
