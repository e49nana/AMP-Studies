"""
magnetic_field.py
=================

Champ magnétique : Biot-Savart, force de Lorentz.

Couvre :
    - Force de Lorentz : F = qv × B
    - Champ d'un fil infini : B = μ₀I/(2πr)
    - Champ d'un solénoïde : B = μ₀nI (uniforme à l'intérieur)
    - Biot-Savart numérique pour une boucle de courant
    - Mouvement d'une particule chargée dans un champ B (cyclotron)
    - Force entre deux fils parallèles

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


MU0 = 4 * np.pi * 1e-7  # perméabilité du vide (T·m/A)
E_CHARGE = 1.602e-19
M_ELECTRON = 9.109e-31
M_PROTON = 1.673e-27


# ======================================================================
#  1. Champs magnétiques
# ======================================================================

def champ_fil_infini(I: float, r: float) -> float:
    """B = μ₀I/(2πr) pour un fil infini."""
    return MU0 * abs(I) / (2 * np.pi * r)


def champ_solenoide(n: float, I: float) -> float:
    """B = μ₀nI pour un solénoïde (n = tours/mètre)."""
    return MU0 * n * I


def champ_boucle_axe(I: float, R: float, z: float) -> float:
    """B sur l'axe d'une boucle circulaire : B = μ₀IR²/(2(R²+z²)^{3/2})."""
    return MU0 * I * R**2 / (2 * (R**2 + z**2)**1.5)


def force_entre_fils(I1: float, I2: float, d: float, L: float) -> float:
    """Force entre deux fils parallèles de longueur L : F = μ₀I₁I₂L/(2πd)."""
    return MU0 * I1 * I2 * L / (2 * np.pi * d)


# ======================================================================
#  2. Force de Lorentz
# ======================================================================

def force_lorentz(q: float, v: np.ndarray, B: np.ndarray, E: np.ndarray = None) -> np.ndarray:
    """F = q(E + v × B)."""
    F = q * np.cross(v, B)
    if E is not None:
        F += q * np.asarray(E, dtype=float)
    return F


def rayon_cyclotron(m: float, v: float, q: float, B: float) -> float:
    """r = mv/(qB) (rayon de Larmor)."""
    return m * v / (abs(q) * B)


def frequence_cyclotron(q: float, B: float, m: float) -> float:
    """ω_c = qB/m (indépendante de v !)."""
    return abs(q) * B / m


# ======================================================================
#  3. Simulation cyclotron
# ======================================================================

def simuler_cyclotron(
    q: float, m: float, v0: np.ndarray, B: np.ndarray,
    t_end: float, h: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Simule le mouvement d'une particule chargée dans un champ B uniforme."""
    state = np.concatenate([np.zeros(3), np.asarray(v0, dtype=float)])
    n = int(t_end / h)
    positions = np.zeros((n+1, 3))
    positions[0] = state[:3]

    B = np.asarray(B, dtype=float)

    def f(t, s):
        r, v = s[:3], s[3:]
        a = q / m * np.cross(v, B)
        return np.concatenate([v, a])

    for k in range(n):
        k1 = f(0, state)
        k2 = f(0, state+h/2*k1)
        k3 = f(0, state+h/2*k2)
        k4 = f(0, state+h*k3)
        state = state + h/6*(k1+2*k2+2*k3+k4)
        positions[k+1] = state[:3]

    return np.linspace(0, t_end, n+1), positions


# ======================================================================
#  4. Biot-Savart numérique
# ======================================================================

def biot_savart_boucle(
    I: float, R: float, n_segments: int = 200,
    pos: np.ndarray = np.array([0, 0, 0]),
) -> np.ndarray:
    """
    Calcule B en pos dû à une boucle circulaire de rayon R dans le plan xy.
    dB = (μ₀I/4π) · (dl × r̂) / r².
    """
    theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    dtheta = 2*np.pi / n_segments
    B_total = np.zeros(3)

    for th in theta:
        # Position sur la boucle
        r_source = np.array([R*np.cos(th), R*np.sin(th), 0])
        # dl = R dθ (-sin θ, cos θ, 0)
        dl = R * dtheta * np.array([-np.sin(th), np.cos(th), 0])
        # Vecteur de la source vers le point d'observation
        r_vec = pos - r_source
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-15:
            continue
        dB = (MU0 * I / (4 * np.pi)) * np.cross(dl, r_vec) / r_mag**3
        B_total += dB

    return B_total


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_champ_fil(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = np.linspace(0.001, 0.1, 300)
    for I in [1, 5, 10]:
        B = [champ_fil_infini(I, ri) * 1e4 for ri in r]  # en Gauss
        ax.plot(r*100, B, linewidth=2, label=f"$I = {I}$ A")

    ax.set_xlabel("$r$ (cm)"); ax.set_ylabel("$B$ (Gauss)")
    ax.set_title("Champ d'un fil infini : $B = \\mu_0 I / (2\\pi r)$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_cyclotron(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    B_field = np.array([0, 0, 1e-3])  # 1 mT en z
    q = E_CHARGE

    for v_factor in [1, 2, 3]:
        v0 = np.array([v_factor * 1e5, 0, 0])
        r_c = rayon_cyclotron(M_PROTON, np.linalg.norm(v0), q, np.linalg.norm(B_field))
        T_c = 2*np.pi / frequence_cyclotron(q, np.linalg.norm(B_field), M_PROTON)
        t, pos = simuler_cyclotron(q, M_PROTON, v0, B_field, 3*T_c, T_c/500)
        ax.plot(pos[:, 0]*1e3, pos[:, 1]*1e3, linewidth=1.5,
                label=f"$v_0 = {v_factor}\\times 10^5$ m/s ($r = {r_c*1e3:.1f}$ mm)")

    ax.set_xlabel("$x$ (mm)"); ax.set_ylabel("$y$ (mm)")
    ax.set_title("Mouvement cyclotron (proton dans $B = 1$ mT)")
    ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_boucle_axe(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    R = 0.05  # 5 cm
    I = 1.0
    z = np.linspace(-0.2, 0.2, 300)
    B_analytique = [champ_boucle_axe(I, R, zi) * 1e4 for zi in z]
    B_biot = [biot_savart_boucle(I, R, pos=np.array([0, 0, zi]))[2] * 1e4 for zi in z[::10]]

    ax.plot(z*100, B_analytique, "b-", linewidth=2, label="analytique")
    ax.plot(z[::10]*100, B_biot, "ro", markersize=5, label="Biot-Savart (numérique)")
    ax.set_xlabel("$z$ (cm)"); ax.set_ylabel("$B_z$ (Gauss)")
    ax.set_title(f"Boucle de courant ($R={R*100:.0f}$ cm, $I={I}$ A)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Champ d'un fil infini ===\n")
    for I, r in [(1, 0.01), (10, 0.01), (10, 0.1)]:
        B = champ_fil_infini(I, r)
        print(f"  I={I}A, r={r*100}cm : B = {B*1e4:.4f} Gauss")

    print(f"\n=== Solénoïde ===\n")
    n, I = 1000, 2  # 1000 tours/m, 2A
    B = champ_solenoide(n, I)
    print(f"  n={n}/m, I={I}A : B = {B*1e3:.4f} mT = {B*1e4:.4f} Gauss")

    print(f"\n=== Cyclotron (proton) ===\n")
    B_val = 1e-3
    v = 1e5
    r_c = rayon_cyclotron(M_PROTON, v, E_CHARGE, B_val)
    f_c = frequence_cyclotron(E_CHARGE, B_val, M_PROTON)
    print(f"  B = {B_val*1e3} mT, v = {v:.0e} m/s")
    print(f"  r_cyclotron = {r_c*1e3:.2f} mm")
    print(f"  ω_cyclotron = {f_c:.2e} rad/s")
    print(f"  f = {f_c/(2*np.pi):.0f} Hz")
    print(f"  → r ∝ v mais ω indépendant de v !")

    print(f"\n=== Biot-Savart (boucle) ===\n")
    R, I = 0.05, 1
    B_centre = biot_savart_boucle(I, R, pos=np.array([0, 0, 0]))
    B_exact = MU0 * I / (2 * R)
    print(f"  B(centre) = {B_centre[2]*1e4:.6f} Gauss (exact: {B_exact*1e4:.6f})")

    print(f"\n=== Force entre fils ===\n")
    F = force_entre_fils(10, 10, 0.1, 1)
    print(f"  I₁=I₂=10A, d=10cm, L=1m : F = {F*1e3:.4f} mN")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_champ_fil(ax=axes[0])
    tracer_cyclotron(ax=axes[1])
    tracer_boucle_axe(ax=axes[2])
    plt.tight_layout()
    plt.savefig("magnetic_field_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
