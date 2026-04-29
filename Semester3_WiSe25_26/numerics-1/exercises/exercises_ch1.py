"""
exercises_ch1.py — Übungen résolues du chapitre 1
==================================================
Référence : Kröger, Numerische Mathematik 1, §1.2–1.4.
Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""
import numpy as np
import matplotlib.pyplot as plt

def uebung_1_3():
    """Übung 1.3 — Boules unitaires pour p = 1, 2, ∞."""
    print("=== Übung 1.3 : Boules unitaires ===")
    theta = np.linspace(0, 2*np.pi, 400)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, p, nom in zip(axes, [1, 2, np.inf], ["p=1", "p=2", "p=∞"]):
        if np.isinf(p):
            normes = np.max(np.abs(pts), axis=1)
        else:
            normes = np.sum(np.abs(pts)**p, axis=1)**(1/p)
        bord = pts / normes[:, None]
        ax.fill(bord[:,0], bord[:,1], alpha=0.3)
        ax.plot(bord[:,0], bord[:,1], linewidth=2)
        ax.set_aspect("equal"); ax.set_title(nom); ax.grid(True, alpha=0.3)
    plt.suptitle("Übung 1.3"); plt.tight_layout()
    plt.savefig("uebung_1_3.png", dpi=120)
    print("  Figure sauvegardée.\n")

def uebung_1_8():
    """Übung 1.8 — cond_rel(x₁/x₂) = 2."""
    print("=== Übung 1.8 : Condition de la division ===")
    print("  φ₁ = (∂f/∂x₁)·x₁/f = 1, φ₂ = -1")
    print("  cond_rel = |φ₁| + |φ₂| = 2 → toujours bien conditionné.\n")

def uebung_1_10():
    """Übung 1.10 — cond_rel(sin x) = |x/tan x|."""
    print("=== Übung 1.10 : Condition de sin(x) ===")
    for x in [0.1, np.pi/4, np.pi/2, np.pi-0.01, np.pi-1e-6]:
        c = abs(x * np.cos(x) / np.sin(x))
        print(f"  x = {x:>10.6f} : cond = {c:>12.2f}")
    print("  → Mal conditionné quand x ≈ kπ.\n")

if __name__ == "__main__":
    uebung_1_3()
    uebung_1_8()
    uebung_1_10()
