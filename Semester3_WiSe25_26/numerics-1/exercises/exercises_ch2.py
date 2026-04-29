"""
exercises_ch2.py — Übungen résolues du chapitre 2
==================================================
Référence : Kröger, Numerische Mathematik 1, §2.1–2.5.
Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""
import numpy as np

def uebung_2_6():
    """Übung 2.6 — ||A||_∞ pour matrice complexe."""
    print("=== Übung 2.6 : ||A||_∞ ===")
    A = np.array([[3,0,2],[-4,1+1j*np.sqrt(3),2],[2j,-1,1-1j]])
    row_sums = np.sum(np.abs(A), axis=1)
    print(f"  Sommes par ligne : {row_sums.real}")
    print(f"  ||A||_∞ = {np.max(row_sums).real:.6f} (numpy: {np.linalg.norm(A,np.inf):.6f})\n")

def uebung_2_11():
    """Übung 2.11 — ||A||_2 et ||A||_F."""
    print("=== Übung 2.11 : Spektral- und Frobeniusnorm ===")
    A = np.array([[1,2],[3,4]], dtype=float)
    print(f"  ||A||_F = √(1+4+9+16) = {np.sqrt(30):.6f}")
    print(f"  ||A||_2 = {np.linalg.norm(A,2):.6f}")
    print(f"  ||A||_2 ≤ ||A||_F : {np.linalg.norm(A,2):.4f} ≤ {np.sqrt(30):.4f} ✓\n")

def uebung_2_18():
    """Übung 2.18 — κ_∞(H₃)."""
    print("=== Übung 2.18 : κ_∞(H₃) ===")
    H = np.array([[1,1/2,1/3],[1/2,1/3,1/4],[1/3,1/4,1/5]])
    print(f"  κ_∞(H₃) = {np.linalg.cond(H,np.inf):.2f}\n")

def uebung_2_21():
    """Übung 2.21 — Gauss avec/sans pivot."""
    print("=== Übung 2.21 : Pivot ===")
    A = np.array([[0.00035,1.2654],[1.2547,1.3182]])
    b = np.array([3.5267,6.8541])
    print(f"  Solution : {np.linalg.solve(A,b)}")
    l21 = A[1,0]/A[0,0]
    print(f"  Sans pivot : ℓ₂₁ = {l21:.1f} → Auslöschung")
    print(f"  Avec pivot : ℓ₂₁ = {A[0,0]/A[1,0]:.6f} → stable\n")

def uebung_2_26():
    """Übung 2.26 — Jacobi et GS, 1 pas."""
    print("=== Übung 2.26 : Jacobi/GS 1 pas ===")
    A = np.array([[5,1,-1],[3,-10,2],[1,-2,5]], dtype=float)
    b = np.array([9,8,7], dtype=float)
    x = np.array([1,1,1], dtype=float)
    x_jac = np.array([(b[i]-sum(A[i,j]*x[j] for j in range(3) if j!=i))/A[i,i] for i in range(3)])
    x_gs = x.copy()
    for i in range(3):
        x_gs[i] = (b[i]-sum(A[i,j]*x_gs[j] for j in range(3) if j!=i))/A[i,i]
    print(f"  Jacobi x₁ = {np.round(x_jac,4)}")
    print(f"  GS     x₁ = {np.round(x_gs,4)}")
    print(f"  Exacte    = {np.linalg.solve(A,b)}\n")

if __name__ == "__main__":
    uebung_2_6()
    uebung_2_11()
    uebung_2_18()
    uebung_2_21()
    uebung_2_26()
