"""Iterative methods for solving linear systems Ax=b.

Includes:
- Jacobi (loop + vectorized)
- Gauss–Seidel (loop + NumPy-friendly)
- SOR (loop + NumPy-friendly)

All functions:
- accept array-like A, b, x0
- return the last iterate (or early return on convergence)
- use Euclidean norm on successive iterate difference as stopping criterion
"""

from __future__ import annotations

import numpy as np


def jacobi(A, b, x0, tol: float = 1e-6, max_iter: int = 100):
    """Jacobi method (loop-based)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x0, dtype=float).copy()

    n = b.size
    for _ in range(max_iter):
        x_new = np.zeros(n, dtype=float)
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - s) / A[i, i]

        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new

    return x


def jacobi_vec(A, b, x0, tol: float = 1e-6, max_iter: int = 100):
    """Jacobi method (fully vectorized)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x0, dtype=float).copy()

    D = np.diag(A)
    if np.any(D == 0):
        raise ZeroDivisionError("Zero diagonal entry encountered in A (Jacobi requires nonzero diagonal).")

    R = A - np.diagflat(D)

    for _ in range(max_iter):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new

    return x


def gauss_seidel(A, b, x0, tol: float = 1e-6, max_iter: int = 100):
    """Gauss–Seidel method (loop-based)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x0, dtype=float).copy()

    n = b.size
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])           # newest values
            s2 = np.dot(A[i, i + 1 :], x_old[i + 1 :])  # old values
            x[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x - x_old) < tol:
            return x

    return x


def gauss_seidel_np(A, b, x0, tol: float = 1e-6, max_iter: int = 100):
    """Gauss–Seidel method (NumPy-friendly; still sequential in i)."""
    return gauss_seidel(A, b, x0, tol=tol, max_iter=max_iter)


def sor(A, b, x0, omega: float, tol: float = 1e-6, max_iter: int = 100):
    """SOR (Successive Over-Relaxation) method (loop-based).

    omega=1 -> Gauss–Seidel
    0<omega<1 -> under-relaxation
    1<omega<2 -> over-relaxation (often faster if chosen well)
    """
    if not (0 < omega < 2):
        raise ValueError("omega should typically be in (0, 2) for SOR.")

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x0, dtype=float).copy()

    n = b.size
    for _ in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1 :], x_old[i + 1 :])
            x_gs = (b[i] - s1 - s2) / A[i, i]
            x[i] = (1.0 - omega) * x_old[i] + omega * x_gs

        if np.linalg.norm(x - x_old) < tol:
            return x

    return x


def sor_np(A, b, x0, omega: float, tol: float = 1e-6, max_iter: int = 100):
    """SOR (NumPy-friendly; still sequential in i)."""
    return sor(A, b, x0, omega=omega, tol=tol, max_iter=max_iter)
