"""
Numerical Differentiation Utilities
====================================
Finite difference methods for approximating derivatives.

Author: Emmanuel Nana Nana
Repo: Scientific-Simulation-Project
"""

import numpy as np
from typing import Callable, Tuple


# =============================================================================
# FIRST DERIVATIVES
# =============================================================================

def forward_difference(f: Callable, x: float, h: float = 1e-5) -> float:
    """
    First derivative using forward difference.
    
    Formula: f'(x) ≈ [f(x+h) - f(x)] / h
    Error: O(h) - First order accurate
    
    Parameters
    ----------
    f : Callable - Function to differentiate
    x : float - Point at which to evaluate derivative
    h : float - Step size (default: 1e-5)
    
    Returns
    -------
    float - Approximate derivative f'(x)
    """
    return (f(x + h) - f(x)) / h


def backward_difference(f: Callable, x: float, h: float = 1e-5) -> float:
    """
    First derivative using backward difference.
    
    Formula: f'(x) ≈ [f(x) - f(x-h)] / h
    Error: O(h) - First order accurate
    """
    return (f(x) - f(x - h)) / h


def central_difference(f: Callable, x: float, h: float = 1e-5) -> float:
    """
    First derivative using central difference.
    
    Formula: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    Error: O(h²) - Second order accurate (more accurate than forward/backward)
    
    This is generally the preferred method for first derivatives.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def five_point_stencil(f: Callable, x: float, h: float = 1e-5) -> float:
    """
    First derivative using five-point stencil.
    
    Formula: f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)
    Error: O(h⁴) - Fourth order accurate
    
    More accurate but requires more function evaluations.
    """
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)


# =============================================================================
# SECOND DERIVATIVES
# =============================================================================

def second_derivative(f: Callable, x: float, h: float = 1e-5) -> float:
    """
    Second derivative using central difference.
    
    Formula: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    Error: O(h²)
    """
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)


def second_derivative_five_point(f: Callable, x: float, h: float = 1e-5) -> float:
    """
    Second derivative using five-point stencil.
    
    Formula: f''(x) ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / (12h²)
    Error: O(h⁴)
    """
    return (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12 * h**2)


# =============================================================================
# MULTIVARIATE: GRADIENT & HESSIAN
# =============================================================================

def gradient(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Numerical gradient for multivariate functions.
    Uses central difference for each component.
    
    Parameters
    ----------
    f : Callable
        Function f: R^n -> R
    x : np.ndarray
        Point at which to evaluate gradient
    h : float
        Step size
        
    Returns
    -------
    np.ndarray
        Gradient vector ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    
    Example
    -------
    >>> f = lambda v: v[0]**2 + v[1]**2  # f(x,y) = x² + y²
    >>> gradient(f, np.array([1.0, 2.0]))
    array([2., 4.])  # ∇f = [2x, 2y] = [2, 4]
    """
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (f(x_forward) - f(x_backward)) / (2 * h)
    
    return grad


def hessian(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Numerical Hessian matrix for multivariate functions.
    
    Parameters
    ----------
    f : Callable
        Function f: R^n -> R
    x : np.ndarray
        Point at which to evaluate Hessian
        
    Returns
    -------
    np.ndarray
        Hessian matrix H where H_ij = ∂²f/∂x_i∂x_j
    
    Example
    -------
    >>> f = lambda v: v[0]**2 + 3*v[1]**2  # f(x,y) = x² + 3y²
    >>> hessian(f, np.array([1.0, 1.0]))
    array([[2., 0.],
           [0., 6.]])  # H = [[2, 0], [0, 6]]
    """
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Use finite difference formula for mixed partials
            x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
            x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
            
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h ** 2)
    
    return H


def jacobian(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Numerical Jacobian matrix for vector-valued functions.
    
    Parameters
    ----------
    f : Callable
        Function f: R^n -> R^m
    x : np.ndarray
        Point at which to evaluate Jacobian
        
    Returns
    -------
    np.ndarray
        Jacobian matrix J where J_ij = ∂f_i/∂x_j
    """
    f0 = np.atleast_1d(f(x))
    m = len(f0)
    n = len(x)
    J = np.zeros((m, n))
    
    for j in range(n):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[j] += h
        x_backward[j] -= h
        J[:, j] = (np.atleast_1d(f(x_forward)) - np.atleast_1d(f(x_backward))) / (2 * h)
    
    return J


# =============================================================================
# LAPLACIAN
# =============================================================================

def laplacian(f: Callable, x: np.ndarray, h: float = 1e-5) -> float:
    """
    Numerical Laplacian (trace of Hessian).
    
    ∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + ... + ∂²f/∂xₙ²
    
    Parameters
    ----------
    f : Callable
        Function f: R^n -> R
    x : np.ndarray
        Point at which to evaluate Laplacian
        
    Returns
    -------
    float
        Laplacian ∇²f(x)
    """
    n = len(x)
    lap = 0.0
    
    for i in range(n):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += h
        x_backward[i] -= h
        lap += (f(x_forward) - 2*f(x) + f(x_backward)) / (h ** 2)
    
    return lap


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def richardson_extrapolation(f: Callable, x: float, h: float, 
                              method: str = 'central') -> Tuple[float, float]:
    """
    Richardson extrapolation for improved accuracy.
    
    Combines two estimates with different step sizes to cancel leading error term.
    
    Parameters
    ----------
    f : Callable - Function to differentiate
    x : float - Point of evaluation
    h : float - Initial step size
    method : str - 'forward' or 'central'
    
    Returns
    -------
    Tuple[float, float] - (improved estimate, error estimate)
    """
    if method == 'central':
        D_h = central_difference(f, x, h)
        D_h2 = central_difference(f, x, h/2)
        # For central difference (O(h²)), extrapolation formula:
        improved = (4 * D_h2 - D_h) / 3
        error = abs(D_h2 - D_h) / 3
    else:  # forward
        D_h = forward_difference(f, x, h)
        D_h2 = forward_difference(f, x, h/2)
        # For forward difference (O(h)), extrapolation formula:
        improved = 2 * D_h2 - D_h
        error = abs(D_h2 - D_h)
    
    return improved, error


# =============================================================================
# EXAMPLE USAGE & TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NUMERICAL DIFFERENTIATION DEMO")
    print("=" * 60)
    
    # --- Test 1: Scalar function ---
    print("\n1. Scalar function: f(x) = x³")
    print("-" * 40)
    
    f = lambda x: x ** 3
    x0 = 2.0
    analytical = 3 * x0**2  # f'(x) = 3x²
    
    print(f"   At x = {x0}:")
    print(f"   Analytical f'(x) = {analytical}")
    print(f"   Forward difference: {forward_difference(f, x0):.10f}")
    print(f"   Central difference: {central_difference(f, x0):.10f}")
    print(f"   Five-point stencil: {five_point_stencil(f, x0):.10f}")
    
    # --- Test 2: Second derivative ---
    print("\n2. Second derivative: f(x) = sin(x)")
    print("-" * 40)
    
    f2 = lambda x: np.sin(x)
    x1 = np.pi / 4
    analytical_2nd = -np.sin(x1)  # f''(x) = -sin(x)
    
    print(f"   At x = π/4:")
    print(f"   Analytical f''(x) = {analytical_2nd:.10f}")
    print(f"   Central (3-point): {second_derivative(f2, x1):.10f}")
    print(f"   Five-point stencil: {second_derivative_five_point(f2, x1):.10f}")
    
    # --- Test 3: Gradient ---
    print("\n3. Gradient: f(x,y) = x² + y²")
    print("-" * 40)
    
    g = lambda v: v[0]**2 + v[1]**2
    point = np.array([1.0, 2.0])
    analytical_grad = np.array([2*point[0], 2*point[1]])  # ∇f = [2x, 2y]
    
    print(f"   At (1, 2):")
    print(f"   Analytical ∇f = {analytical_grad}")
    print(f"   Numerical ∇f  = {gradient(g, point)}")
    
    # --- Test 4: Hessian ---
    print("\n4. Hessian: f(x,y) = x² + 3y² + xy")
    print("-" * 40)
    
    h_func = lambda v: v[0]**2 + 3*v[1]**2 + v[0]*v[1]
    point2 = np.array([1.0, 1.0])
    # H = [[2, 1], [1, 6]]
    
    print(f"   At (1, 1):")
    print(f"   Analytical H = [[2, 1], [1, 6]]")
    print(f"   Numerical H  =")
    H = hessian(h_func, point2)
    print(f"   {H}")
    
    # --- Test 5: Richardson extrapolation ---
    print("\n5. Richardson extrapolation: f(x) = eˣ")
    print("-" * 40)
    
    f_exp = lambda x: np.exp(x)
    x2 = 1.0
    analytical_exp = np.exp(x2)  # f'(eˣ) = eˣ
    
    improved, error = richardson_extrapolation(f_exp, x2, h=0.1)
    central = central_difference(f_exp, x2, h=0.1)
    
    print(f"   At x = 1:")
    print(f"   Analytical f'(x) = {analytical_exp:.10f}")
    print(f"   Central (h=0.1):  {central:.10f} (error: {abs(central - analytical_exp):.2e})")
    print(f"   Richardson:       {improved:.10f} (error: {abs(improved - analytical_exp):.2e})")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
