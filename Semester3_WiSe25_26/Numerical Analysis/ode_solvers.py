"""
ODE Solvers — Numerical Methods
================================
Implementation of common ordinary differential equation solvers.
From simple Euler to adaptive Runge-Kutta methods.

Author: Emmanuel Nana Nana
Date: January 17, 2026
Repo: AMP-Studies / Scientific-Simulation-Project
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ODESolution:
    """Container for ODE solution results."""
    t: np.ndarray          # Time points
    y: np.ndarray          # Solution values
    method: str            # Method name
    n_steps: int           # Number of steps taken
    n_evals: int           # Number of function evaluations
    
    def plot(self, label: Optional[str] = None, **kwargs):
        """Plot the solution."""
        label = label or self.method
        plt.plot(self.t, self.y, label=label, **kwargs)


# =============================================================================
# BASIC METHODS
# =============================================================================

def euler(f: Callable, y0: float, t_span: Tuple[float, float], 
          n_steps: int = 100) -> ODESolution:
    """
    Euler's method (forward Euler).
    
    Solves: dy/dt = f(t, y), y(t0) = y0
    
    Formula: y_{n+1} = y_n + h * f(t_n, y_n)
    
    Order: O(h) — First order accurate
    
    Parameters
    ----------
    f : Callable
        Function f(t, y) defining the ODE
    y0 : float
        Initial condition
    t_span : Tuple[float, float]
        Time interval (t0, tf)
    n_steps : int
        Number of steps
        
    Returns
    -------
    ODESolution
        Solution container with t, y arrays
    
    Example
    -------
    >>> f = lambda t, y: -2 * y  # dy/dt = -2y
    >>> sol = euler(f, y0=1.0, t_span=(0, 2), n_steps=100)
    >>> # Exact solution: y = e^(-2t)
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    for i in range(n_steps):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    
    return ODESolution(t=t, y=y, method="Euler", n_steps=n_steps, n_evals=n_steps)


def heun(f: Callable, y0: float, t_span: Tuple[float, float],
         n_steps: int = 100) -> ODESolution:
    """
    Heun's method (improved Euler / explicit trapezoidal).
    
    Formula:
        k1 = f(t_n, y_n)
        k2 = f(t_n + h, y_n + h*k1)
        y_{n+1} = y_n + (h/2) * (k1 + k2)
    
    Order: O(h²) — Second order accurate
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + (h / 2) * (k1 + k2)
    
    return ODESolution(t=t, y=y, method="Heun", n_steps=n_steps, n_evals=2*n_steps)


def midpoint(f: Callable, y0: float, t_span: Tuple[float, float],
             n_steps: int = 100) -> ODESolution:
    """
    Midpoint method (explicit).
    
    Formula:
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + (h/2)*k1)
        y_{n+1} = y_n + h * k2
    
    Order: O(h²) — Second order accurate
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + (h/2) * k1)
        y[i + 1] = y[i] + h * k2
    
    return ODESolution(t=t, y=y, method="Midpoint", n_steps=n_steps, n_evals=2*n_steps)


# =============================================================================
# RUNGE-KUTTA METHODS
# =============================================================================

def rk4(f: Callable, y0: float, t_span: Tuple[float, float],
        n_steps: int = 100) -> ODESolution:
    """
    Classic 4th-order Runge-Kutta method (RK4).
    
    The "workhorse" of ODE solvers — excellent balance of accuracy and efficiency.
    
    Formula:
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + (h/2)*k1)
        k3 = f(t_n + h/2, y_n + (h/2)*k2)
        k4 = f(t_n + h, y_n + h*k3)
        y_{n+1} = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    Order: O(h⁴) — Fourth order accurate
    
    Example
    -------
    >>> # Solve: dy/dt = y*cos(t), y(0) = 1
    >>> f = lambda t, y: y * np.cos(t)
    >>> sol = rk4(f, y0=1.0, t_span=(0, 2*np.pi), n_steps=50)
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + (h/2) * k1)
        k3 = f(t[i] + h/2, y[i] + (h/2) * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        
        y[i + 1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return ODESolution(t=t, y=y, method="RK4", n_steps=n_steps, n_evals=4*n_steps)


def rk45_adaptive(f: Callable, y0: float, t_span: Tuple[float, float],
                  tol: float = 1e-6, h_init: float = 0.1,
                  h_min: float = 1e-10, h_max: float = 1.0) -> ODESolution:
    """
    Runge-Kutta-Fehlberg 4(5) with adaptive step size.
    
    Uses embedded RK4 and RK5 formulas to estimate error and adapt step size.
    
    Parameters
    ----------
    f : Callable
        ODE function f(t, y)
    y0 : float
        Initial condition
    t_span : Tuple[float, float]
        Time interval
    tol : float
        Error tolerance
    h_init : float
        Initial step size
    h_min, h_max : float
        Step size bounds
        
    Returns
    -------
    ODESolution
        Solution with adaptive time points
    """
    t0, tf = t_span
    
    # Butcher tableau coefficients for RK45
    c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    a = [
        [],
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]
    b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])      # RK4
    b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])  # RK5
    
    t_list = [t0]
    y_list = [y0]
    
    t = t0
    y = y0
    h = h_init
    n_evals = 0
    
    while t < tf:
        if t + h > tf:
            h = tf - t
        
        # Compute k values
        k = np.zeros(6)
        k[0] = f(t, y)
        k[1] = f(t + c[1]*h, y + h * a[1][0] * k[0])
        k[2] = f(t + c[2]*h, y + h * (a[2][0]*k[0] + a[2][1]*k[1]))
        k[3] = f(t + c[3]*h, y + h * (a[3][0]*k[0] + a[3][1]*k[1] + a[3][2]*k[2]))
        k[4] = f(t + c[4]*h, y + h * (a[4][0]*k[0] + a[4][1]*k[1] + a[4][2]*k[2] + a[4][3]*k[3]))
        k[5] = f(t + c[5]*h, y + h * (a[5][0]*k[0] + a[5][1]*k[1] + a[5][2]*k[2] + a[5][3]*k[3] + a[5][4]*k[4]))
        n_evals += 6
        
        # RK4 and RK5 estimates
        y4 = y + h * np.dot(b4, k)
        y5 = y + h * np.dot(b5, k)
        
        # Error estimate
        error = np.abs(y5 - y4)
        
        # Accept step?
        if error < tol or h <= h_min:
            t = t + h
            y = y5  # Use higher-order estimate
            t_list.append(t)
            y_list.append(y)
        
        # Adjust step size
        if error > 0:
            h_new = 0.9 * h * (tol / error) ** 0.2
            h = max(h_min, min(h_max, h_new))
    
    return ODESolution(
        t=np.array(t_list),
        y=np.array(y_list),
        method="RK45 Adaptive",
        n_steps=len(t_list) - 1,
        n_evals=n_evals
    )


# =============================================================================
# SYSTEMS OF ODEs
# =============================================================================

def rk4_system(f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
               n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    RK4 for systems of ODEs.
    
    Solves: dy/dt = f(t, y), where y is a vector
    
    Parameters
    ----------
    f : Callable
        Function f(t, y) returning array of derivatives
    y0 : np.ndarray
        Initial conditions vector
    t_span : Tuple[float, float]
        Time interval
    n_steps : int
        Number of steps
        
    Returns
    -------
    t : np.ndarray
        Time points
    y : np.ndarray
        Solution array (n_steps+1, n_equations)
    
    Example
    -------
    >>> # Lotka-Volterra predator-prey model
    >>> def lotka_volterra(t, y):
    ...     alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075
    ...     dydt = np.array([
    ...         alpha * y[0] - beta * y[0] * y[1],   # Prey
    ...         delta * y[0] * y[1] - gamma * y[1]   # Predator
    ...     ])
    ...     return dydt
    >>> t, y = rk4_system(lotka_volterra, y0=np.array([10, 5]), t_span=(0, 50))
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    n_eq = len(y0)
    
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, n_eq))
    y[0] = y0
    
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + (h/2) * k1)
        k3 = f(t[i] + h/2, y[i] + (h/2) * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        
        y[i + 1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def convergence_test(f: Callable, y0: float, t_span: Tuple[float, float],
                     exact: Callable, method: Callable,
                     n_steps_list: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test convergence order of a method.
    
    Parameters
    ----------
    f : ODE function
    y0 : Initial condition
    t_span : Time interval
    exact : Exact solution function y(t)
    method : ODE solver to test
    n_steps_list : List of step counts to test
    
    Returns
    -------
    h_values : Step sizes
    errors : Corresponding errors at final time
    """
    if n_steps_list is None:
        n_steps_list = [10, 20, 40, 80, 160, 320]
    
    tf = t_span[1]
    y_exact = exact(tf)
    
    h_values = []
    errors = []
    
    for n in n_steps_list:
        sol = method(f, y0, t_span, n_steps=n)
        h = (t_span[1] - t_span[0]) / n
        error = np.abs(sol.y[-1] - y_exact)
        
        h_values.append(h)
        errors.append(error)
    
    return np.array(h_values), np.array(errors)


def estimate_order(h_values: np.ndarray, errors: np.ndarray) -> float:
    """Estimate convergence order from error data."""
    log_h = np.log(h_values)
    log_e = np.log(errors)
    
    # Linear fit: log(error) = p * log(h) + c
    p, _ = np.polyfit(log_h, log_e, 1)
    return p


# =============================================================================
# EXAMPLE & DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ODE SOLVERS DEMO")
    print("=" * 60)
    
    # --- Test Problem: dy/dt = -2y, y(0) = 1 ---
    # Exact solution: y = e^(-2t)
    
    f = lambda t, y: -2 * y
    exact = lambda t: np.exp(-2 * t)
    y0 = 1.0
    t_span = (0, 3)
    n_steps = 30
    
    print("\n1. Comparing Methods")
    print("-" * 40)
    print(f"   Problem: dy/dt = -2y, y(0) = 1")
    print(f"   Exact solution: y = e^(-2t)")
    print(f"   Steps: {n_steps}")
    
    # Solve with different methods
    sol_euler = euler(f, y0, t_span, n_steps)
    sol_heun = heun(f, y0, t_span, n_steps)
    sol_rk4 = rk4(f, y0, t_span, n_steps)
    
    # Errors at t = 3
    t_final = t_span[1]
    print(f"\n   Errors at t = {t_final}:")
    print(f"   Euler:   {abs(sol_euler.y[-1] - exact(t_final)):.6e}")
    print(f"   Heun:    {abs(sol_heun.y[-1] - exact(t_final)):.6e}")
    print(f"   RK4:     {abs(sol_rk4.y[-1] - exact(t_final)):.6e}")
    
    # --- Convergence Test ---
    print("\n2. Convergence Order Test")
    print("-" * 40)
    
    h_euler, e_euler = convergence_test(f, y0, t_span, exact, euler)
    h_rk4, e_rk4 = convergence_test(f, y0, t_span, exact, rk4)
    
    order_euler = estimate_order(h_euler, e_euler)
    order_rk4 = estimate_order(h_rk4, e_rk4)
    
    print(f"   Euler estimated order: {order_euler:.2f} (expected: 1)")
    print(f"   RK4 estimated order:   {order_rk4:.2f} (expected: 4)")
    
    # --- Adaptive Method ---
    print("\n3. Adaptive Step Size (RK45)")
    print("-" * 40)
    
    sol_adaptive = rk45_adaptive(f, y0, t_span, tol=1e-8)
    print(f"   Steps taken: {sol_adaptive.n_steps}")
    print(f"   Function evals: {sol_adaptive.n_evals}")
    print(f"   Final error: {abs(sol_adaptive.y[-1] - exact(t_final)):.6e}")
    
    # --- System of ODEs: Lotka-Volterra ---
    print("\n4. System of ODEs: Lotka-Volterra")
    print("-" * 40)
    
    def lotka_volterra(t, y):
        alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075
        return np.array([
            alpha * y[0] - beta * y[0] * y[1],
            delta * y[0] * y[1] - gamma * y[1]
        ])
    
    t_lv, y_lv = rk4_system(lotka_volterra, np.array([10.0, 5.0]), (0, 50), n_steps=500)
    print(f"   Initial: Prey = 10, Predator = 5")
    print(f"   Final:   Prey = {y_lv[-1, 0]:.2f}, Predator = {y_lv[-1, 1]:.2f}")
    
    # --- Plot ---
    print("\n5. Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Method comparison
    t_exact = np.linspace(t_span[0], t_span[1], 200)
    axes[0, 0].plot(t_exact, exact(t_exact), 'k-', lw=2, label='Exact')
    axes[0, 0].plot(sol_euler.t, sol_euler.y, 'ro--', ms=4, label='Euler')
    axes[0, 0].plot(sol_heun.t, sol_heun.y, 'gs--', ms=4, label='Heun')
    axes[0, 0].plot(sol_rk4.t, sol_rk4.y, 'b^--', ms=4, label='RK4')
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Method Comparison: dy/dt = -2y')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Convergence
    axes[0, 1].loglog(h_euler, e_euler, 'ro-', label=f'Euler (order ≈ {order_euler:.1f})')
    axes[0, 1].loglog(h_rk4, e_rk4, 'b^-', label=f'RK4 (order ≈ {order_rk4:.1f})')
    axes[0, 1].set_xlabel('Step size h')
    axes[0, 1].set_ylabel('Error at t = 3')
    axes[0, 1].set_title('Convergence Analysis')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Adaptive steps
    axes[1, 0].plot(sol_adaptive.t, sol_adaptive.y, 'g.-', label='RK45 Adaptive')
    axes[1, 0].plot(t_exact, exact(t_exact), 'k--', alpha=0.5, label='Exact')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title(f'Adaptive RK45 ({sol_adaptive.n_steps} steps)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Lotka-Volterra
    axes[1, 1].plot(t_lv, y_lv[:, 0], 'b-', label='Prey')
    axes[1, 1].plot(t_lv, y_lv[:, 1], 'r-', label='Predator')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Population')
    axes[1, 1].set_title('Lotka-Volterra Predator-Prey Model')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ode_solvers_demo.png', dpi=150)
    print("   Saved: ode_solvers_demo.png")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
