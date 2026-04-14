# ODE Solvers (Gewöhnliche Differentialgleichungen)

## 📐 Introduction

Numerical methods for solving initial value problems (IVP):

```
y'(t) = f(t, y(t)),  y(t₀) = y₀
```

This document covers single-step and multi-step methods with stability analysis, essential for your Numerik exam.

---

## 🎯 1. Problem Classification

### Initial Value Problem (Anfangswertproblem)

```
Find y(t) such that:
y'(t) = f(t, y)
y(t₀) = y₀
```

### Existence and Uniqueness (Picard-Lindelöf)

If f is continuous and Lipschitz in y:

```
|f(t, y₁) - f(t, y₂)| ≤ L|y₁ - y₂|
```

Then there exists a unique solution in some interval [t₀, t₀ + α].

### Systems of ODEs

```
y' = f(t, y),  y ∈ ℝⁿ

Example:
y₁' = y₂
y₂' = -y₁
```

Higher-order ODEs can be converted to first-order systems.

---

## 📏 2. Euler Methods

### Forward Euler (Explizites Euler-Verfahren)

```
yₙ₊₁ = yₙ + h·f(tₙ, yₙ)

Local truncation error: O(h²)
Global error: O(h)
```

### Backward Euler (Implizites Euler-Verfahren)

```
yₙ₊₁ = yₙ + h·f(tₙ₊₁, yₙ₊₁)

Requires solving nonlinear equation at each step!
Local truncation error: O(h²)
Global error: O(h)
```

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def forward_euler(f, t_span, y0, h):
    """
    Forward Euler method.
    
    Parameters:
        f: Function f(t, y)
        t_span: (t0, tf) time interval
        y0: Initial condition (scalar or array)
        h: Step size
    
    Returns:
        t: Time points
        y: Solution values
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    
    y0 = np.atleast_1d(y0)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(n - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    
    return t, y.squeeze()


def backward_euler(f, df_dy, t_span, y0, h, tol=1e-10, max_iter=50):
    """
    Backward Euler method with Newton iteration.
    
    Parameters:
        f: Function f(t, y)
        df_dy: Jacobian ∂f/∂y
        t_span: (t0, tf)
        y0: Initial condition
        h: Step size
    
    Returns:
        t, y
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    
    y = np.zeros(n)
    y[0] = y0
    
    for i in range(n - 1):
        # Newton iteration to solve: y_{n+1} = y_n + h*f(t_{n+1}, y_{n+1})
        y_new = y[i]  # Initial guess
        
        for _ in range(max_iter):
            F = y_new - y[i] - h * f(t[i + 1], y_new)
            dF = 1 - h * df_dy(t[i + 1], y_new)
            
            delta = F / dF
            y_new = y_new - delta
            
            if abs(delta) < tol:
                break
        
        y[i + 1] = y_new
    
    return t, y


# Example: y' = -2y, y(0) = 1, exact: y = e^(-2t)
f = lambda t, y: -2 * y
df_dy = lambda t, y: -2
exact = lambda t: np.exp(-2 * t)

t_span = (0, 3)
y0 = 1
h = 0.5

t_fwd, y_fwd = forward_euler(f, t_span, y0, h)
t_bwd, y_bwd = backward_euler(f, df_dy, t_span, y0, h)
t_exact = np.linspace(0, 3, 100)

plt.figure(figsize=(10, 6))
plt.plot(t_exact, exact(t_exact), 'k-', linewidth=2, label='Exact')
plt.plot(t_fwd, y_fwd, 'bo-', label=f'Forward Euler (h={h})')
plt.plot(t_bwd, y_bwd, 'rs-', label=f'Backward Euler (h={h})')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Euler Methods Comparison')
plt.legend()
plt.grid(True)
plt.savefig('euler_methods.png', dpi=150)
plt.show()
```

---

## 🚀 3. Runge-Kutta Methods

### General s-stage Runge-Kutta

```
yₙ₊₁ = yₙ + h·Σᵢ₌₁ˢ bᵢkᵢ

Where:
k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + c₂h, yₙ + h·a₂₁k₁)
k₃ = f(tₙ + c₃h, yₙ + h·(a₃₁k₁ + a₃₂k₂))
...
kᵢ = f(tₙ + cᵢh, yₙ + h·Σⱼ₌₁ⁱ⁻¹ aᵢⱼkⱼ)
```

### Butcher Tableau

```
c₁ | a₁₁  a₁₂  ...  a₁ₛ
c₂ | a₂₁  a₂₂  ...  a₂ₛ
 : |  :    :   ...   :
cₛ | aₛ₁  aₛ₂  ...  aₛₛ
---+-------------------
   | b₁   b₂   ...  bₛ
```

### Classical Methods

#### Heun's Method (RK2)

```
    0 |
    1 | 1
   ---|------
      | 1/2  1/2

k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + h, yₙ + h·k₁)
yₙ₊₁ = yₙ + h/2·(k₁ + k₂)

Order: 2
```

#### Midpoint Method (RK2)

```
    0 |
  1/2 | 1/2
  ----|------
      | 0    1

k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + h/2, yₙ + h/2·k₁)
yₙ₊₁ = yₙ + h·k₂

Order: 2
```

#### Classical RK4

```
    0 |
  1/2 | 1/2
  1/2 | 0    1/2
    1 | 0    0    1
  ----|----------------
      | 1/6  1/3  1/3  1/6

k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + h/2, yₙ + h/2·k₁)
k₃ = f(tₙ + h/2, yₙ + h/2·k₂)
k₄ = f(tₙ + h, yₙ + h·k₃)
yₙ₊₁ = yₙ + h/6·(k₁ + 2k₂ + 2k₃ + k₄)

Order: 4
```

### Python Implementation

```python
def rk2_heun(f, t_span, y0, h):
    """Heun's method (RK2)."""
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    
    y0 = np.atleast_1d(y0)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(n - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + h / 2 * (k1 + k2)
    
    return t, y.squeeze()


def rk2_midpoint(f, t_span, y0, h):
    """Midpoint method (RK2)."""
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    
    y0 = np.atleast_1d(y0)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(n - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        y[i + 1] = y[i] + h * k2
    
    return t, y.squeeze()


def rk4_classic(f, t_span, y0, h):
    """Classical 4th-order Runge-Kutta."""
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    
    y0 = np.atleast_1d(y0)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(n - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y.squeeze()


# General Runge-Kutta from Butcher tableau
def runge_kutta(f, t_span, y0, h, A, b, c):
    """
    General explicit Runge-Kutta method.
    
    Parameters:
        A: Runge-Kutta matrix (lower triangular for explicit)
        b: Weights
        c: Nodes
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    s = len(b)  # Number of stages
    
    y0 = np.atleast_1d(y0)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(n - 1):
        k = np.zeros((s, len(y0)))
        
        for j in range(s):
            y_stage = y[i] + h * np.sum(A[j, :j, np.newaxis] * k[:j], axis=0)
            k[j] = f(t[i] + c[j] * h, y_stage)
        
        y[i + 1] = y[i] + h * np.sum(b[:, np.newaxis] * k, axis=0)
    
    return t, y.squeeze()
```

### Convergence Comparison

```python
def convergence_test():
    """Compare convergence of different RK methods."""
    
    # Test problem: y' = -y, y(0) = 1, exact: y = e^(-t)
    f = lambda t, y: -y
    exact = lambda t: np.exp(-t)
    
    t_span = (0, 1)
    y0 = 1
    
    methods = [
        ("Forward Euler", forward_euler),
        ("Heun (RK2)", rk2_heun),
        ("Midpoint (RK2)", rk2_midpoint),
        ("RK4 Classic", rk4_classic),
    ]
    
    step_sizes = [0.2, 0.1, 0.05, 0.025, 0.0125]
    
    plt.figure(figsize=(12, 8))
    
    for name, method in methods:
        errors = []
        for h in step_sizes:
            t, y = method(f, t_span, y0, h)
            error = np.max(np.abs(y - exact(t)))
            errors.append(error)
        
        plt.loglog(step_sizes, errors, 'o-', linewidth=2, label=name)
    
    # Reference lines
    h_ref = np.array(step_sizes)
    plt.loglog(h_ref, h_ref, 'k--', alpha=0.5, label='O(h)')
    plt.loglog(h_ref, h_ref**2, 'k-.', alpha=0.5, label='O(h²)')
    plt.loglog(h_ref, h_ref**4, 'k:', alpha=0.5, label='O(h⁴)')
    
    plt.xlabel('Step size h', fontsize=12)
    plt.ylabel('Max error', fontsize=12)
    plt.title('ODE Solver Convergence', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ode_convergence.png', dpi=150)
    plt.show()
```

---

## ⚖️ 4. Stability Analysis

### Test Equation

```
y' = λy,  y(0) = 1,  λ ∈ ℂ

Exact solution: y(t) = e^(λt)
Stable if Re(λ) < 0
```

### Stability Function

For the test equation, one step gives:

```
yₙ₊₁ = R(z)·yₙ,  where z = hλ
```

R(z) is the **stability function**.

### Stability Functions

| Method | R(z) |
|--------|------|
| Forward Euler | 1 + z |
| Backward Euler | 1/(1-z) |
| Heun (RK2) | 1 + z + z²/2 |
| RK4 | 1 + z + z²/2 + z³/6 + z⁴/24 |

### Stability Region

```
S = {z ∈ ℂ : |R(z)| ≤ 1}
```

Method is stable for step sizes h where hλ ∈ S.

### Python Visualization

```python
def plot_stability_regions():
    """Plot stability regions for different methods."""
    
    # Stability functions
    R_euler = lambda z: 1 + z
    R_backward = lambda z: 1 / (1 - z)
    R_heun = lambda z: 1 + z + z**2/2
    R_rk4 = lambda z: 1 + z + z**2/2 + z**3/6 + z**4/24
    
    # Grid
    x = np.linspace(-4, 2, 500)
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    methods = [
        ("Forward Euler", R_euler),
        ("Backward Euler", R_backward),
        ("Heun (RK2)", R_heun),
        ("RK4", R_rk4),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for ax, (name, R) in zip(axes, methods):
        with np.errstate(divide='ignore', invalid='ignore'):
            stability = np.abs(R(Z))
        
        ax.contourf(X, Y, stability, levels=[0, 1], colors=['lightblue'])
        ax.contour(X, Y, stability, levels=[1], colors=['blue'], linewidths=2)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_title(f'{name}: |R(z)| ≤ 1')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stability_regions.png', dpi=150)
    plt.show()


plot_stability_regions()
```

### A-Stability

A method is **A-stable** if S contains the entire left half-plane {z : Re(z) ≤ 0}.

- Forward Euler: NOT A-stable
- Backward Euler: A-stable
- Trapezoidal: A-stable
- RK4: NOT A-stable (but large stability region)

---

## 🔄 5. Multistep Methods

### General Linear Multistep

```
Σⱼ₌₀ᵏ αⱼyₙ₊ⱼ = h·Σⱼ₌₀ᵏ βⱼf(tₙ₊ⱼ, yₙ₊ⱼ)

Explicit if βₖ = 0
Implicit if βₖ ≠ 0
```

### Adams-Bashforth (Explicit)

```
k=1: yₙ₊₁ = yₙ + h·fₙ                           (= Forward Euler)
k=2: yₙ₊₁ = yₙ + h/2·(3fₙ - fₙ₋₁)
k=3: yₙ₊₁ = yₙ + h/12·(23fₙ - 16fₙ₋₁ + 5fₙ₋₂)
k=4: yₙ₊₁ = yₙ + h/24·(55fₙ - 59fₙ₋₁ + 37fₙ₋₂ - 9fₙ₋₃)
```

### Adams-Moulton (Implicit)

```
k=0: yₙ₊₁ = yₙ + h·fₙ₊₁                         (= Backward Euler)
k=1: yₙ₊₁ = yₙ + h/2·(fₙ₊₁ + fₙ)               (= Trapezoidal)
k=2: yₙ₊₁ = yₙ + h/12·(5fₙ₊₁ + 8fₙ - fₙ₋₁)
k=3: yₙ₊₁ = yₙ + h/24·(9fₙ₊₁ + 19fₙ - 5fₙ₋₁ + fₙ₋₂)
```

### BDF Methods (Backward Differentiation)

```
BDF1: yₙ₊₁ - yₙ = h·fₙ₊₁                        (= Backward Euler)
BDF2: 3yₙ₊₁ - 4yₙ + yₙ₋₁ = 2h·fₙ₊₁
BDF3: 11yₙ₊₁ - 18yₙ + 9yₙ₋₁ - 2yₙ₋₂ = 6h·fₙ₊₁
```

### Python Implementation

```python
def adams_bashforth_2(f, t_span, y0, h):
    """Adams-Bashforth 2-step method."""
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    
    y = np.zeros(n)
    y[0] = y0
    
    # Start with RK2 for first step
    k1 = f(t[0], y[0])
    k2 = f(t[0] + h, y[0] + h * k1)
    y[1] = y[0] + h/2 * (k1 + k2)
    
    # Adams-Bashforth steps
    f_prev = f(t[0], y[0])
    for i in range(1, n - 1):
        f_curr = f(t[i], y[i])
        y[i + 1] = y[i] + h/2 * (3*f_curr - f_prev)
        f_prev = f_curr
    
    return t, y


def adams_bashforth_4(f, t_span, y0, h):
    """Adams-Bashforth 4-step method."""
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    
    y = np.zeros(n)
    y[0] = y0
    
    # Start with RK4 for first 3 steps
    for i in range(min(3, n - 1)):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # Store function values
    f_vals = [f(t[i], y[i]) for i in range(4)]
    
    # Adams-Bashforth steps
    for i in range(3, n - 1):
        y[i + 1] = y[i] + h/24 * (55*f_vals[3] - 59*f_vals[2] + 37*f_vals[1] - 9*f_vals[0])
        
        # Shift function values
        f_vals[0] = f_vals[1]
        f_vals[1] = f_vals[2]
        f_vals[2] = f_vals[3]
        f_vals[3] = f(t[i + 1], y[i + 1])
    
    return t, y


def adams_moulton_2(f, df_dy, t_span, y0, h, tol=1e-10):
    """Adams-Moulton 2-step (predictor-corrector with AB2)."""
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    
    y = np.zeros(n)
    y[0] = y0
    
    # First step with trapezoidal
    f0 = f(t[0], y[0])
    y[1] = y[0] + h * f0  # Predictor
    for _ in range(10):  # Corrector iterations
        y[1] = y[0] + h/2 * (f0 + f(t[1], y[1]))
    
    # Main loop: Predict with AB2, Correct with AM2
    f_prev = f0
    for i in range(1, n - 1):
        f_curr = f(t[i], y[i])
        
        # Predict (AB2)
        y_pred = y[i] + h/2 * (3*f_curr - f_prev)
        
        # Correct (AM2)
        for _ in range(10):
            f_pred = f(t[i + 1], y_pred)
            y_pred = y[i] + h/12 * (5*f_pred + 8*f_curr - f_prev)
        
        y[i + 1] = y_pred
        f_prev = f_curr
    
    return t, y
```

---

## 📊 6. Stiff Equations

### What is Stiffness?

A problem is **stiff** if:
- Solution has components with vastly different time scales
- Explicit methods require impractically small step sizes
- Implicit methods work much better

### Example: Stiff System

```
y₁' = -80.6y₁ + 119.4y₂
y₂' = 79.6y₁ - 120.4y₂

Eigenvalues: λ₁ = -1, λ₂ = -200
Stiffness ratio: |λ₂/λ₁| = 200
```

### Python Demonstration

```python
def demonstrate_stiffness():
    """Show stiffness problem with explicit vs implicit."""
    
    # Stiff ODE: y' = -100(y - sin(t)) + cos(t)
    # Solution: y = sin(t) + e^(-100t) (fast transient)
    
    f = lambda t, y: -100*(y - np.sin(t)) + np.cos(t)
    df_dy = lambda t, y: -100
    exact = lambda t: np.sin(t)  # After transient
    
    t_span = (0, 1)
    y0 = 1  # Initial transient
    
    # Try different step sizes with Forward Euler
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    step_sizes = [0.025, 0.019, 0.015]  # Around stability limit h < 2/100 = 0.02
    
    for ax, h in zip(axes, step_sizes):
        try:
            t, y = forward_euler(f, t_span, y0, h)
            ax.plot(t, y, 'b.-', label='Forward Euler')
        except:
            ax.text(0.5, 0.5, 'UNSTABLE', transform=ax.transAxes, 
                   fontsize=20, ha='center')
        
        t_exact = np.linspace(0, 1, 200)
        ax.plot(t_exact, exact(t_exact), 'k-', label='Exact (steady)')
        
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_title(f'h = {h}')
        ax.legend()
        ax.grid(True)
        ax.set_ylim(-3, 3)
    
    plt.suptitle('Stiffness: Forward Euler needs h < 0.02', fontsize=14)
    plt.tight_layout()
    plt.savefig('stiffness_demo.png', dpi=150)
    plt.show()
    
    # Compare with Backward Euler
    plt.figure(figsize=(10, 6))
    
    h = 0.1  # Large step size
    t_fwd, y_fwd = forward_euler(f, t_span, y0, 0.019)  # Just stable
    t_bwd, y_bwd = backward_euler(f, df_dy, t_span, y0, h)
    t_exact = np.linspace(0, 1, 200)
    
    plt.plot(t_exact, exact(t_exact), 'k-', linewidth=2, label='Exact')
    plt.plot(t_fwd, y_fwd, 'b.-', label='Forward Euler (h=0.019)')
    plt.plot(t_bwd, y_bwd, 'ro-', label='Backward Euler (h=0.1)')
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Backward Euler Handles Stiffness')
    plt.legend()
    plt.grid(True)
    plt.savefig('stiff_comparison.png', dpi=150)
    plt.show()
```

---

## 🔧 7. Adaptive Step Size

### Error Estimation

Use embedded RK pair (e.g., RK4(5) Dormand-Prince):

```
Compute two approximations:
ŷₙ₊₁ (order p)
yₙ₊₁ (order p+1)

Error estimate: err = |ŷₙ₊₁ - yₙ₊₁|
```

### Step Size Control

```
h_new = h · (tol / err)^(1/(p+1)) · safety_factor

Typical safety factor: 0.8 - 0.9
```

### Python Implementation

```python
def rk45_adaptive(f, t_span, y0, tol=1e-6, h_init=0.1, h_min=1e-10, h_max=1.0):
    """
    Adaptive RK4(5) Dormand-Prince method.
    """
    # Dormand-Prince coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ])
    
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])  # 5th order
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])  # 4th order
    
    t0, tf = t_span
    t = [t0]
    y0 = np.atleast_1d(y0)
    y = [y0.copy()]
    
    h = h_init
    t_curr = t0
    y_curr = y0.copy()
    
    while t_curr < tf:
        if t_curr + h > tf:
            h = tf - t_curr
        
        # Compute stages
        k = np.zeros((7, len(y0)))
        k[0] = f(t_curr, y_curr)
        
        for i in range(1, 7):
            y_stage = y_curr + h * np.sum(A[i, :i, np.newaxis] * k[:i], axis=0)
            k[i] = f(t_curr + c[i] * h, y_stage)
        
        # Two estimates
        y5 = y_curr + h * np.sum(b5[:, np.newaxis] * k, axis=0)
        y4 = y_curr + h * np.sum(b4[:, np.newaxis] * k, axis=0)
        
        # Error estimate
        err = np.max(np.abs(y5 - y4))
        
        if err < tol or h <= h_min:
            # Accept step
            t_curr += h
            y_curr = y5.copy()
            t.append(t_curr)
            y.append(y_curr.copy())
        
        # Adjust step size
        if err > 0:
            h_new = 0.9 * h * (tol / err) ** 0.2
            h = max(h_min, min(h_max, h_new))
        
    return np.array(t), np.array(y).squeeze()


# Example with adaptive stepping
f = lambda t, y: np.array([-0.5*y[0] + y[1], -y[0] - 0.5*y[1]])
t_span = (0, 20)
y0 = np.array([1, 0])

t, y = rk45_adaptive(f, t_span, y0, tol=1e-8)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, y[:, 0], 'b-', label='y₁')
plt.plot(t, y[:, 1], 'r-', label='y₂')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t[:-1], np.diff(t), 'g.-')
plt.xlabel('t')
plt.ylabel('Step size h')
plt.title('Adaptive Step Sizes')
plt.grid(True)

plt.tight_layout()
plt.savefig('adaptive_rk45.png', dpi=150)
plt.show()
```

---

## 📋 8. Method Summary

| Method | Order | Stages | Stability | Stiff? |
|--------|-------|--------|-----------|--------|
| Forward Euler | 1 | 1 | Conditional | ❌ |
| Backward Euler | 1 | 1 | A-stable | ✅ |
| Heun (RK2) | 2 | 2 | Conditional | ❌ |
| RK4 | 4 | 4 | Conditional | ❌ |
| Adams-Bashforth 4 | 4 | 1* | Conditional | ❌ |
| Adams-Moulton 3 | 4 | 1* | Larger | ⚠️ |
| BDF2 | 2 | 1* | A-stable | ✅ |

*Multistep: uses previous values, not stages

---

## 📋 9. Exam Checklist (Klausur)

### Formulas to Know

- [ ] Forward Euler: yₙ₊₁ = yₙ + h·f(tₙ, yₙ)
- [ ] Backward Euler: yₙ₊₁ = yₙ + h·f(tₙ₊₁, yₙ₊₁)
- [ ] RK4: k₁,k₂,k₃,k₄ formulas + yₙ₊₁ = yₙ + h/6(k₁+2k₂+2k₃+k₄)
- [ ] Stability function R(z) for test equation
- [ ] Adams-Bashforth 2: yₙ₊₁ = yₙ + h/2(3fₙ - fₙ₋₁)

### Key Concepts

- [ ] Local vs global truncation error
- [ ] Order of a method
- [ ] Stability region and A-stability
- [ ] Why implicit methods for stiff problems
- [ ] Butcher tableau interpretation

### Common Exam Tasks

- [ ] One step of Euler/RK by hand
- [ ] Determine order from Butcher tableau
- [ ] Sketch stability region
- [ ] Identify if problem is stiff
- [ ] Choose appropriate method for given problem

---

## 🔗 Related Documents

- [01-root-finding.md](./01-root-finding.md) - Root finding methods
- [02-interpolation.md](./02-interpolation.md) - Polynomial interpolation
- [03-integration.md](./03-integration.md) - Numerical integration

---

## 📚 References

- Stoer & Bulirsch, "Numerische Mathematik 2", Kapitel 7
- Hairer, Nørsett, Wanner, "Solving Ordinary Differential Equations I"
- Burden & Faires, "Numerical Analysis", Chapter 5

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: January 27, 2026*
