# Polynomial Interpolation (Polynominterpolation)

## 📐 Introduction

Given n+1 data points (x₀,y₀), (x₁,y₁), ..., (xₙ,yₙ) with distinct xᵢ, find a polynomial P(x) of degree ≤ n such that P(xᵢ) = yᵢ for all i. This document covers classical interpolation methods essential for your Numerik exam.

---

## 🎯 1. Existence and Uniqueness

### Theorem

For n+1 distinct points (x₀,y₀), ..., (xₙ,yₙ), there exists **exactly one** polynomial P(x) of degree ≤ n satisfying P(xᵢ) = yᵢ.

### Proof Sketch (Vandermonde)

The interpolation problem is equivalent to solving:

```
| 1  x₀  x₀²  ...  x₀ⁿ | | a₀ |   | y₀ |
| 1  x₁  x₁²  ...  x₁ⁿ | | a₁ |   | y₁ |
| :   :   :   ...   :  | | :  | = | :  |
| 1  xₙ  xₙ²  ...  xₙⁿ | | aₙ |   | yₙ |
```

The Vandermonde matrix is invertible iff all xᵢ are distinct.

```python
import numpy as np

def vandermonde_interpolation(x, y):
    """
    Solve interpolation via Vandermonde matrix.
    WARNING: Numerically unstable for large n!
    """
    n = len(x)
    V = np.vander(x, increasing=True)
    coeffs = np.linalg.solve(V, y)
    return coeffs  # a₀ + a₁x + a₂x² + ...
```

---

## 🔷 2. Lagrange Interpolation

### Formula

```
P(x) = Σᵢ₌₀ⁿ yᵢ · Lᵢ(x)

Where Lagrange basis polynomials:
Lᵢ(x) = Πⱼ≠ᵢ (x - xⱼ)/(xᵢ - xⱼ)
```

### Properties of Lᵢ(x)

- Lᵢ(xⱼ) = δᵢⱼ (Kronecker delta)
- Lᵢ(xᵢ) = 1
- Lᵢ(xⱼ) = 0 for j ≠ i
- deg(Lᵢ) = n

### Python Implementation

```python
import numpy as np

def lagrange_basis(x_points, i, x):
    """
    Compute Lagrange basis polynomial Lᵢ(x).
    
    Parameters:
        x_points: Array of interpolation nodes
        i: Index of basis polynomial
        x: Point(s) to evaluate at
    
    Returns:
        Lᵢ(x)
    """
    n = len(x_points)
    result = np.ones_like(x, dtype=float)
    
    for j in range(n):
        if j != i:
            result *= (x - x_points[j]) / (x_points[i] - x_points[j])
    
    return result


def lagrange_interpolation(x_points, y_points, x):
    """
    Lagrange interpolation polynomial.
    
    Parameters:
        x_points: Interpolation nodes (x₀, ..., xₙ)
        y_points: Function values (y₀, ..., yₙ)
        x: Point(s) to evaluate
    
    Returns:
        P(x)
    """
    x = np.atleast_1d(x)
    n = len(x_points)
    result = np.zeros_like(x, dtype=float)
    
    for i in range(n):
        result += y_points[i] * lagrange_basis(x_points, i, x)
    
    return result


# Example: Interpolate sin(x) at 5 points
x_nodes = np.linspace(0, np.pi, 5)
y_nodes = np.sin(x_nodes)

x_eval = np.linspace(0, np.pi, 100)
y_interp = lagrange_interpolation(x_nodes, y_nodes, x_eval)

# Error
y_exact = np.sin(x_eval)
max_error = np.max(np.abs(y_interp - y_exact))
print(f"Max interpolation error: {max_error:.6e}")
```

### Barycentric Form (Numerically Stable)

```python
def barycentric_weights(x_points):
    """Compute barycentric weights."""
    n = len(x_points)
    w = np.ones(n)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] /= (x_points[i] - x_points[j])
    
    return w


def barycentric_interpolation(x_points, y_points, x):
    """
    Barycentric Lagrange interpolation (stable form).
    
    P(x) = Σᵢ wᵢyᵢ/(x-xᵢ) / Σᵢ wᵢ/(x-xᵢ)
    """
    x = np.atleast_1d(x)
    w = barycentric_weights(x_points)
    
    result = np.zeros_like(x, dtype=float)
    
    for k, xk in enumerate(x):
        # Check if xk is a node
        idx = np.where(np.abs(x_points - xk) < 1e-14)[0]
        if len(idx) > 0:
            result[k] = y_points[idx[0]]
        else:
            terms = w / (xk - x_points)
            result[k] = np.sum(terms * y_points) / np.sum(terms)
    
    return result
```

---

## 📊 3. Newton Interpolation (Dividierte Differenzen)

### Divided Differences

```
f[xᵢ] = yᵢ                                    (0th order)
f[xᵢ,xᵢ₊₁] = (f[xᵢ₊₁] - f[xᵢ])/(xᵢ₊₁ - xᵢ)   (1st order)
f[xᵢ,...,xᵢ₊ₖ] = (f[xᵢ₊₁,...,xᵢ₊ₖ] - f[xᵢ,...,xᵢ₊ₖ₋₁])/(xᵢ₊ₖ - xᵢ)
```

### Newton Form

```
P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...
     = Σₖ₌₀ⁿ f[x₀,...,xₖ] · Πⱼ₌₀ᵏ⁻¹(x-xⱼ)
```

### Advantages over Lagrange

- Adding a new point: Only compute new divided difference
- Efficient evaluation via Horner's scheme
- Natural error estimate

### Python Implementation

```python
def divided_differences(x, y):
    """
    Compute divided differences table.
    
    Returns:
        F: Full divided differences table (lower triangular)
        coeffs: Diagonal elements = Newton coefficients
    """
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y  # First column is y values
    
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i+1, j-1] - F[i, j-1]) / (x[i+j] - x[i])
    
    return F, F[0, :]  # Table and coefficients


def newton_interpolation(x_points, y_points, x):
    """
    Newton interpolation using divided differences.
    
    Parameters:
        x_points: Interpolation nodes
        y_points: Function values
        x: Point(s) to evaluate
    
    Returns:
        P(x)
    """
    x = np.atleast_1d(x)
    _, coeffs = divided_differences(x_points, y_points)
    n = len(coeffs)
    
    # Horner's scheme (nested multiplication)
    result = np.full_like(x, coeffs[-1], dtype=float)
    
    for k in range(n - 2, -1, -1):
        result = result * (x - x_points[k]) + coeffs[k]
    
    return result


# Example with divided differences table
x = np.array([1.0, 2.0, 4.0, 5.0])
y = np.array([1.0, 3.0, 2.0, 4.0])

F, coeffs = divided_differences(x, y)
print("Divided Differences Table:")
print(F)
print(f"\nNewton coefficients: {coeffs}")
```

### Adding Points Efficiently

```python
class NewtonInterpolator:
    """Newton interpolation with efficient point addition."""
    
    def __init__(self):
        self.x = []
        self.y = []
        self.coeffs = []
    
    def add_point(self, x_new, y_new):
        """Add a new interpolation point."""
        self.x.append(x_new)
        self.y.append(y_new)
        
        n = len(self.x)
        
        if n == 1:
            self.coeffs.append(y_new)
        else:
            # Compute new divided difference
            # Need to compute f[x₀,...,xₙ₋₁]
            d = [y_new]
            for k in range(n - 2, -1, -1):
                d.insert(0, (d[0] - self._get_dd(k, n-2-k)) / (x_new - self.x[k]))
            self.coeffs.append(d[0])
    
    def _get_dd(self, start, order):
        """Get divided difference f[x_start, ..., x_{start+order}]."""
        if order == 0:
            return self.y[start]
        # Recompute (could cache for efficiency)
        x_sub = self.x[start:start+order+1]
        y_sub = self.y[start:start+order+1]
        _, c = divided_differences(np.array(x_sub), np.array(y_sub))
        return c[order]
    
    def evaluate(self, x):
        """Evaluate interpolation polynomial."""
        return newton_interpolation(
            np.array(self.x), 
            np.array(self.y), 
            x
        )
```

---

## 📉 4. Interpolation Error

### Error Formula

For f ∈ Cⁿ⁺¹[a,b], the interpolation error is:

```
f(x) - P(x) = f⁽ⁿ⁺¹⁾(ξ)/(n+1)! · Πᵢ₌₀ⁿ(x - xᵢ)

for some ξ ∈ [a,b] depending on x
```

### Error Bound

```
|f(x) - P(x)| ≤ Mₙ₊₁/(n+1)! · |ωₙ₊₁(x)|

Where:
Mₙ₊₁ = max|f⁽ⁿ⁺¹⁾(x)| on [a,b]
ωₙ₊₁(x) = Πᵢ₌₀ⁿ(x - xᵢ)  (nodal polynomial)
```

### Python Implementation

```python
def nodal_polynomial(x_points, x):
    """Compute ω(x) = Π(x - xᵢ)."""
    result = np.ones_like(x, dtype=float)
    for xi in x_points:
        result *= (x - xi)
    return result


def interpolation_error_bound(x_points, M_n1, x):
    """
    Compute error bound for interpolation.
    
    Parameters:
        x_points: Interpolation nodes
        M_n1: Bound on |f^(n+1)(x)|
        x: Evaluation point(s)
    
    Returns:
        Error bound
    """
    n = len(x_points) - 1
    omega = np.abs(nodal_polynomial(x_points, x))
    factorial = np.math.factorial(n + 1)
    return M_n1 / factorial * omega


# Example: Error for sin(x) interpolation
# f(x) = sin(x), |f^(n+1)(x)| ≤ 1 for all n
x_nodes = np.linspace(0, np.pi, 5)
x_eval = np.linspace(0, np.pi, 100)

bound = interpolation_error_bound(x_nodes, 1.0, x_eval)
print(f"Max error bound: {np.max(bound):.6e}")
```

---

## ⚡ 5. Chebyshev Nodes (Tschebyscheff-Knoten)

### The Runge Phenomenon

Equidistant nodes → oscillations at boundaries for high-degree interpolation!

```python
def runge_function(x):
    """Classic example: f(x) = 1/(1+25x²)"""
    return 1 / (1 + 25 * x**2)


def demonstrate_runge():
    """Show Runge phenomenon."""
    import matplotlib.pyplot as plt
    
    x_fine = np.linspace(-1, 1, 500)
    y_exact = runge_function(x_fine)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_fine, y_exact, 'k-', linewidth=2, label='f(x) = 1/(1+25x²)')
    
    for n in [5, 10, 15]:
        # Equidistant nodes
        x_equi = np.linspace(-1, 1, n + 1)
        y_equi = runge_function(x_equi)
        
        y_interp = lagrange_interpolation(x_equi, y_equi, x_fine)
        plt.plot(x_fine, y_interp, '--', label=f'n={n} equidistant')
    
    plt.ylim(-1, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Runge Phenomenon: Equidistant Interpolation Fails')
    plt.legend()
    plt.grid(True)
    plt.savefig('runge_phenomenon.png', dpi=150)
    plt.show()
```

### Chebyshev Nodes

Optimal node placement to minimize |ωₙ₊₁(x)|:

```
xₖ = cos((2k+1)π/(2n+2)), k = 0, 1, ..., n

On interval [a,b]:
xₖ = (a+b)/2 + (b-a)/2 · cos((2k+1)π/(2n+2))
```

### Properties

- Minimize max|ωₙ₊₁(x)| over [-1,1]
- |ωₙ₊₁(x)| ≤ 1/2ⁿ (compared to 1 for equidistant)
- Clustered near endpoints

```python
def chebyshev_nodes(n, a=-1, b=1):
    """
    Generate n+1 Chebyshev nodes on [a,b].
    
    Parameters:
        n: Degree (generates n+1 points)
        a, b: Interval
    
    Returns:
        Array of Chebyshev nodes
    """
    k = np.arange(n + 1)
    nodes = np.cos((2*k + 1) * np.pi / (2*n + 2))
    
    # Transform from [-1,1] to [a,b]
    return (a + b) / 2 + (b - a) / 2 * nodes


def compare_nodes():
    """Compare equidistant vs Chebyshev nodes."""
    import matplotlib.pyplot as plt
    
    x_fine = np.linspace(-1, 1, 500)
    y_exact = runge_function(x_fine)
    
    n = 15
    
    # Equidistant
    x_equi = np.linspace(-1, 1, n + 1)
    y_equi = runge_function(x_equi)
    y_interp_equi = lagrange_interpolation(x_equi, y_equi, x_fine)
    
    # Chebyshev
    x_cheb = chebyshev_nodes(n)
    y_cheb = runge_function(x_cheb)
    y_interp_cheb = lagrange_interpolation(x_cheb, y_cheb, x_fine)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_fine, y_exact, 'k-', linewidth=2, label='Exact')
    plt.plot(x_fine, y_interp_equi, 'r--', label='Equidistant (n=15)')
    plt.plot(x_fine, y_interp_cheb, 'b-', label='Chebyshev (n=15)')
    plt.scatter(x_cheb, y_cheb, c='blue', s=50, zorder=5)
    
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Chebyshev Nodes Eliminate Runge Phenomenon')
    plt.legend()
    plt.grid(True)
    plt.savefig('chebyshev_vs_equidistant.png', dpi=150)
    plt.show()
    
    print(f"Max error (equidistant): {np.max(np.abs(y_interp_equi - y_exact)):.4f}")
    print(f"Max error (Chebyshev): {np.max(np.abs(y_interp_cheb - y_exact)):.6f}")
```

---

## 🌊 6. Spline Interpolation

### Why Splines?

- High-degree polynomials oscillate (Runge)
- Splines: Low-degree piecewise polynomials
- Smooth connections at nodes

### Cubic Spline Definition

On each interval [xᵢ, xᵢ₊₁], S(x) is a cubic polynomial such that:
1. S(xᵢ) = yᵢ (interpolation)
2. S ∈ C²[a,b] (twice continuously differentiable)

### Natural Cubic Spline

Additional conditions: S''(x₀) = S''(xₙ) = 0

### Derivation

Let Mᵢ = S''(xᵢ). On [xᵢ, xᵢ₊₁]:

```
S(x) = Mᵢ(xᵢ₊₁-x)³/(6hᵢ) + Mᵢ₊₁(x-xᵢ)³/(6hᵢ) 
     + (yᵢ - Mᵢhᵢ²/6)(xᵢ₊₁-x)/hᵢ + (yᵢ₊₁ - Mᵢ₊₁hᵢ²/6)(x-xᵢ)/hᵢ

Where hᵢ = xᵢ₊₁ - xᵢ
```

### Linear System for Mᵢ

```
μᵢMᵢ₋₁ + 2Mᵢ + λᵢMᵢ₊₁ = dᵢ,  i = 1,...,n-1

Where:
μᵢ = hᵢ₋₁/(hᵢ₋₁ + hᵢ)
λᵢ = hᵢ/(hᵢ₋₁ + hᵢ)
dᵢ = 6/(hᵢ₋₁ + hᵢ) · [(yᵢ₊₁-yᵢ)/hᵢ - (yᵢ-yᵢ₋₁)/hᵢ₋₁]
```

### Python Implementation

```python
def cubic_spline_natural(x, y):
    """
    Compute natural cubic spline coefficients.
    
    Returns:
        M: Second derivatives at nodes
    """
    n = len(x) - 1
    h = np.diff(x)
    
    # Build tridiagonal system
    mu = np.zeros(n - 1)
    lam = np.zeros(n - 1)
    d = np.zeros(n - 1)
    
    for i in range(n - 1):
        mu[i] = h[i] / (h[i] + h[i + 1])
        lam[i] = h[i + 1] / (h[i] + h[i + 1])
        d[i] = 6 / (h[i] + h[i + 1]) * (
            (y[i + 2] - y[i + 1]) / h[i + 1] - 
            (y[i + 1] - y[i]) / h[i]
        )
    
    # Tridiagonal matrix
    A = np.diag(2 * np.ones(n - 1))
    A += np.diag(lam[:-1], 1)
    A += np.diag(mu[1:], -1)
    
    # Solve for internal M values
    M_internal = np.linalg.solve(A, d)
    
    # Add boundary conditions M₀ = Mₙ = 0
    M = np.zeros(n + 1)
    M[1:-1] = M_internal
    
    return M


def evaluate_cubic_spline(x_nodes, y_nodes, M, x):
    """
    Evaluate cubic spline at point(s) x.
    """
    x = np.atleast_1d(x)
    result = np.zeros_like(x)
    
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)
    
    for k, xk in enumerate(x):
        # Find interval
        i = np.searchsorted(x_nodes, xk) - 1
        i = np.clip(i, 0, n - 1)
        
        # Evaluate spline
        t1 = x_nodes[i + 1] - xk
        t2 = xk - x_nodes[i]
        
        result[k] = (
            M[i] * t1**3 / (6 * h[i]) +
            M[i + 1] * t2**3 / (6 * h[i]) +
            (y_nodes[i] - M[i] * h[i]**2 / 6) * t1 / h[i] +
            (y_nodes[i + 1] - M[i + 1] * h[i]**2 / 6) * t2 / h[i]
        )
    
    return result


# Example
x_nodes = np.array([0, 1, 2, 3, 4])
y_nodes = np.array([0, 1, 0, 1, 0])

M = cubic_spline_natural(x_nodes, y_nodes)
x_fine = np.linspace(0, 4, 100)
y_spline = evaluate_cubic_spline(x_nodes, y_nodes, M, x_fine)
```

### Using SciPy

```python
from scipy.interpolate import CubicSpline

# Natural spline (second derivative = 0 at boundaries)
cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')

# Evaluate
y_spline = cs(x_fine)

# Get derivatives
y_prime = cs(x_fine, 1)   # First derivative
y_double_prime = cs(x_fine, 2)  # Second derivative
```

---

## 📊 7. Comparison of Methods

```python
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def compare_interpolation_methods():
    """Compare all interpolation methods."""
    
    # Test function
    f = lambda x: np.sin(2 * x) * np.exp(-x/3)
    
    x_nodes = np.linspace(0, 2*np.pi, 8)
    y_nodes = f(x_nodes)
    x_fine = np.linspace(0, 2*np.pi, 200)
    y_exact = f(x_fine)
    
    # Methods
    y_lagrange = lagrange_interpolation(x_nodes, y_nodes, x_fine)
    y_newton = newton_interpolation(x_nodes, y_nodes, x_fine)
    
    M = cubic_spline_natural(x_nodes, y_nodes)
    y_spline = evaluate_cubic_spline(x_nodes, y_nodes, M, x_fine)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Interpolation comparison
    axes[0].plot(x_fine, y_exact, 'k-', linewidth=2, label='Exact')
    axes[0].plot(x_fine, y_lagrange, 'r--', label='Lagrange')
    axes[0].plot(x_fine, y_newton, 'g:', linewidth=2, label='Newton')
    axes[0].plot(x_fine, y_spline, 'b-', label='Cubic Spline')
    axes[0].scatter(x_nodes, y_nodes, c='black', s=50, zorder=5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Interpolation Methods Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Error comparison
    axes[1].semilogy(x_fine, np.abs(y_lagrange - y_exact), 'r-', label='Lagrange')
    axes[1].semilogy(x_fine, np.abs(y_spline - y_exact), 'b-', label='Cubic Spline')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('|Error|')
    axes[1].set_title('Interpolation Error')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=150)
    plt.show()
    
    print(f"Max error Lagrange: {np.max(np.abs(y_lagrange - y_exact)):.6e}")
    print(f"Max error Spline: {np.max(np.abs(y_spline - y_exact)):.6e}")
```

### Summary Table

| Method | Degree | Smoothness | Global/Local | Best For |
|--------|--------|------------|--------------|----------|
| Lagrange | n | C^∞ | Global | Exact formulas |
| Newton | n | C^∞ | Global | Adding points |
| Linear Spline | 1 | C⁰ | Local | Simple approx |
| Cubic Spline | 3 | C² | Local | Smooth curves |

---

## 📋 8. Exam Checklist (Klausur)

### Formulas to Know

- [ ] Lagrange: Lᵢ(x) = Πⱼ≠ᵢ (x-xⱼ)/(xᵢ-xⱼ)
- [ ] Divided differences: f[xᵢ,xⱼ] = (f[xⱼ]-f[xᵢ])/(xⱼ-xᵢ)
- [ ] Newton form: P(x) = Σₖ f[x₀,...,xₖ]·Πⱼ<ₖ(x-xⱼ)
- [ ] Error: f(x)-P(x) = f⁽ⁿ⁺¹⁾(ξ)/(n+1)!·ω(x)
- [ ] Chebyshev: xₖ = cos((2k+1)π/(2n+2))

### Key Concepts

- [ ] Uniqueness of interpolating polynomial
- [ ] Runge phenomenon and why it occurs
- [ ] Why Chebyshev nodes are optimal
- [ ] Spline advantages over high-degree polynomials
- [ ] Natural spline boundary conditions

### Common Exam Tasks

- [ ] Construct divided differences table by hand
- [ ] Write Newton polynomial from table
- [ ] Calculate Lagrange basis Lᵢ(x)
- [ ] Estimate interpolation error
- [ ] Explain Runge phenomenon

---

## 🔗 Related Documents

- [01-root-finding.md](./01-root-finding.md) - Root finding methods
- [03-integration.md](./03-integration.md) - Numerical integration
- [04-ode-solvers.md](./04-ode-solvers.md) - ODE solving methods

---

## 📚 References

- Stoer & Bulirsch, "Numerische Mathematik 1", Kapitel 2
- Quarteroni et al., "Numerische Mathematik 1", Kapitel 8
- Burden & Faires, "Numerical Analysis", Chapter 3

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: January 25, 2026*
