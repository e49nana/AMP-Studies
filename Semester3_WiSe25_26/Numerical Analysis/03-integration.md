# Numerical Integration (Numerische Integration)

## 📐 Introduction

Numerical integration approximates definite integrals when analytical solutions are impossible or impractical. This document covers quadrature rules from basic to advanced, with error analysis essential for your Numerik exam.

---

## 🎯 1. Basic Quadrature Rules

### General Form

```
∫ₐᵇ f(x)dx ≈ Σᵢ₌₀ⁿ wᵢ·f(xᵢ)

Where:
xᵢ = Quadrature nodes (Stützstellen)
wᵢ = Weights (Gewichte)
```

### Interpolatory Quadrature

Idea: Integrate the interpolating polynomial instead of f.

```
∫ₐᵇ f(x)dx ≈ ∫ₐᵇ Pₙ(x)dx = Σᵢ₌₀ⁿ wᵢ·f(xᵢ)

Where wᵢ = ∫ₐᵇ Lᵢ(x)dx
```

---

## 📏 2. Newton-Cotes Formulas

### Closed Newton-Cotes (endpoints included)

Using equidistant nodes on [a,b]: xᵢ = a + i·h, h = (b-a)/n

#### Midpoint Rule (n=0)

```
∫ₐᵇ f(x)dx ≈ (b-a)·f((a+b)/2)

Error: E = (b-a)³/24 · f''(ξ)
```

#### Trapezoidal Rule (n=1, Trapezregel)

```
∫ₐᵇ f(x)dx ≈ (b-a)/2 · [f(a) + f(b)]

Error: E = -(b-a)³/12 · f''(ξ)
```

#### Simpson's Rule (n=2, Simpsonregel)

```
∫ₐᵇ f(x)dx ≈ (b-a)/6 · [f(a) + 4f((a+b)/2) + f(b)]

Error: E = -(b-a)⁵/2880 · f⁽⁴⁾(ξ)
```

#### Simpson's 3/8 Rule (n=3)

```
∫ₐᵇ f(x)dx ≈ (b-a)/8 · [f(x₀) + 3f(x₁) + 3f(x₂) + f(x₃)]

Error: E = -(b-a)⁵/6480 · f⁽⁴⁾(ξ)
```

### Python Implementation

```python
import numpy as np

def midpoint_rule(f, a, b):
    """Midpoint rule for integration."""
    return (b - a) * f((a + b) / 2)


def trapezoidal_rule(f, a, b):
    """Trapezoidal rule for integration."""
    return (b - a) / 2 * (f(a) + f(b))


def simpson_rule(f, a, b):
    """Simpson's rule for integration."""
    return (b - a) / 6 * (f(a) + 4*f((a + b) / 2) + f(b))


def simpson_38_rule(f, a, b):
    """Simpson's 3/8 rule for integration."""
    h = (b - a) / 3
    x0, x1, x2, x3 = a, a + h, a + 2*h, b
    return (b - a) / 8 * (f(x0) + 3*f(x1) + 3*f(x2) + f(x3))


# Example: ∫₀¹ e^x dx = e - 1 ≈ 1.71828
f = lambda x: np.exp(x)
exact = np.exp(1) - 1

print(f"Exact:       {exact:.10f}")
print(f"Midpoint:    {midpoint_rule(f, 0, 1):.10f}, Error: {abs(midpoint_rule(f, 0, 1) - exact):.2e}")
print(f"Trapezoidal: {trapezoidal_rule(f, 0, 1):.10f}, Error: {abs(trapezoidal_rule(f, 0, 1) - exact):.2e}")
print(f"Simpson:     {simpson_rule(f, 0, 1):.10f}, Error: {abs(simpson_rule(f, 0, 1) - exact):.2e}")
```

---

## 🔄 3. Composite Rules (Zusammengesetzte Formeln)

### Idea

Divide [a,b] into n subintervals, apply simple rule on each.

### Composite Trapezoidal Rule

```
∫ₐᵇ f(x)dx ≈ h/2 · [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]
           = h · [f(a)/2 + Σᵢ₌₁ⁿ⁻¹ f(xᵢ) + f(b)/2]

Where h = (b-a)/n

Error: E = -(b-a)h²/12 · f''(ξ) = O(h²)
```

### Composite Simpson's Rule

```
∫ₐᵇ f(x)dx ≈ h/3 · [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ)]

Where n must be EVEN, h = (b-a)/n

Error: E = -(b-a)h⁴/180 · f⁽⁴⁾(ξ) = O(h⁴)
```

### Python Implementation

```python
def composite_trapezoidal(f, a, b, n):
    """
    Composite trapezoidal rule.
    
    Parameters:
        f: Function to integrate
        a, b: Integration bounds
        n: Number of subintervals
    
    Returns:
        Approximation of ∫ₐᵇ f(x)dx
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)


def composite_simpson(f, a, b, n):
    """
    Composite Simpson's rule.
    
    Parameters:
        f: Function to integrate
        a, b: Integration bounds
        n: Number of subintervals (must be EVEN)
    
    Returns:
        Approximation of ∫ₐᵇ f(x)dx
    """
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # Coefficients: 1, 4, 2, 4, 2, ..., 4, 1
    result = y[0] + y[-1]
    result += 4 * np.sum(y[1:-1:2])  # Odd indices
    result += 2 * np.sum(y[2:-1:2])  # Even indices (except endpoints)
    
    return h / 3 * result


def composite_midpoint(f, a, b, n):
    """Composite midpoint rule."""
    h = (b - a) / n
    midpoints = a + h * (np.arange(n) + 0.5)
    return h * np.sum(f(midpoints))


# Convergence test
f = lambda x: np.exp(x)
exact = np.exp(1) - 1

print("Convergence Analysis:")
print("-" * 60)
print(f"{'n':<8} {'Trapezoidal':<15} {'Simpson':<15} {'Midpoint':<15}")
print("-" * 60)

for n in [4, 8, 16, 32, 64, 128]:
    trap_err = abs(composite_trapezoidal(f, 0, 1, n) - exact)
    simp_err = abs(composite_simpson(f, 0, 1, n) - exact)
    mid_err = abs(composite_midpoint(f, 0, 1, n) - exact)
    print(f"{n:<8} {trap_err:<15.2e} {simp_err:<15.2e} {mid_err:<15.2e}")
```

### Convergence Visualization

```python
import matplotlib.pyplot as plt

def plot_convergence():
    """Visualize convergence rates."""
    f = lambda x: np.sin(x)
    exact = 1 - np.cos(1)  # ∫₀¹ sin(x)dx
    
    ns = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    hs = 1.0 / ns
    
    trap_errors = [abs(composite_trapezoidal(f, 0, 1, n) - exact) for n in ns]
    simp_errors = [abs(composite_simpson(f, 0, 1, n) - exact) for n in ns]
    mid_errors = [abs(composite_midpoint(f, 0, 1, n) - exact) for n in ns]
    
    plt.figure(figsize=(10, 8))
    plt.loglog(hs, trap_errors, 'bo-', linewidth=2, label='Trapezoidal O(h²)')
    plt.loglog(hs, simp_errors, 'rs-', linewidth=2, label='Simpson O(h⁴)')
    plt.loglog(hs, mid_errors, 'g^-', linewidth=2, label='Midpoint O(h²)')
    
    # Reference lines
    plt.loglog(hs, hs**2 * trap_errors[0]/hs[0]**2, 'b--', alpha=0.5, label='O(h²) ref')
    plt.loglog(hs, hs**4 * simp_errors[0]/hs[0]**4, 'r--', alpha=0.5, label='O(h⁴) ref')
    
    plt.xlabel('Step size h', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Quadrature Convergence Rates', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('quadrature_convergence.png', dpi=150)
    plt.show()
```

---

## ⚡ 4. Romberg Integration

### Richardson Extrapolation

If T(h) has error expansion:

```
T(h) = I + c₁h² + c₂h⁴ + c₃h⁶ + ...
```

Then combine:

```
T₁(h) = (4T(h/2) - T(h)) / 3 = I + O(h⁴)
T₂(h) = (16T₁(h/2) - T₁(h)) / 15 = I + O(h⁶)
```

### Romberg Tableau

```
T₀,₀
T₁,₀  T₁,₁
T₂,₀  T₂,₁  T₂,₂
T₃,₀  T₃,₁  T₃,₂  T₃,₃
...

Where:
Tₖ,₀ = Composite trapezoidal with 2ᵏ subintervals
Tₖ,ⱼ = (4ʲTₖ,ⱼ₋₁ - Tₖ₋₁,ⱼ₋₁) / (4ʲ - 1)
```

### Python Implementation

```python
def romberg(f, a, b, max_iter=10, tol=1e-12):
    """
    Romberg integration.
    
    Parameters:
        f: Function to integrate
        a, b: Integration bounds
        max_iter: Maximum iterations
        tol: Tolerance for convergence
    
    Returns:
        result: Approximation of integral
        R: Full Romberg tableau
    """
    R = np.zeros((max_iter, max_iter))
    
    # First column: composite trapezoidal
    h = b - a
    R[0, 0] = h / 2 * (f(a) + f(b))
    
    for k in range(1, max_iter):
        h /= 2
        n = 2**k
        
        # Add new function evaluations
        x_new = a + h * (2 * np.arange(2**(k-1)) + 1)
        R[k, 0] = R[k-1, 0] / 2 + h * np.sum(f(x_new))
        
        # Richardson extrapolation
        for j in range(1, k + 1):
            factor = 4**j
            R[k, j] = (factor * R[k, j-1] - R[k-1, j-1]) / (factor - 1)
        
        # Check convergence
        if k > 0 and abs(R[k, k] - R[k-1, k-1]) < tol:
            return R[k, k], R[:k+1, :k+1]
    
    return R[-1, -1], R


def print_romberg_tableau(R):
    """Pretty print Romberg tableau."""
    n = R.shape[0]
    print("Romberg Tableau:")
    print("-" * (15 * n))
    
    for i in range(n):
        row = ""
        for j in range(i + 1):
            row += f"{R[i,j]:14.10f} "
        print(row)


# Example
f = lambda x: np.exp(x)
exact = np.exp(1) - 1

result, R = romberg(f, 0, 1, max_iter=6)
print_romberg_tableau(R)
print(f"\nRomberg result: {result:.15f}")
print(f"Exact value:    {exact:.15f}")
print(f"Error:          {abs(result - exact):.2e}")
```

---

## 🎯 5. Gaussian Quadrature (Gauß-Quadratur)

### Idea

Choose nodes AND weights optimally to maximize polynomial exactness.

### Theorem

n-point Gaussian quadrature is exact for polynomials up to degree 2n-1.

### Gauss-Legendre Quadrature

Standard interval [-1, 1]:

```
∫₋₁¹ f(x)dx ≈ Σᵢ₌₁ⁿ wᵢ·f(xᵢ)

Nodes xᵢ = roots of Legendre polynomial Pₙ(x)
Weights wᵢ = 2/[(1-xᵢ²)(P'ₙ(xᵢ))²]
```

### Standard Nodes and Weights

| n | Nodes xᵢ | Weights wᵢ |
|---|----------|------------|
| 1 | 0 | 2 |
| 2 | ±1/√3 ≈ ±0.5773 | 1 |
| 3 | 0, ±√(3/5) ≈ ±0.7746 | 8/9, 5/9 |
| 4 | ±0.3399, ±0.8611 | 0.6521, 0.3479 |
| 5 | 0, ±0.5385, ±0.9062 | 0.5689, 0.4786, 0.2369 |

### Transformation to [a,b]

```
∫ₐᵇ f(x)dx = (b-a)/2 · ∫₋₁¹ f((b-a)t/2 + (a+b)/2)dt

Transformed nodes: x̃ᵢ = (b-a)/2 · xᵢ + (a+b)/2
Transformed weights: w̃ᵢ = (b-a)/2 · wᵢ
```

### Python Implementation

```python
def gauss_legendre_nodes_weights(n):
    """
    Compute Gauss-Legendre nodes and weights.
    
    Uses numpy's built-in function for accuracy.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def gauss_quadrature(f, a, b, n):
    """
    Gauss-Legendre quadrature on [a,b].
    
    Parameters:
        f: Function to integrate
        a, b: Integration bounds
        n: Number of quadrature points
    
    Returns:
        Approximation of ∫ₐᵇ f(x)dx
    """
    nodes, weights = gauss_legendre_nodes_weights(n)
    
    # Transform from [-1,1] to [a,b]
    transformed_nodes = (b - a) / 2 * nodes + (a + b) / 2
    transformed_weights = (b - a) / 2 * weights
    
    return np.sum(transformed_weights * f(transformed_nodes))


# Example: Compare with Newton-Cotes
f = lambda x: np.exp(x)
exact = np.exp(1) - 1

print("Gauss-Legendre vs Simpson:")
print("-" * 50)

for n in [2, 3, 4, 5]:
    gauss_result = gauss_quadrature(f, 0, 1, n)
    gauss_err = abs(gauss_result - exact)
    
    # Simpson needs 2n points for comparison
    simp_result = composite_simpson(f, 0, 1, 2*n)
    simp_err = abs(simp_result - exact)
    
    print(f"n={n}: Gauss error = {gauss_err:.2e}, Simpson({2*n} pts) error = {simp_err:.2e}")
```

### Composite Gauss Quadrature

```python
def composite_gauss(f, a, b, n_intervals, n_points=3):
    """
    Composite Gauss-Legendre quadrature.
    
    Parameters:
        f: Function to integrate
        a, b: Integration bounds
        n_intervals: Number of subintervals
        n_points: Gauss points per interval
    
    Returns:
        Approximation of integral
    """
    h = (b - a) / n_intervals
    result = 0.0
    
    for i in range(n_intervals):
        ai = a + i * h
        bi = ai + h
        result += gauss_quadrature(f, ai, bi, n_points)
    
    return result
```

---

## 🌊 6. Other Gaussian Quadratures

### Gauss-Chebyshev

For integrals with weight function w(x) = 1/√(1-x²):

```
∫₋₁¹ f(x)/√(1-x²) dx ≈ π/n · Σᵢ₌₁ⁿ f(xᵢ)

Nodes: xᵢ = cos((2i-1)π/(2n))
Weights: wᵢ = π/n (all equal!)
```

```python
def gauss_chebyshev(f, n):
    """
    Gauss-Chebyshev quadrature for ∫₋₁¹ f(x)/√(1-x²) dx.
    """
    i = np.arange(1, n + 1)
    nodes = np.cos((2*i - 1) * np.pi / (2*n))
    weights = np.pi / n * np.ones(n)
    
    return np.sum(weights * f(nodes))
```

### Gauss-Laguerre

For integrals on [0,∞) with weight e^(-x):

```
∫₀^∞ f(x)e^(-x) dx ≈ Σᵢ₌₁ⁿ wᵢ·f(xᵢ)
```

```python
def gauss_laguerre(f, n):
    """
    Gauss-Laguerre quadrature for ∫₀^∞ f(x)e^(-x) dx.
    """
    nodes, weights = np.polynomial.laguerre.laggauss(n)
    return np.sum(weights * f(nodes))


# Example: ∫₀^∞ x² e^(-x) dx = 2 (Gamma function)
f = lambda x: x**2
result = gauss_laguerre(f, 5)
print(f"∫₀^∞ x² e^(-x) dx ≈ {result:.10f} (exact: 2)")
```

### Gauss-Hermite

For integrals on (-∞,∞) with weight e^(-x²):

```
∫₋∞^∞ f(x)e^(-x²) dx ≈ Σᵢ₌₁ⁿ wᵢ·f(xᵢ)
```

```python
def gauss_hermite(f, n):
    """
    Gauss-Hermite quadrature for ∫₋∞^∞ f(x)e^(-x²) dx.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    return np.sum(weights * f(nodes))


# Example: ∫₋∞^∞ e^(-x²) dx = √π
f = lambda x: np.ones_like(x)
result = gauss_hermite(f, 5)
print(f"∫₋∞^∞ e^(-x²) dx ≈ {result:.10f} (exact: {np.sqrt(np.pi):.10f})")
```

---

## 📉 7. Error Analysis Summary

### Newton-Cotes Errors

| Rule | Error | Order |
|------|-------|-------|
| Midpoint | (b-a)³/24 · f''(ξ) | O(h²) |
| Trapezoidal | -(b-a)³/12 · f''(ξ) | O(h²) |
| Simpson | -(b-a)⁵/2880 · f⁽⁴⁾(ξ) | O(h⁴) |

### Composite Rules Errors

| Rule | Error | Order |
|------|-------|-------|
| Composite Trap | -(b-a)h²/12 · f''(ξ) | O(h²) |
| Composite Simp | -(b-a)h⁴/180 · f⁽⁴⁾(ξ) | O(h⁴) |
| Composite Mid | (b-a)h²/24 · f''(ξ) | O(h²) |

### Gaussian Quadrature Errors

```
Error for n-point Gauss: E = f⁽²ⁿ⁾(ξ)/(2n)! · ∫₋₁¹ [Pₙ(x)]² dx

≈ (b-a)^(2n+1) · (n!)⁴/[(2n+1)((2n)!)³] · f⁽²ⁿ⁾(ξ)
```

---

## 🔧 8. Adaptive Quadrature

### Idea

Automatically refine where error is large.

```python
def adaptive_simpson(f, a, b, tol=1e-8, max_depth=50):
    """
    Adaptive Simpson quadrature.
    
    Recursively subdivides intervals where error estimate is large.
    """
    def simpson(f, a, b):
        return (b - a) / 6 * (f(a) + 4*f((a+b)/2) + f(b))
    
    def adaptive_helper(a, b, fa, fm, fb, S, tol, depth):
        c = (a + b) / 2
        fc = f(c)
        d = (a + c) / 2
        e = (c + b) / 2
        fd = f(d)
        fe = f(e)
        
        S_left = (c - a) / 6 * (fa + 4*fd + fc)
        S_right = (b - c) / 6 * (fc + 4*fe + fb)
        S_new = S_left + S_right
        
        # Error estimate
        error = (S_new - S) / 15
        
        if depth >= max_depth or abs(error) < tol:
            return S_new + error  # Richardson extrapolation
        
        # Subdivide
        return (adaptive_helper(a, c, fa, fd, fc, S_left, tol/2, depth+1) +
                adaptive_helper(c, b, fc, fe, fb, S_right, tol/2, depth+1))
    
    fa, fb = f(a), f(b)
    fm = f((a + b) / 2)
    S = simpson(f, a, b)
    
    return adaptive_helper(a, b, fa, fm, fb, S, tol, 0)


# Example: Function with varying behavior
f = lambda x: np.sin(10*x) * np.exp(-x)

result_adaptive = adaptive_simpson(f, 0, 3, tol=1e-10)
result_simpson = composite_simpson(f, 0, 3, 100)

# Exact (computed with high precision)
from scipy.integrate import quad
exact, _ = quad(f, 0, 3)

print(f"Adaptive Simpson: {result_adaptive:.12f}")
print(f"Composite Simpson (n=100): {result_simpson:.12f}")
print(f"SciPy quad: {exact:.12f}")
```

---

## 📊 9. Method Comparison

```python
def compare_all_methods():
    """Compare all integration methods."""
    
    # Test function: oscillatory
    f = lambda x: np.sin(5*x) * np.exp(-x/2)
    a, b = 0, 2*np.pi
    
    from scipy.integrate import quad
    exact, _ = quad(f, a, b)
    
    print("Integration Methods Comparison")
    print("=" * 70)
    print(f"Function: sin(5x)·e^(-x/2) on [0, 2π]")
    print(f"Exact value: {exact:.12f}")
    print("-" * 70)
    
    methods = [
        ("Trapezoidal (n=10)", lambda: composite_trapezoidal(f, a, b, 10)),
        ("Trapezoidal (n=100)", lambda: composite_trapezoidal(f, a, b, 100)),
        ("Simpson (n=10)", lambda: composite_simpson(f, a, b, 10)),
        ("Simpson (n=100)", lambda: composite_simpson(f, a, b, 100)),
        ("Gauss-Legendre (n=5)", lambda: gauss_quadrature(f, a, b, 5)),
        ("Gauss-Legendre (n=10)", lambda: gauss_quadrature(f, a, b, 10)),
        ("Composite Gauss (10×3)", lambda: composite_gauss(f, a, b, 10, 3)),
        ("Romberg", lambda: romberg(f, a, b, max_iter=8)[0]),
        ("Adaptive Simpson", lambda: adaptive_simpson(f, a, b, tol=1e-10)),
    ]
    
    for name, method in methods:
        result = method()
        error = abs(result - exact)
        print(f"{name:<25} Result: {result:12.8f}  Error: {error:.2e}")


compare_all_methods()
```

### Summary Table

| Method | Polynomial Exactness | Error Order | Best For |
|--------|---------------------|-------------|----------|
| Trapezoidal | 1 | O(h²) | Simple, smooth f |
| Simpson | 3 | O(h⁴) | General purpose |
| Romberg | Increasing | Exponential | Smooth f, high accuracy |
| Gauss-n | 2n-1 | O(h^(2n)) | Smooth f, few evaluations |
| Adaptive | Varies | Controlled | Varying behavior |

---

## 📋 10. Exam Checklist (Klausur)

### Formulas to Know

- [ ] Trapez: (b-a)/2 · [f(a) + f(b)], Error O(h²)
- [ ] Simpson: (b-a)/6 · [f(a) + 4f(m) + f(b)], Error O(h⁴)
- [ ] Composite Trapez: h·[f₀/2 + f₁ + ... + fₙ₋₁ + fₙ/2]
- [ ] Composite Simpson: h/3·[f₀ + 4f₁ + 2f₂ + 4f₃ + ... + fₙ]
- [ ] Romberg: Tₖ,ⱼ = (4ʲTₖ,ⱼ₋₁ - Tₖ₋₁,ⱼ₋₁)/(4ʲ-1)
- [ ] Gauss-2: nodes ±1/√3, weights 1

### Key Concepts

- [ ] Polynomial exactness (Genauigkeitsgrad)
- [ ] Why Simpson is O(h⁴) not O(h³)
- [ ] Romberg extrapolation principle
- [ ] Gauss nodes = roots of orthogonal polynomials
- [ ] When to use which method

### Common Exam Tasks

- [ ] Compute integral by hand with Trapez/Simpson
- [ ] Build Romberg tableau
- [ ] Calculate Gauss quadrature with given nodes/weights
- [ ] Determine polynomial exactness of a rule
- [ ] Estimate error for given step size

---

## 🔗 Related Documents

- [01-root-finding.md](./01-root-finding.md) - Root finding methods
- [02-interpolation.md](./02-interpolation.md) - Polynomial interpolation
- [04-ode-solvers.md](./04-ode-solvers.md) - ODE solving methods

---

## 📚 References

- Stoer & Bulirsch, "Numerische Mathematik 1", Kapitel 3
- Quarteroni et al., "Numerische Mathematik 1", Kapitel 9
- Burden & Faires, "Numerical Analysis", Chapter 4

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: January 26, 2026*
