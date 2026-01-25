# Root Finding Methods (Nullstellenberechnung)

## 📐 Introduction

Root finding is fundamental to numerical analysis: given f(x), find x* such that f(x*) = 0. This document covers the classical methods with Python implementations, convergence analysis, and practical considerations for your Numerik exam.

---

## 🎯 1. Bisection Method (Bisektion)

### Concept

If f is continuous on [a,b] and f(a)·f(b) < 0, then there exists at least one root in (a,b).

```
Algorithm:
1. Check: f(a)·f(b) < 0
2. Compute midpoint: c = (a+b)/2
3. If f(c) = 0 → done
4. If f(a)·f(c) < 0 → root in [a,c], set b = c
5. Else → root in [c,b], set a = c
6. Repeat until |b-a| < tolerance
```

### Convergence

- **Order**: Linear (p = 1)
- **Rate**: Error halves each iteration
- **Iterations needed**: n ≥ log₂((b-a)/ε)

### Python Implementation

```python
import numpy as np

def bisection(f, a, b, tol=1e-10, max_iter=100):
    """
    Bisection method for root finding.
    
    Parameters:
        f: Function f(x)
        a, b: Interval bounds where f(a)*f(b) < 0
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        root: Approximated root
        iterations: Number of iterations
        history: List of (a, b, c, f(c)) tuples
    """
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    history = []
    
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        history.append((a, b, c, fc))
        
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c, i + 1, history
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    return c, max_iter, history


# Example: Find root of f(x) = x³ - x - 2
f = lambda x: x**3 - x - 2

root, iters, hist = bisection(f, 1, 2)
print(f"Root: {root:.10f}")
print(f"Iterations: {iters}")
print(f"f(root) = {f(root):.2e}")
```

### Convergence Visualization

```python
import matplotlib.pyplot as plt

def plot_bisection_convergence(history, true_root):
    """Plot error vs iterations for bisection."""
    errors = [abs(h[2] - true_root) for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(errors) + 1), errors, 'bo-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Error |xₙ - x*|')
    plt.title('Bisection: Linear Convergence')
    plt.grid(True)
    plt.savefig('bisection_convergence.png', dpi=150)
    plt.show()
```

---

## 🚀 2. Newton-Raphson Method (Newton-Verfahren)

### Concept

Use tangent line approximation to find better estimates:

```
xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
```

Geometrically: Draw tangent at (xₙ, f(xₙ)), find where it crosses x-axis.

### Convergence

- **Order**: Quadratic (p = 2) near simple roots
- **Rate**: Digits of accuracy roughly double each iteration
- **Condition**: Requires f'(x) ≠ 0 near root

### Error Analysis

For simple root with f'(x*) ≠ 0:

```
eₙ₊₁ ≈ (f''(x*) / 2f'(x*)) · eₙ²
```

### Python Implementation

```python
def newton_raphson(f, df, x0, tol=1e-10, max_iter=50):
    """
    Newton-Raphson method for root finding.
    
    Parameters:
        f: Function f(x)
        df: Derivative f'(x)
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        root: Approximated root
        iterations: Number of iterations
        history: List of (x, f(x), f'(x)) tuples
    """
    x = x0
    history = []
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        history.append((x, fx, dfx))
        
        if abs(fx) < tol:
            return x, i + 1, history
        
        if abs(dfx) < 1e-15:
            raise ValueError(f"Derivative near zero at x = {x}")
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol:
            return x_new, i + 1, history
        
        x = x_new
    
    return x, max_iter, history


# Example: f(x) = x³ - x - 2, f'(x) = 3x² - 1
f = lambda x: x**3 - x - 2
df = lambda x: 3*x**2 - 1

root, iters, hist = newton_raphson(f, df, 1.5)
print(f"Root: {root:.15f}")
print(f"Iterations: {iters}")
```

### Newton with Numerical Derivative

```python
def newton_numerical(f, x0, tol=1e-10, max_iter=50, h=1e-8):
    """Newton's method with numerical derivative."""
    x = x0
    
    for i in range(max_iter):
        fx = f(x)
        
        if abs(fx) < tol:
            return x, i + 1
        
        # Central difference approximation
        dfx = (f(x + h) - f(x - h)) / (2 * h)
        
        if abs(dfx) < 1e-15:
            raise ValueError("Derivative near zero")
        
        x = x - fx / dfx
    
    return x, max_iter
```

---

## 📊 3. Secant Method (Sekantenverfahren)

### Concept

Approximate f'(xₙ) using finite difference:

```
f'(xₙ) ≈ (f(xₙ) - f(xₙ₋₁)) / (xₙ - xₙ₋₁)

xₙ₊₁ = xₙ - f(xₙ) · (xₙ - xₙ₋₁) / (f(xₙ) - f(xₙ₋₁))
```

### Convergence

- **Order**: Superlinear (p ≈ 1.618, the golden ratio φ)
- **Rate**: Faster than bisection, slower than Newton
- **Advantage**: No derivative needed

### Python Implementation

```python
def secant_method(f, x0, x1, tol=1e-10, max_iter=50):
    """
    Secant method for root finding.
    
    Parameters:
        f: Function f(x)
        x0, x1: Two initial guesses
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        root: Approximated root
        iterations: Number of iterations
        history: List of x values
    """
    history = [x0, x1]
    
    for i in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        
        if abs(fx1) < tol:
            return x1, i + 1, history
        
        if abs(fx1 - fx0) < 1e-15:
            raise ValueError("Division by zero in secant formula")
        
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        history.append(x2)
        
        if abs(x2 - x1) < tol:
            return x2, i + 1, history
        
        x0, x1 = x1, x2
    
    return x1, max_iter, history


# Example
f = lambda x: x**3 - x - 2

root, iters, hist = secant_method(f, 1, 2)
print(f"Root: {root:.15f}")
print(f"Iterations: {iters}")
```

---

## 🔄 4. Fixed-Point Iteration (Fixpunktiteration)

### Concept

Transform f(x) = 0 into x = g(x), then iterate:

```
xₙ₊₁ = g(xₙ)
```

### Convergence Theorem (Banachscher Fixpunktsatz)

If g: [a,b] → [a,b] and |g'(x)| ≤ L < 1 for all x ∈ [a,b], then:
1. g has exactly one fixed point x* in [a,b]
2. The iteration converges for any x₀ ∈ [a,b]
3. Convergence rate: |xₙ₊₁ - x*| ≤ L|xₙ - x*|

### Python Implementation

```python
def fixed_point(g, x0, tol=1e-10, max_iter=100):
    """
    Fixed-point iteration x = g(x).
    
    Parameters:
        g: Function g(x) where we seek x* = g(x*)
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        root: Fixed point
        iterations: Number of iterations
        history: List of x values
    """
    x = x0
    history = [x]
    
    for i in range(max_iter):
        x_new = g(x)
        history.append(x_new)
        
        if abs(x_new - x) < tol:
            return x_new, i + 1, history
        
        x = x_new
    
    return x, max_iter, history


# Example: x³ - x - 2 = 0 → x = (x + 2)^(1/3)
g = lambda x: (x + 2) ** (1/3)

root, iters, hist = fixed_point(g, 1.5)
print(f"Fixed point: {root:.15f}")
print(f"Iterations: {iters}")
```

### Choosing g(x)

For f(x) = x³ - x - 2 = 0, possible g(x):

| g(x) | g'(x) at root | Converges? |
|------|---------------|------------|
| (x + 2)^(1/3) | ≈ 0.18 | ✅ Yes |
| x³ - 2 | ≈ 5.5 | ❌ No |
| √(x + 2)/x^(1/2) | varies | Maybe |

**Rule**: Choose g(x) such that |g'(x*)| < 1

---

## ⚡ 5. Regula Falsi (False Position)

### Concept

Like bisection, but uses linear interpolation instead of midpoint:

```
c = b - f(b) · (b - a) / (f(b) - f(a))
```

### Python Implementation

```python
def regula_falsi(f, a, b, tol=1e-10, max_iter=100):
    """
    Regula Falsi (False Position) method.
    
    Parameters:
        f: Function f(x)
        a, b: Interval bounds where f(a)*f(b) < 0
        tol: Tolerance
        max_iter: Maximum iterations
    
    Returns:
        root, iterations, history
    """
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    history = []
    
    for i in range(max_iter):
        fa, fb = f(a), f(b)
        
        # Linear interpolation
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        history.append((a, b, c, fc))
        
        if abs(fc) < tol:
            return c, i + 1, history
        
        if fa * fc < 0:
            b = c
        else:
            a = c
    
    return c, max_iter, history
```

### Illinois Algorithm (Improved Regula Falsi)

Standard Regula Falsi can be slow when one endpoint stays fixed. Illinois method fixes this:

```python
def illinois_method(f, a, b, tol=1e-10, max_iter=100):
    """Illinois algorithm - improved Regula Falsi."""
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    fa, fb = f(a), f(b)
    side = 0  # Track which side was updated
    
    for i in range(max_iter):
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        
        if abs(fc) < tol:
            return c, i + 1
        
        if fa * fc < 0:
            b, fb = c, fc
            if side == -1:  # Same side twice
                fa /= 2      # Halve the function value
            side = -1
        else:
            a, fa = c, fc
            if side == 1:
                fb /= 2
            side = 1
    
    return c, max_iter
```

---

## 📈 6. Convergence Comparison

### Visual Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_methods(f, df, a, b, x0, true_root):
    """Compare convergence of all methods."""
    
    # Run all methods
    _, _, hist_bisect = bisection(f, a, b)
    _, _, hist_newton = newton_raphson(f, df, x0)
    _, _, hist_secant = secant_method(f, a, b)
    
    # Extract errors
    err_bisect = [abs(h[2] - true_root) for h in hist_bisect]
    err_newton = [abs(h[0] - true_root) for h in hist_newton]
    err_secant = [abs(h - true_root) for h in hist_secant]
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.semilogy(range(1, len(err_bisect)+1), err_bisect, 'b-o', 
                 label=f'Bisection (n={len(err_bisect)})', linewidth=2)
    plt.semilogy(range(1, len(err_newton)+1), err_newton, 'r-s', 
                 label=f'Newton (n={len(err_newton)})', linewidth=2)
    plt.semilogy(range(1, len(err_secant)+1), err_secant, 'g-^', 
                 label=f'Secant (n={len(err_secant)})', linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Error |xₙ - x*|', fontsize=12)
    plt.title('Convergence Comparison of Root-Finding Methods', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example
f = lambda x: x**3 - x - 2
df = lambda x: 3*x**2 - 1
true_root = 1.5213797068045676  # Computed to high precision

compare_methods(f, df, 1, 2, 1.5, true_root)
```

### Summary Table

| Method | Order | Derivative? | Bracketing? | Robustness |
|--------|-------|-------------|-------------|------------|
| Bisection | 1 | No | Yes | ⭐⭐⭐⭐⭐ |
| Newton | 2 | Yes | No | ⭐⭐⭐ |
| Secant | 1.618 | No | No | ⭐⭐⭐⭐ |
| Regula Falsi | 1-1.618 | No | Yes | ⭐⭐⭐⭐ |
| Fixed-Point | 1 | No | No | ⭐⭐ |

---

## ⚠️ 7. Common Pitfalls

### Newton's Method Failures

```python
# 1. Derivative zero at root (multiple root)
f = lambda x: x**2  # Double root at x=0
# Newton converges linearly, not quadratically

# 2. Cycling
f = lambda x: x**3 - 2*x + 2
# From x0=0, Newton cycles between 0 and 1

# 3. Divergence
f = lambda x: np.arctan(x)
# From x0=1.5, Newton diverges

# 4. Wrong root
f = lambda x: x**3 - x
# Has roots at -1, 0, 1 - may converge to unexpected one
```

### Safe Newton Implementation

```python
def newton_safe(f, df, x0, a, b, tol=1e-10, max_iter=50):
    """Newton's method with bisection fallback."""
    x = x0
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(fx) < tol:
            return x, i + 1
        
        # Compute Newton step
        if abs(dfx) > 1e-15:
            x_newton = x - fx / dfx
        else:
            x_newton = None
        
        # Check if Newton step stays in bounds
        if x_newton is not None and a < x_newton < b:
            x_new = x_newton
        else:
            # Fall back to bisection step
            x_new = (a + b) / 2
        
        # Update bracket
        if f(a) * f(x_new) < 0:
            b = x_new
        else:
            a = x_new
        
        x = x_new
    
    return x, max_iter
```

---

## 📋 8. Exam Checklist (Klausur)

### Formulas to Know

- [ ] Bisection: c = (a+b)/2, iterations ≥ log₂((b-a)/ε)
- [ ] Newton: xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
- [ ] Secant: xₙ₊₁ = xₙ - f(xₙ)(xₙ - xₙ₋₁)/(f(xₙ) - f(xₙ₋₁))
- [ ] Fixed-point convergence: |g'(x*)| < 1
- [ ] Newton error: eₙ₊₁ ≈ (f''(x*)/2f'(x*))·eₙ²

### Convergence Orders

- [ ] Bisection: p = 1 (linear)
- [ ] Newton: p = 2 (quadratic) for simple roots
- [ ] Secant: p ≈ 1.618 (superlinear)
- [ ] Fixed-point: p = 1 (linear)

### When to Use What

- [ ] Bisection: Always works if bracketed, slow but safe
- [ ] Newton: Fast near root, needs good initial guess
- [ ] Secant: No derivative needed, nearly as fast as Newton
- [ ] Fixed-point: When natural transformation exists

---

## 🔗 Related Documents

- [02-interpolation.md](./02-interpolation.md) - Polynomial interpolation
- [03-integration.md](./03-integration.md) - Numerical integration
- [04-ode-solvers.md](./04-ode-solvers.md) - ODE solving methods

---

## 📚 References

- Stoer & Bulirsch, "Numerische Mathematik 1"
- Quarteroni, Sacco, Saleri, "Numerische Mathematik 1"
- Burden & Faires, "Numerical Analysis"

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: January 24, 2026*
