# Metric and Normed Spaces (Metrische und normierte Räume)

## 📐 Introduction

Functional analysis extends linear algebra to infinite-dimensional spaces. This document covers the foundational concepts of metric and normed spaces, essential for your Funktionale Analysis exam.

---

## 🎯 1. Metric Spaces (Metrische Räume)

### Definition

A **metric space** is a pair (X, d) where X is a set and d: X × X → ℝ is a **metric** (distance function) satisfying:

```
1. d(x, y) ≥ 0                    (Non-negativity)
2. d(x, y) = 0 ⟺ x = y           (Identity of indiscernibles)
3. d(x, y) = d(y, x)              (Symmetry)
4. d(x, z) ≤ d(x, y) + d(y, z)    (Triangle inequality)
```

### Common Metrics on ℝⁿ

```
Euclidean (l²):    d₂(x, y) = √(Σᵢ |xᵢ - yᵢ|²)
Manhattan (l¹):    d₁(x, y) = Σᵢ |xᵢ - yᵢ|
Maximum (l∞):      d∞(x, y) = maxᵢ |xᵢ - yᵢ|
p-metric (lᵖ):     dₚ(x, y) = (Σᵢ |xᵢ - yᵢ|ᵖ)^(1/p)
```

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def euclidean_metric(x, y):
    """l² metric."""
    return np.sqrt(np.sum((x - y)**2))

def manhattan_metric(x, y):
    """l¹ metric."""
    return np.sum(np.abs(x - y))

def max_metric(x, y):
    """l∞ metric."""
    return np.max(np.abs(x - y))

def p_metric(x, y, p):
    """lᵖ metric."""
    return np.sum(np.abs(x - y)**p)**(1/p)


# Example
x = np.array([1, 2, 3])
y = np.array([4, 0, 1])

print("=== Metrics on ℝ³ ===")
print(f"x = {x}, y = {y}")
print(f"d₁(x,y) = {manhattan_metric(x, y)}")
print(f"d₂(x,y) = {euclidean_metric(x, y):.4f}")
print(f"d∞(x,y) = {max_metric(x, y)}")


def plot_unit_balls():
    """Visualize unit balls for different metrics in ℝ²."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    theta = np.linspace(0, 2*np.pi, 1000)
    
    # l¹ ball (diamond)
    t = np.linspace(0, 2*np.pi, 1000)
    r1 = 1 / (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    x1, y1 = r1 * np.cos(t), r1 * np.sin(t)
    axes[0].fill(x1, y1, alpha=0.3, color='blue')
    axes[0].plot(x1, y1, 'b-', linewidth=2)
    axes[0].set_title('l¹ Ball (Manhattan)', fontsize=12)
    
    # l² ball (circle)
    x2, y2 = np.cos(theta), np.sin(theta)
    axes[1].fill(x2, y2, alpha=0.3, color='green')
    axes[1].plot(x2, y2, 'g-', linewidth=2)
    axes[1].set_title('l² Ball (Euclidean)', fontsize=12)
    
    # l∞ ball (square)
    square_x = [1, 1, -1, -1, 1]
    square_y = [1, -1, -1, 1, 1]
    axes[2].fill(square_x, square_y, alpha=0.3, color='red')
    axes[2].plot(square_x, square_y, 'r-', linewidth=2)
    axes[2].set_title('l∞ Ball (Maximum)', fontsize=12)
    
    for ax in axes:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
    
    plt.suptitle('Unit Balls: {x : d(x, 0) ≤ 1}', fontsize=14)
    plt.tight_layout()
    plt.savefig('unit_balls.png', dpi=150)
    plt.show()


plot_unit_balls()
```

### Discrete Metric

```
d(x, y) = { 0  if x = y
          { 1  if x ≠ y

Every set becomes a metric space with this metric!
```

### Function Spaces

```
C[a,b] = {f: [a,b] → ℝ | f continuous}

Supremum metric:
d∞(f, g) = sup_{x∈[a,b]} |f(x) - g(x)| = ‖f - g‖∞

L² metric:
d₂(f, g) = (∫ₐᵇ |f(x) - g(x)|² dx)^(1/2)
```

---

## 📊 2. Topological Concepts

### Open Ball (Offene Kugel)

```
B(x, r) = Bᵣ(x) = {y ∈ X : d(x, y) < r}
```

### Closed Ball (Abgeschlossene Kugel)

```
B̄(x, r) = {y ∈ X : d(x, y) ≤ r}
```

### Open and Closed Sets

```
A ⊆ X is open if:
∀x ∈ A ∃r > 0: B(x, r) ⊆ A

A is closed if X \ A is open
Equivalently: A contains all its limit points
```

### Interior, Closure, Boundary

```
Interior:  A° = int(A) = largest open set ⊆ A
Closure:   Ā = cl(A) = smallest closed set ⊇ A
Boundary:  ∂A = Ā \ A°
```

### Python Demonstration

```python
def topological_concepts_demo():
    """Visualize open/closed balls and boundary."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Open ball
    theta = np.linspace(0, 2*np.pi, 100)
    x, y = np.cos(theta), np.sin(theta)
    
    axes[0].fill(x, y, alpha=0.3, color='blue')
    axes[0].plot(x, y, 'b--', linewidth=2, label='Boundary (not included)')
    axes[0].scatter([0], [0], color='red', s=100, zorder=5, label='Center')
    axes[0].set_title('Open Ball B(0, 1)', fontsize=12)
    axes[0].legend()
    
    # Closed ball
    axes[1].fill(x, y, alpha=0.3, color='green')
    axes[1].plot(x, y, 'g-', linewidth=2, label='Boundary (included)')
    axes[1].scatter([0], [0], color='red', s=100, zorder=5)
    axes[1].set_title('Closed Ball B̄(0, 1)', fontsize=12)
    axes[1].legend()
    
    # Set with interior, closure, boundary
    # Square [0,1] × [0,1]
    square = plt.Rectangle((0, 0), 1, 1, fill=True, alpha=0.3, color='orange')
    axes[2].add_patch(square)
    axes[2].plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'orange', linewidth=3, 
                 label='Boundary ∂A')
    axes[2].scatter([0.5], [0.5], color='purple', s=100, zorder=5, 
                    label='Interior point')
    axes[2].scatter([1], [0.5], color='red', s=100, zorder=5, 
                    label='Boundary point')
    axes[2].set_title('Set A = [0,1]²', fontsize=12)
    axes[2].legend()
    axes[2].set_xlim(-0.5, 1.5)
    axes[2].set_ylim(-0.5, 1.5)
    
    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('topological_concepts.png', dpi=150)
    plt.show()


topological_concepts_demo()
```

---

## 🔄 3. Convergence and Completeness

### Convergence (Konvergenz)

```
xₙ → x in (X, d) ⟺ d(xₙ, x) → 0 as n → ∞

⟺ ∀ε > 0 ∃N ∈ ℕ: n ≥ N ⟹ d(xₙ, x) < ε
```

### Cauchy Sequence (Cauchy-Folge)

```
(xₙ) is Cauchy ⟺ ∀ε > 0 ∃N ∈ ℕ: n, m ≥ N ⟹ d(xₙ, xₘ) < ε

"Terms get arbitrarily close to each other"
```

### Key Relationship

```
Convergent ⟹ Cauchy (always!)
Cauchy ⟹ Convergent (only in complete spaces!)
```

### Completeness (Vollständigkeit)

```
(X, d) is complete ⟺ every Cauchy sequence converges in X
```

### Examples

```
Complete:
- (ℝⁿ, d₂): Euclidean space
- (C[a,b], d∞): Continuous functions with sup-norm
- (lᵖ, dₚ): Sequence spaces for p ≥ 1

Not Complete:
- (ℚ, |·|): Rationals with standard metric
- (C[a,b], d₂): Continuous functions with L² metric
```

### Python Demonstration

```python
def cauchy_sequence_demo():
    """Demonstrate Cauchy sequences."""
    
    print("=== Cauchy Sequences ===\n")
    
    # Example 1: Convergent sequence in ℝ
    # xₙ = 1/n → 0
    def x_n(n):
        return 1/n
    
    print("Sequence xₙ = 1/n:")
    for n in [10, 100, 1000]:
        # Check Cauchy property: |xₙ - xₘ| for n, m ≥ N
        max_diff = max(abs(x_n(i) - x_n(j)) 
                      for i in range(n, n+10) 
                      for j in range(n, n+10))
        print(f"  N = {n}: max|xₙ - xₘ| = {max_diff:.6f}")
    
    print("\n→ Cauchy sequence, converges to 0 in ℝ (complete)")
    
    # Example 2: Cauchy in ℚ but not convergent in ℚ
    # Sequence converging to √2
    print("\n" + "="*50)
    print("\nSequence approximating √2 (Cauchy in ℚ, not convergent in ℚ):")
    
    def sqrt2_approx(n):
        """Newton's method for √2."""
        x = 1.0
        for _ in range(n):
            x = (x + 2/x) / 2
        return x
    
    for n in range(1, 8):
        val = sqrt2_approx(n)
        print(f"  x_{n} = {val:.10f}, |x - √2| = {abs(val - np.sqrt(2)):.2e}")
    
    print(f"\n√2 = {np.sqrt(2):.10f} ∉ ℚ")
    print("→ Cauchy in ℚ but limit not in ℚ (ℚ is not complete)")


cauchy_sequence_demo()
```

---

## 📏 4. Normed Spaces (Normierte Räume)

### Definition

A **normed space** is a pair (V, ‖·‖) where V is a vector space over 𝕂 (ℝ or ℂ) and ‖·‖: V → ℝ is a **norm** satisfying:

```
1. ‖x‖ ≥ 0 and ‖x‖ = 0 ⟺ x = 0     (Positive definiteness)
2. ‖αx‖ = |α| · ‖x‖                  (Homogeneity)
3. ‖x + y‖ ≤ ‖x‖ + ‖y‖              (Triangle inequality)
```

### Induced Metric

Every norm induces a metric:
```
d(x, y) = ‖x - y‖
```

### Common Norms on ℝⁿ

```
‖x‖₁ = Σᵢ |xᵢ|                    (l¹ norm)
‖x‖₂ = √(Σᵢ |xᵢ|²)               (l² / Euclidean norm)
‖x‖∞ = maxᵢ |xᵢ|                  (l∞ / sup norm)
‖x‖ₚ = (Σᵢ |xᵢ|ᵖ)^(1/p)          (lᵖ norm, p ≥ 1)
```

### Norm Equivalence

```
Two norms ‖·‖ₐ and ‖·‖ᵦ on V are equivalent if:
∃c, C > 0: c‖x‖ₐ ≤ ‖x‖ᵦ ≤ C‖x‖ₐ  ∀x ∈ V
```

**Theorem:** On finite-dimensional spaces, ALL norms are equivalent!

```python
def norm_equivalence_demo():
    """Demonstrate norm equivalence in ℝⁿ."""
    
    print("=== Norm Equivalence in ℝⁿ ===\n")
    
    # Generate random vectors
    np.random.seed(42)
    n_vectors = 1000
    dim = 3
    
    vectors = np.random.randn(n_vectors, dim)
    
    # Compute norms
    norm_1 = np.sum(np.abs(vectors), axis=1)
    norm_2 = np.sqrt(np.sum(vectors**2, axis=1))
    norm_inf = np.max(np.abs(vectors), axis=1)
    
    # Find equivalence constants
    print("Equivalence constants (empirical):")
    print(f"\n‖x‖∞ ≤ ‖x‖₂ ≤ √n · ‖x‖∞")
    print(f"  Max ratio ‖x‖₂/‖x‖∞ = {np.max(norm_2/norm_inf):.4f}")
    print(f"  √{dim} = {np.sqrt(dim):.4f}")
    
    print(f"\n‖x‖₂ ≤ ‖x‖₁ ≤ √n · ‖x‖₂")
    print(f"  Max ratio ‖x‖₁/‖x‖₂ = {np.max(norm_1/norm_2):.4f}")
    print(f"  √{dim} = {np.sqrt(dim):.4f}")
    
    print(f"\n‖x‖∞ ≤ ‖x‖₁ ≤ n · ‖x‖∞")
    print(f"  Max ratio ‖x‖₁/‖x‖∞ = {np.max(norm_1/norm_inf):.4f}")
    print(f"  {dim} = {dim}")
    
    # Theoretical bounds
    print("\n" + "="*50)
    print("\nTheoretical equivalence in ℝⁿ:")
    print("‖x‖∞ ≤ ‖x‖₂ ≤ √n · ‖x‖∞")
    print("‖x‖₂ ≤ ‖x‖₁ ≤ √n · ‖x‖₂")
    print("‖x‖∞ ≤ ‖x‖₁ ≤ n · ‖x‖∞")


norm_equivalence_demo()
```

---

## 🌟 5. Banach Spaces (Banach-Räume)

### Definition

A **Banach space** is a complete normed space.

```
(V, ‖·‖) is Banach ⟺ every Cauchy sequence in V converges in V
```

### Examples

| Space | Norm | Banach? |
|-------|------|---------|
| (ℝⁿ, ‖·‖ₚ) | Any p ≥ 1 | ✅ Yes |
| (C[a,b], ‖·‖∞) | sup norm | ✅ Yes |
| (lᵖ, ‖·‖ₚ) | lᵖ norm | ✅ Yes |
| (L^p[a,b], ‖·‖ₚ) | Lᵖ norm | ✅ Yes |
| (C[a,b], ‖·‖₂) | L² norm | ❌ No |

### Sequence Spaces lᵖ

```
lᵖ = {(xₙ)ₙ∈ℕ : Σₙ |xₙ|ᵖ < ∞}

‖x‖ₚ = (Σₙ |xₙ|ᵖ)^(1/p)

Special cases:
l¹: absolutely summable sequences
l²: square-summable sequences
l∞: bounded sequences, ‖x‖∞ = supₙ |xₙ|
```

### c₀ Space

```
c₀ = {(xₙ) ∈ l∞ : lim_{n→∞} xₙ = 0}

c₀ is a closed subspace of l∞, hence Banach.
```

```python
def banach_space_examples():
    """Examples of Banach spaces."""
    
    print("=== Banach Space Examples ===\n")
    
    # l² sequence
    def l2_norm(x):
        return np.sqrt(np.sum(x**2))
    
    # Example: sequence xₙ = 1/n²
    N = 1000
    x = np.array([1/n**2 for n in range(1, N+1)])
    
    print(f"Sequence xₙ = 1/n²:")
    print(f"  ‖x‖₂ = {l2_norm(x):.6f}")
    print(f"  Theoretical: π²/6 ≈ {np.pi**2/6:.6f}... wait, that's Σ1/n²")
    print(f"  For ‖x‖₂² = Σ1/n⁴ = π⁴/90 ≈ {np.pi**4/90:.6f}")
    print(f"  So ‖x‖₂ ≈ {np.sqrt(np.pi**4/90):.6f}")
    
    # Check if in various lᵖ spaces
    print(f"\n  x ∈ l¹? Σ|xₙ| = {np.sum(np.abs(x)):.4f} < ∞ ✓")
    print(f"  x ∈ l²? Σ|xₙ|² = {np.sum(x**2):.4f} < ∞ ✓")
    
    # Example NOT in l¹ but in l²
    print("\n" + "="*50)
    y = np.array([1/n for n in range(1, N+1)])
    print(f"\nSequence yₙ = 1/n:")
    print(f"  Partial sum Σ|yₙ| = {np.sum(np.abs(y)):.4f} (diverges as N→∞)")
    print(f"  Partial sum Σ|yₙ|² = {np.sum(y**2):.4f} (converges to π²/6)")
    print("  → y ∈ l² but y ∉ l¹")


banach_space_examples()
```

---

## 📐 6. Important Inequalities

### Hölder's Inequality

```
For p, q > 1 with 1/p + 1/q = 1:

Σᵢ |xᵢyᵢ| ≤ ‖x‖ₚ · ‖y‖_q

Integral form:
∫|fg| ≤ ‖f‖ₚ · ‖g‖_q
```

### Minkowski's Inequality

```
‖x + y‖ₚ ≤ ‖x‖ₚ + ‖y‖ₚ

(This IS the triangle inequality for lᵖ norm)
```

### Cauchy-Schwarz (Special case p = q = 2)

```
|⟨x, y⟩| ≤ ‖x‖₂ · ‖y‖₂

Equality iff x and y are linearly dependent.
```

```python
def inequalities_demo():
    """Demonstrate important inequalities."""
    
    print("=== Important Inequalities ===\n")
    
    # Cauchy-Schwarz
    x = np.array([1, 2, 3, 4])
    y = np.array([2, -1, 0, 3])
    
    inner = np.abs(np.dot(x, y))
    product_norms = np.linalg.norm(x) * np.linalg.norm(y)
    
    print("Cauchy-Schwarz: |⟨x,y⟩| ≤ ‖x‖₂ · ‖y‖₂")
    print(f"  x = {x}, y = {y}")
    print(f"  |⟨x,y⟩| = {inner}")
    print(f"  ‖x‖₂ · ‖y‖₂ = {product_norms:.4f}")
    print(f"  {inner} ≤ {product_norms:.4f} ✓")
    
    # Hölder's inequality
    print("\n" + "="*50)
    print("\nHölder's Inequality: Σ|xᵢyᵢ| ≤ ‖x‖ₚ · ‖y‖_q (1/p + 1/q = 1)")
    
    p, q = 3, 1.5  # 1/3 + 2/3 = 1... wait, 1/3 + 1/1.5 = 1/3 + 2/3 = 1
    # Let's use p=4, q=4/3 (1/4 + 3/4 = 1)
    p, q = 4, 4/3
    
    x = np.array([1, 2, 1, 3])
    y = np.array([2, 1, 1, 1])
    
    lhs = np.sum(np.abs(x * y))
    rhs = np.linalg.norm(x, p) * np.linalg.norm(y, q)
    
    print(f"  p = {p}, q = {q:.4f} (1/p + 1/q = {1/p + 1/q})")
    print(f"  Σ|xᵢyᵢ| = {lhs}")
    print(f"  ‖x‖ₚ · ‖y‖_q = {rhs:.4f}")
    print(f"  {lhs} ≤ {rhs:.4f} ✓")
    
    # Minkowski
    print("\n" + "="*50)
    print("\nMinkowski's Inequality: ‖x+y‖ₚ ≤ ‖x‖ₚ + ‖y‖ₚ")
    
    for p in [1, 2, 3, np.inf]:
        lhs = np.linalg.norm(x + y, p)
        rhs = np.linalg.norm(x, p) + np.linalg.norm(y, p)
        print(f"  p = {p}: {lhs:.4f} ≤ {rhs:.4f} ✓")


inequalities_demo()
```

---

## 🔄 7. Compactness (Kompaktheit)

### Definition

```
K ⊆ X is compact ⟺ every open cover has a finite subcover
                 ⟺ every sequence has a convergent subsequence
                    (in metric spaces)
```

### Heine-Borel Theorem (ℝⁿ)

```
In ℝⁿ: K is compact ⟺ K is closed and bounded
```

### In Infinite Dimensions

```
Closed and bounded ⇏ compact in infinite-dimensional spaces!

Example: Closed unit ball in l² is NOT compact.
```

### Compact Operators

We'll cover these in detail in the next document on operators.

---

## 📋 8. Summary Table

| Concept | Definition | Key Property |
|---------|------------|--------------|
| Metric space | (X, d) with distance d | Triangle inequality |
| Normed space | (V, ‖·‖) with norm | Induces metric d(x,y) = ‖x-y‖ |
| Banach space | Complete normed space | Cauchy ⟹ convergent |
| Open set | Contains ball around each point | Complement of closed |
| Closed set | Contains all limit points | Complement of open |
| Compact | Every sequence has convergent subsequence | Closed + bounded in ℝⁿ |

---

## 📋 9. Exam Checklist (Klausur)

### Definitions to Know

- [ ] Metric: 4 axioms (non-neg, identity, symmetry, triangle)
- [ ] Norm: 3 axioms (pos-def, homogeneity, triangle)
- [ ] Cauchy sequence
- [ ] Complete space / Banach space
- [ ] Open/closed balls and sets

### Key Theorems

- [ ] All norms equivalent in finite dimensions
- [ ] Normed space complete ⟺ Banach space
- [ ] Heine-Borel in ℝⁿ
- [ ] Hölder and Minkowski inequalities

### Common Exam Tasks

- [ ] Verify something is a metric/norm
- [ ] Show a sequence is Cauchy
- [ ] Prove a space is (not) complete
- [ ] Apply Hölder/Cauchy-Schwarz
- [ ] Determine if set is open/closed/compact

### Standard Examples

- [ ] lᵖ spaces and their norms
- [ ] C[a,b] with sup-norm
- [ ] ℚ is not complete
- [ ] Closed unit ball in l² is not compact

---

## 🔗 Related Documents

- [02-operators.md](./02-operators.md) - Linear operators
- [03-hilbert-spaces.md](./03-hilbert-spaces.md) - Inner product spaces
- [04-fundamental-theorems.md](./04-fundamental-theorems.md) - Big theorems

---

## 📚 References

- Werner, "Funktionalanalysis", Kapitel I-II
- Kreyszig, "Introductory Functional Analysis with Applications"
- Brezis, "Functional Analysis"

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: February 2, 2026*
