# Hilbert Spaces (Hilbert-Räume)

## 📐 Introduction

Hilbert spaces combine the completeness of Banach spaces with the geometric structure of inner products. They are the natural setting for quantum mechanics, Fourier analysis, and many areas of applied mathematics. Essential for your Funktionale Analysis exam!

---

## 🎯 1. Inner Product Spaces (Prä-Hilbert-Räume)

### Definition

An **inner product** on a vector space H over 𝕂 (ℝ or ℂ) is a map ⟨·,·⟩: H × H → 𝕂 satisfying:

```
1. ⟨x, x⟩ ≥ 0 and ⟨x, x⟩ = 0 ⟺ x = 0    (Positive definiteness)
2. ⟨x, y⟩ = ⟨y, x⟩̄                        (Conjugate symmetry)
3. ⟨αx + βy, z⟩ = α⟨x, z⟩ + β⟨y, z⟩      (Linearity in first argument)
```

Note: In physics convention, linearity is in the second argument.

### Induced Norm

```
‖x‖ = √⟨x, x⟩
```

This makes every inner product space a normed space.

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class InnerProductSpace:
    """Abstract inner product space."""
    
    def inner(self, x, y):
        """Compute ⟨x, y⟩."""
        raise NotImplementedError
    
    def norm(self, x):
        """‖x‖ = √⟨x,x⟩."""
        return np.sqrt(np.real(self.inner(x, x)))
    
    def distance(self, x, y):
        """d(x, y) = ‖x - y‖."""
        return self.norm(x - y)
    
    def angle(self, x, y):
        """Angle between x and y."""
        cos_theta = np.real(self.inner(x, y)) / (self.norm(x) * self.norm(y))
        cos_theta = np.clip(cos_theta, -1, 1)
        return np.arccos(cos_theta)


class EuclideanSpace(InnerProductSpace):
    """ℝⁿ with standard inner product."""
    
    def inner(self, x, y):
        return np.dot(x, y)


class L2Space(InnerProductSpace):
    """L²[a,b] with ⟨f,g⟩ = ∫f(x)g(x)dx."""
    
    def __init__(self, a=0, b=1):
        self.a, self.b = a, b
    
    def inner(self, f, g):
        """⟨f, g⟩ = ∫ₐᵇ f(x)·g(x) dx."""
        result, _ = integrate.quad(lambda x: f(x) * g(x), self.a, self.b)
        return result


# Example: ℝ³
R3 = EuclideanSpace()
x = np.array([1, 2, 3])
y = np.array([4, -1, 2])

print("=== Euclidean Space ℝ³ ===")
print(f"x = {x}, y = {y}")
print(f"⟨x, y⟩ = {R3.inner(x, y)}")
print(f"‖x‖ = {R3.norm(x):.4f}")
print(f"‖y‖ = {R3.norm(y):.4f}")
print(f"Angle = {np.degrees(R3.angle(x, y)):.2f}°")

# Example: L²[0,1]
L2 = L2Space(0, 1)
f = lambda x: x
g = lambda x: x**2

print("\n=== L²[0,1] ===")
print("f(x) = x, g(x) = x²")
print(f"⟨f, g⟩ = ∫₀¹ x·x² dx = ∫₀¹ x³ dx = 1/4 = {L2.inner(f, g):.4f}")
print(f"‖f‖ = √(∫₀¹ x² dx) = √(1/3) = {L2.norm(f):.4f}")
```

---

## 📊 2. Cauchy-Schwarz and Parallelogram

### Cauchy-Schwarz Inequality (Cauchy-Schwarz-Ungleichung)

```
|⟨x, y⟩| ≤ ‖x‖ · ‖y‖

Equality iff x and y are linearly dependent.
```

### Parallelogram Law (Parallelogrammgleichung)

```
‖x + y‖² + ‖x - y‖² = 2(‖x‖² + ‖y‖²)
```

**Key theorem:** A norm comes from an inner product ⟺ it satisfies the parallelogram law.

### Polarization Identity

Recover inner product from norm:

```
Real case:
⟨x, y⟩ = ¼(‖x + y‖² - ‖x - y‖²)

Complex case:
⟨x, y⟩ = ¼(‖x + y‖² - ‖x - y‖² + i‖x + iy‖² - i‖x - iy‖²)
```

```python
def verify_inner_product_properties():
    """Verify Cauchy-Schwarz and Parallelogram law."""
    
    print("=== Cauchy-Schwarz Inequality ===\n")
    
    x = np.array([1, 2, 3])
    y = np.array([4, -1, 2])
    
    lhs = np.abs(np.dot(x, y))
    rhs = np.linalg.norm(x) * np.linalg.norm(y)
    
    print(f"|⟨x, y⟩| = {lhs}")
    print(f"‖x‖·‖y‖ = {rhs:.4f}")
    print(f"|⟨x, y⟩| ≤ ‖x‖·‖y‖: {lhs:.4f} ≤ {rhs:.4f} ✓")
    
    # Equality case
    z = 2 * x
    lhs_eq = np.abs(np.dot(x, z))
    rhs_eq = np.linalg.norm(x) * np.linalg.norm(z)
    print(f"\nFor z = 2x (linearly dependent):")
    print(f"|⟨x, z⟩| = ‖x‖·‖z‖: {lhs_eq:.4f} = {rhs_eq:.4f} ✓")
    
    print("\n" + "="*50)
    print("\n=== Parallelogram Law ===\n")
    
    lhs_para = np.linalg.norm(x + y)**2 + np.linalg.norm(x - y)**2
    rhs_para = 2 * (np.linalg.norm(x)**2 + np.linalg.norm(y)**2)
    
    print(f"‖x + y‖² + ‖x - y‖² = {lhs_para:.4f}")
    print(f"2(‖x‖² + ‖y‖²) = {rhs_para:.4f}")
    print(f"Equal: {np.isclose(lhs_para, rhs_para)} ✓")
    
    # Visualize in 2D
    visualize_parallelogram_law()


def visualize_parallelogram_law():
    """Visualize parallelogram law."""
    
    x = np.array([2, 1])
    y = np.array([1, 2])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Parallelogram
    vertices = np.array([[0, 0], x, x + y, y, [0, 0]])
    ax.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=2)
    ax.fill(vertices[:-1, 0], vertices[:-1, 1], alpha=0.2, color='blue')
    
    # Diagonals
    ax.plot([0, (x + y)[0]], [0, (x + y)[1]], 'r-', linewidth=2, label='x + y')
    ax.plot([x[0], y[0]], [x[1], y[1]], 'g-', linewidth=2, label='x - y (shifted)')
    
    # Vectors
    ax.annotate('', xy=x, xytext=[0, 0], arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=y, xytext=[0, 0], arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.text(x[0]/2 - 0.2, x[1]/2 + 0.2, 'x', fontsize=14)
    ax.text(y[0]/2 + 0.2, y[1]/2, 'y', fontsize=14)
    
    # Labels
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Parallelogram Law: ‖x+y‖² + ‖x-y‖² = 2(‖x‖² + ‖y‖²)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 4)
    
    plt.tight_layout()
    plt.savefig('parallelogram_law.png', dpi=150)
    plt.show()


verify_inner_product_properties()
```

---

## 🌟 3. Hilbert Space Definition

### Definition

A **Hilbert space** is a complete inner product space.

```
(H, ⟨·,·⟩) is Hilbert ⟺ every Cauchy sequence converges in H
```

### Examples

| Space | Inner Product | Hilbert? |
|-------|---------------|----------|
| ℝⁿ, ℂⁿ | ⟨x,y⟩ = Σᵢ xᵢȳᵢ | ✅ Yes |
| l² | ⟨x,y⟩ = Σₙ xₙȳₙ | ✅ Yes |
| L²[a,b] | ⟨f,g⟩ = ∫ f·ḡ dx | ✅ Yes |
| C[a,b] with L² | ⟨f,g⟩ = ∫ fg dx | ❌ No (not complete) |

### The Space l²

```
l² = {(xₙ)ₙ∈ℕ : Σₙ |xₙ|² < ∞}

⟨x, y⟩ = Σₙ₌₁^∞ xₙȳₙ

This is THE prototypical infinite-dimensional Hilbert space.
```

```python
def l2_space_example():
    """Working with l² sequences."""
    
    print("=== The Hilbert Space l² ===\n")
    
    # Example sequences
    # x = (1, 1/2, 1/3, 1/4, ...) - harmonic sequence
    # This is in l² since Σ 1/n² = π²/6 < ∞
    
    N = 10000  # Truncation
    x = np.array([1/n for n in range(1, N+1)])
    y = np.array([1/n**2 for n in range(1, N+1)])
    
    norm_x = np.sqrt(np.sum(x**2))
    norm_y = np.sqrt(np.sum(y**2))
    inner_xy = np.sum(x * y)
    
    print("x = (1, 1/2, 1/3, ...)")
    print("y = (1, 1/4, 1/9, ...)")
    print(f"\n‖x‖² = Σ 1/n² = π²/6 ≈ {np.pi**2/6:.6f}")
    print(f"Computed ‖x‖² ≈ {np.sum(x**2):.6f}")
    print(f"\n⟨x, y⟩ = Σ 1/n³ = ζ(3) ≈ 1.202")
    print(f"Computed ⟨x, y⟩ ≈ {inner_xy:.6f}")
    
    # Standard basis
    print("\n" + "="*50)
    print("\nStandard orthonormal basis of l²:")
    print("eₙ = (0, ..., 0, 1, 0, ...) with 1 in position n")
    print("⟨eₙ, eₘ⟩ = δₙₘ (Kronecker delta)")


l2_space_example()
```

---

## 📐 4. Orthogonality (Orthogonalität)

### Definition

```
x ⊥ y  ⟺  ⟨x, y⟩ = 0   (x orthogonal to y)
```

### Pythagorean Theorem

```
x ⊥ y  ⟹  ‖x + y‖² = ‖x‖² + ‖y‖²
```

### Orthogonal Complement

```
M⊥ = {x ∈ H : ⟨x, m⟩ = 0 ∀m ∈ M}

Properties:
- M⊥ is always a closed subspace
- (M⊥)⊥ = span(M)̄ (closure of span)
- M ∩ M⊥ = {0}
```

```python
def orthogonality_demo():
    """Demonstrate orthogonality concepts."""
    
    print("=== Orthogonality ===\n")
    
    # Orthogonal vectors in ℝ³
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    
    print("Standard basis in ℝ³:")
    print(f"⟨e₁, e₂⟩ = {np.dot(x, y)}")
    print(f"⟨e₁, e₃⟩ = {np.dot(x, z)}")
    print(f"⟨e₂, e₃⟩ = {np.dot(y, z)}")
    print("All orthogonal! ✓")
    
    # Pythagorean theorem
    print("\n" + "="*50)
    print("\n=== Pythagorean Theorem ===\n")
    
    a = np.array([3, 0])
    b = np.array([0, 4])
    
    print(f"a = {a}, b = {b}")
    print(f"⟨a, b⟩ = {np.dot(a, b)} (orthogonal)")
    print(f"\n‖a‖² = {np.linalg.norm(a)**2}")
    print(f"‖b‖² = {np.linalg.norm(b)**2}")
    print(f"‖a‖² + ‖b‖² = {np.linalg.norm(a)**2 + np.linalg.norm(b)**2}")
    print(f"‖a + b‖² = {np.linalg.norm(a + b)**2}")
    print("Equal! ✓ (3-4-5 triangle)")
    
    # Orthogonal complement
    print("\n" + "="*50)
    print("\n=== Orthogonal Complement ===\n")
    
    print("M = span{(1, 1, 0)} in ℝ³")
    print("M⊥ = {x : x₁ + x₂ = 0} = span{(1, -1, 0), (0, 0, 1)}")
    
    m = np.array([1, 1, 0])
    v1 = np.array([1, -1, 0])
    v2 = np.array([0, 0, 1])
    
    print(f"\n⟨m, v₁⟩ = {np.dot(m, v1)}")
    print(f"⟨m, v₂⟩ = {np.dot(m, v2)}")
    print("Both in M⊥ ✓")


orthogonality_demo()
```

---

## 🎯 5. Orthogonal Projection (Orthogonalprojektion)

### Projection Theorem

```
Let M be a closed subspace of Hilbert space H.
For every x ∈ H, there exists a unique decomposition:

x = m + m⊥  where m ∈ M, m⊥ ∈ M⊥

The map P: H → M, Px = m is the orthogonal projection onto M.
```

### Best Approximation

```
Px = argmin{‖x - m‖ : m ∈ M}

"P projects x onto the closest point in M"
```

### Properties of Projections

```
1. P² = P (idempotent)
2. P* = P (self-adjoint)
3. ‖P‖ = 1 (if P ≠ 0)
4. ker(P) = M⊥
5. ran(P) = M
```

```python
def projection_demo():
    """Demonstrate orthogonal projection."""
    
    print("=== Orthogonal Projection ===\n")
    
    # Project onto a line in ℝ²
    # M = span{u}, project x onto M
    
    u = np.array([1, 1]) / np.sqrt(2)  # Unit vector
    x = np.array([3, 1])
    
    # Projection formula: P_M(x) = ⟨x, u⟩ u
    proj = np.dot(x, u) * u
    perp = x - proj
    
    print(f"u = {u} (unit vector spanning M)")
    print(f"x = {x}")
    print(f"\nProjection onto M:")
    print(f"P_M(x) = ⟨x, u⟩·u = {np.dot(x, u):.4f} · u = {proj}")
    print(f"\nOrthogonal component:")
    print(f"x - P_M(x) = {perp}")
    print(f"\nVerify orthogonality: ⟨proj, perp⟩ = {np.dot(proj, perp):.10f} ≈ 0 ✓")
    
    # Visualize
    visualize_projection(u, x, proj, perp)
    
    # Projection onto subspace in ℝ³
    print("\n" + "="*50)
    print("\n=== Projection onto Plane in ℝ³ ===\n")
    
    # M = xy-plane = span{e₁, e₂}
    x3d = np.array([2, 3, 5])
    proj_3d = np.array([2, 3, 0])  # Just zero out z-component
    
    print(f"x = {x3d}")
    print(f"M = xy-plane")
    print(f"P_M(x) = {proj_3d}")
    print(f"x - P_M(x) = {x3d - proj_3d} ∈ M⊥")


def visualize_projection(u, x, proj, perp):
    """Visualize projection in 2D."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Line M
    t = np.linspace(-1, 4, 100)
    ax.plot(t * u[0], t * u[1], 'b-', linewidth=1, label='M = span{u}')
    
    # Vectors
    ax.annotate('', xy=x, xytext=[0, 0], 
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=proj, xytext=[0, 0],
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=x, xytext=proj,
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    # Right angle marker
    size = 0.2
    ax.plot([proj[0], proj[0] + size*perp[0]/np.linalg.norm(perp)],
            [proj[1], proj[1] + size*perp[1]/np.linalg.norm(perp)], 'k-')
    
    # Labels
    ax.text(x[0] + 0.1, x[1] + 0.1, 'x', fontsize=14, color='red')
    ax.text(proj[0] - 0.3, proj[1] + 0.1, 'P(x)', fontsize=14, color='green')
    ax.text((x[0] + proj[0])/2 + 0.1, (x[1] + proj[1])/2, 'x - P(x)', 
            fontsize=12, color='orange')
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Orthogonal Projection onto Line')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 3)
    
    plt.tight_layout()
    plt.savefig('projection.png', dpi=150)
    plt.show()


projection_demo()
```

---

## 📚 6. Orthonormal Systems (Orthonormalsysteme)

### Definitions

```
Orthogonal system: ⟨eᵢ, eⱼ⟩ = 0 for i ≠ j
Orthonormal system (ONS): ⟨eᵢ, eⱼ⟩ = δᵢⱼ
Orthonormal basis (ONB): complete ONS (span is dense in H)
```

### Fourier Coefficients

```
For ONS {eₙ}, the Fourier coefficients of x are:
cₙ = ⟨x, eₙ⟩
```

### Bessel's Inequality

```
Σₙ |⟨x, eₙ⟩|² ≤ ‖x‖²
```

### Parseval's Identity (for ONB)

```
Σₙ |⟨x, eₙ⟩|² = ‖x‖²  (equality for ONB!)
```

```python
def orthonormal_systems_demo():
    """Demonstrate ONS and Fourier coefficients."""
    
    print("=== Orthonormal Systems ===\n")
    
    # Standard basis in ℝ³
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    
    x = np.array([3, 4, 5])
    
    # Fourier coefficients
    c1 = np.dot(x, e1)
    c2 = np.dot(x, e2)
    c3 = np.dot(x, e3)
    
    print(f"x = {x}")
    print(f"\nFourier coefficients:")
    print(f"c₁ = ⟨x, e₁⟩ = {c1}")
    print(f"c₂ = ⟨x, e₂⟩ = {c2}")
    print(f"c₃ = ⟨x, e₃⟩ = {c3}")
    
    print(f"\nx = c₁e₁ + c₂e₂ + c₃e₃ = {c1*e1 + c2*e2 + c3*e3}")
    
    # Parseval's identity
    print("\n" + "="*50)
    print("\n=== Parseval's Identity ===\n")
    
    lhs = c1**2 + c2**2 + c3**2
    rhs = np.linalg.norm(x)**2
    
    print(f"Σ |cₙ|² = {c1}² + {c2}² + {c3}² = {lhs}")
    print(f"‖x‖² = {rhs}")
    print(f"Equal: {lhs} = {rhs} ✓")


def fourier_series_example():
    """Fourier series in L²[-π, π]."""
    
    print("\n=== Fourier Series in L²[-π, π] ===\n")
    
    # ONB: {1/√(2π), cos(nx)/√π, sin(nx)/√π}
    
    # Example: f(x) = x on [-π, π]
    # Fourier series: f(x) = Σ bₙ sin(nx) where bₙ = 2(-1)^(n+1)/n
    
    x = np.linspace(-np.pi, np.pi, 1000)
    f = x  # Original function
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, f, 'k-', linewidth=2, label='f(x) = x')
    
    # Partial sums
    for N in [1, 3, 5, 10]:
        fourier_sum = np.zeros_like(x)
        for n in range(1, N + 1):
            bn = 2 * (-1)**(n+1) / n
            fourier_sum += bn * np.sin(n * x)
        
        plt.plot(x, fourier_sum, '--', linewidth=1.5, label=f'N = {N}')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Fourier Series Approximation of f(x) = x')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-np.pi, np.pi)
    plt.savefig('fourier_series.png', dpi=150)
    plt.show()
    
    # Parseval for this function
    print("Parseval's identity for f(x) = x:")
    print("‖f‖² = ∫_{-π}^π x² dx = 2π³/3")
    print(f"     = {2*np.pi**3/3:.4f}")
    
    # Sum of Fourier coefficients squared
    N_terms = 1000
    fourier_sum_sq = sum((2*(-1)**(n+1)/n)**2 for n in range(1, N_terms+1))
    # Need to multiply by π (normalization)
    print(f"\nΣ |bₙ|² · π = {fourier_sum_sq * np.pi:.4f}")
    print("(Converges to 2π³/3 as N → ∞)")


orthonormal_systems_demo()
fourier_series_example()
```

---

## 🔄 7. Gram-Schmidt Process

### Algorithm

Given linearly independent {v₁, v₂, ...}, produce ONS {e₁, e₂, ...}:

```
u₁ = v₁
e₁ = u₁ / ‖u₁‖

For k ≥ 2:
uₖ = vₖ - Σⱼ₌₁^(k-1) ⟨vₖ, eⱼ⟩ eⱼ
eₖ = uₖ / ‖uₖ‖
```

```python
def gram_schmidt(vectors):
    """
    Gram-Schmidt orthonormalization.
    
    Parameters:
        vectors: List of linearly independent vectors
    
    Returns:
        List of orthonormal vectors
    """
    n = len(vectors)
    orthonormal = []
    
    for k in range(n):
        # Start with vₖ
        u = vectors[k].astype(float).copy()
        
        # Subtract projections onto previous eⱼ
        for j in range(k):
            u -= np.dot(vectors[k], orthonormal[j]) * orthonormal[j]
        
        # Normalize
        e = u / np.linalg.norm(u)
        orthonormal.append(e)
    
    return orthonormal


def gram_schmidt_demo():
    """Demonstrate Gram-Schmidt process."""
    
    print("=== Gram-Schmidt Process ===\n")
    
    # Input vectors
    v1 = np.array([1, 1, 0])
    v2 = np.array([1, 0, 1])
    v3 = np.array([0, 1, 1])
    
    vectors = [v1, v2, v3]
    
    print("Input vectors:")
    for i, v in enumerate(vectors, 1):
        print(f"  v{i} = {v}")
    
    # Apply Gram-Schmidt
    orthonormal = gram_schmidt(vectors)
    
    print("\nOrthonormal vectors:")
    for i, e in enumerate(orthonormal, 1):
        print(f"  e{i} = [{e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}]")
    
    # Verify orthonormality
    print("\nVerification:")
    for i in range(len(orthonormal)):
        for j in range(i, len(orthonormal)):
            inner = np.dot(orthonormal[i], orthonormal[j])
            expected = 1 if i == j else 0
            print(f"  ⟨e{i+1}, e{j+1}⟩ = {inner:.6f} (expected {expected})")


gram_schmidt_demo()
```

---

## 🌟 8. Riesz Representation Theorem

### Theorem

```
Let H be a Hilbert space and f: H → 𝕂 a bounded linear functional.
Then there exists a unique y ∈ H such that:

f(x) = ⟨x, y⟩  ∀x ∈ H

Moreover, ‖f‖ = ‖y‖.
```

### Consequence

```
The dual space H* is isometrically isomorphic to H itself!
H* ≅ H
```

### Python Illustration

```python
def riesz_representation_demo():
    """Illustrate Riesz representation theorem."""
    
    print("=== Riesz Representation Theorem ===\n")
    
    # In ℝⁿ, every linear functional f can be written as f(x) = ⟨x, y⟩
    
    # Example: f(x) = 2x₁ + 3x₂ - x₃ on ℝ³
    # This equals ⟨x, y⟩ where y = (2, 3, -1)
    
    y = np.array([2, 3, -1])
    
    def f(x):
        return 2*x[0] + 3*x[1] - x[2]
    
    def inner_with_y(x):
        return np.dot(x, y)
    
    # Test on random vectors
    print("f(x) = 2x₁ + 3x₂ - x₃")
    print(f"Riesz representative: y = {y}\n")
    
    np.random.seed(42)
    for _ in range(3):
        x = np.random.randn(3)
        print(f"x = [{x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}]")
        print(f"  f(x) = {f(x):.4f}")
        print(f"  ⟨x, y⟩ = {inner_with_y(x):.4f}")
        print()
    
    # Norm equality
    print(f"‖f‖ = sup{{|f(x)| : ‖x‖ = 1}} = ‖y‖ = {np.linalg.norm(y):.4f}")


riesz_representation_demo()
```

---

## 📋 9. Summary Table

| Concept | Definition | Key Property |
|---------|------------|--------------|
| Inner product | ⟨·,·⟩: H × H → 𝕂 | Induces norm ‖x‖ = √⟨x,x⟩ |
| Hilbert space | Complete inner product space | Has projection theorem |
| Orthogonality | ⟨x, y⟩ = 0 | Pythagorean theorem |
| ONB | Complete orthonormal system | Parseval: Σ\|cₙ\|² = ‖x‖² |
| Projection | P²= P, P* = P | Best approximation in M |

---

## 📋 10. Exam Checklist (Klausur)

### Definitions to Know

- [ ] Inner product (3 axioms)
- [ ] Hilbert space = complete inner product space
- [ ] Orthogonal complement M⊥
- [ ] Orthonormal basis (ONB)

### Key Theorems

- [ ] Cauchy-Schwarz: |⟨x,y⟩| ≤ ‖x‖·‖y‖
- [ ] Parallelogram law
- [ ] Projection theorem
- [ ] Parseval's identity
- [ ] Riesz representation theorem

### Common Exam Tasks

- [ ] Verify inner product axioms
- [ ] Apply Cauchy-Schwarz
- [ ] Gram-Schmidt orthonormalization
- [ ] Compute orthogonal projection
- [ ] Find Fourier coefficients

### Standard Examples

- [ ] l² with standard inner product
- [ ] L²[a,b] with integral inner product
- [ ] Fourier series in L²[-π, π]

---

## 🔗 Related Documents

- [01-metric-normed-spaces.md](./01-metric-normed-spaces.md) - Metric and normed spaces
- [02-operators.md](./02-operators.md) - Linear operators
- [04-fundamental-theorems.md](./04-fundamental-theorems.md) - Big theorems

---

## 📚 References

- Werner, "Funktionalanalysis", Kapitel V
- Kreyszig, "Introductory Functional Analysis", Chapters 3-4
- Young, "An Introduction to Hilbert Space"

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: February 4, 2026*
