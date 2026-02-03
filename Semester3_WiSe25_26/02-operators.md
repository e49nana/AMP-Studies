# Linear Operators (Lineare Operatoren)

## 📐 Introduction

Linear operators are the functions between normed spaces that preserve vector space structure. Understanding their properties—especially boundedness and continuity—is central to functional analysis and essential for your exam.

---

## 🎯 1. Basic Definitions

### Linear Operator

```
T: X → Y is linear if:
1. T(x + y) = Tx + Ty         (Additivity)
2. T(αx) = αTx                (Homogeneity)

Equivalently: T(αx + βy) = αTx + βTy
```

### Notation

```
Tx or T(x)    - operator applied to x
ker(T)        - kernel (Kern): {x ∈ X : Tx = 0}
ran(T), R(T)  - range (Bild): {Tx : x ∈ X}
```

### Examples

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Example 1: Matrix as operator on ℝⁿ
def matrix_operator():
    """Matrix operator T: ℝ³ → ℝ²."""
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    
    x = np.array([1, 0, 1])
    y = np.array([0, 1, 1])
    alpha, beta = 2, 3
    
    # Verify linearity
    lhs = A @ (alpha * x + beta * y)
    rhs = alpha * (A @ x) + beta * (A @ y)
    
    print("=== Matrix Operator ===")
    print(f"A = \n{A}")
    print(f"\nT(αx + βy) = {lhs}")
    print(f"αTx + βTy = {rhs}")
    print(f"Linear: {np.allclose(lhs, rhs)}")


# Example 2: Differentiation operator
def differentiation_operator():
    """D: C¹[0,1] → C[0,1], Df = f'."""
    print("\n=== Differentiation Operator ===")
    print("D: C¹[0,1] → C[0,1]")
    print("Df = f'")
    print("\nker(D) = {constant functions}")
    print("D is linear but NOT bounded on C¹ with sup-norm!")


# Example 3: Integration operator
def integration_operator():
    """I: C[0,1] → C[0,1], (If)(x) = ∫₀ˣ f(t)dt."""
    print("\n=== Integration Operator ===")
    print("(If)(x) = ∫₀ˣ f(t)dt")
    
    # Verify linearity numerically
    f = lambda t: t**2
    g = lambda t: np.sin(t)
    
    x = 0.5
    alpha, beta = 2, 3
    
    # I(αf + βg)(x)
    lhs, _ = integrate.quad(lambda t: alpha*f(t) + beta*g(t), 0, x)
    
    # αI(f)(x) + βI(g)(x)
    If, _ = integrate.quad(f, 0, x)
    Ig, _ = integrate.quad(g, 0, x)
    rhs = alpha * If + beta * Ig
    
    print(f"\nAt x = {x}:")
    print(f"I(αf + βg)(x) = {lhs:.6f}")
    print(f"αI(f)(x) + βI(g)(x) = {rhs:.6f}")
    print(f"Linear: {np.isclose(lhs, rhs)}")


matrix_operator()
differentiation_operator()
integration_operator()
```

---

## 📊 2. Bounded Operators (Beschränkte Operatoren)

### Definition

```
T: X → Y is bounded if:
∃M > 0: ‖Tx‖_Y ≤ M‖x‖_X  ∀x ∈ X

"T doesn't stretch vectors by more than factor M"
```

### Operator Norm

```
‖T‖ = sup{‖Tx‖ : ‖x‖ ≤ 1}
    = sup{‖Tx‖ : ‖x‖ = 1}
    = sup{‖Tx‖/‖x‖ : x ≠ 0}
    = inf{M : ‖Tx‖ ≤ M‖x‖ ∀x}
```

### Key Property

```
‖Tx‖ ≤ ‖T‖ · ‖x‖  ∀x ∈ X
```

### Python Implementation

```python
def operator_norm_examples():
    """Compute operator norms."""
    
    print("=== Operator Norms ===\n")
    
    # Example 1: Matrix operator
    A = np.array([[3, 1],
                  [0, 2]])
    
    # Operator norm (induced by l² norm) = largest singular value
    U, s, Vt = np.linalg.svd(A)
    op_norm_2 = s[0]
    
    # Verify by maximizing over unit sphere
    n_samples = 10000
    theta = np.linspace(0, 2*np.pi, n_samples)
    unit_vectors = np.array([np.cos(theta), np.sin(theta)]).T
    norms_Ax = np.array([np.linalg.norm(A @ x) for x in unit_vectors])
    op_norm_numerical = np.max(norms_Ax)
    
    print(f"Matrix A = \n{A}\n")
    print(f"‖A‖₂ (SVD) = σ_max = {op_norm_2:.6f}")
    print(f"‖A‖₂ (numerical) = {op_norm_numerical:.6f}")
    
    # Different operator norms
    print(f"\n‖A‖₁ (max column sum) = {np.linalg.norm(A, 1):.6f}")
    print(f"‖A‖∞ (max row sum) = {np.linalg.norm(A, np.inf):.6f}")
    print(f"‖A‖_F (Frobenius) = {np.linalg.norm(A, 'fro'):.6f}")
    
    # Example 2: Integration operator on C[0,1]
    print("\n" + "="*50)
    print("\nIntegration operator I on (C[0,1], ‖·‖∞):")
    print("(If)(x) = ∫₀ˣ f(t)dt")
    print("\n|(If)(x)| = |∫₀ˣ f(t)dt| ≤ ∫₀ˣ |f(t)|dt ≤ x·‖f‖∞ ≤ ‖f‖∞")
    print("So ‖If‖∞ ≤ ‖f‖∞")
    print("Therefore ‖I‖ ≤ 1")
    print("\nAchieved by f(t) = 1: (If)(x) = x, ‖If‖∞ = 1 = ‖f‖∞")
    print("So ‖I‖ = 1")


operator_norm_examples()
```

### Visualization

```python
def visualize_operator_action():
    """Visualize how operator transforms unit ball."""
    
    A = np.array([[2, 1],
                  [0, 1.5]])
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Image under A
    image = A @ unit_circle
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original unit ball
    axes[0].plot(unit_circle[0], unit_circle[1], 'b-', linewidth=2)
    axes[0].fill(unit_circle[0], unit_circle[1], alpha=0.3)
    axes[0].set_title('Unit Ball B₁(0)', fontsize=12)
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].axvline(0, color='k', linewidth=0.5)
    
    # Image T(B₁(0))
    axes[1].plot(image[0], image[1], 'r-', linewidth=2)
    axes[1].fill(image[0], image[1], alpha=0.3, color='red')
    axes[1].set_title(f'T(B₁(0)), ‖T‖ = {np.linalg.norm(A, 2):.2f}', fontsize=12)
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].axvline(0, color='k', linewidth=0.5)
    
    # Show max stretch direction
    U, s, Vt = np.linalg.svd(A)
    max_dir = Vt[0]  # Direction of max stretch
    axes[0].arrow(0, 0, max_dir[0]*0.9, max_dir[1]*0.9, 
                  head_width=0.1, color='green', linewidth=2)
    stretched = A @ max_dir
    axes[1].arrow(0, 0, stretched[0]*0.9, stretched[1]*0.9,
                  head_width=0.1, color='green', linewidth=2)
    
    plt.suptitle('Operator transforms unit ball to ellipse', fontsize=14)
    plt.tight_layout()
    plt.savefig('operator_action.png', dpi=150)
    plt.show()


visualize_operator_action()
```

---

## 🔄 3. Continuity and Boundedness

### Fundamental Theorem

```
For linear operators T: X → Y between normed spaces:

T is continuous ⟺ T is bounded ⟺ T is continuous at 0
```

### Proof Sketch

```
Bounded ⟹ Continuous:
‖Txₙ - Tx‖ = ‖T(xₙ - x)‖ ≤ ‖T‖ · ‖xₙ - x‖ → 0

Continuous at 0 ⟹ Bounded:
If not bounded, ∃xₙ: ‖Txₙ‖ > n‖xₙ‖
Let yₙ = xₙ/(n‖xₙ‖), then ‖yₙ‖ = 1/n → 0
But ‖Tyₙ‖ = ‖Txₙ‖/(n‖xₙ‖) > 1 ↛ 0. Contradiction!
```

### Unbounded Operators

```python
def unbounded_operator_example():
    """Example: Differentiation is unbounded."""
    
    print("=== Unbounded Operator: Differentiation ===\n")
    print("D: (C¹[0,1], ‖·‖∞) → (C[0,1], ‖·‖∞)")
    print("Df = f'\n")
    
    print("Consider fₙ(x) = sin(nx)/n")
    print("‖fₙ‖∞ = 1/n → 0")
    print("\nBut f'ₙ(x) = cos(nx)")
    print("‖f'ₙ‖∞ = 1 ↛ 0")
    print("\n⟹ D is not continuous at 0")
    print("⟹ D is unbounded!")
    
    # Numerical illustration
    x = np.linspace(0, 1, 1000)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for n in [1, 5, 10, 20]:
        f = np.sin(n * np.pi * x) / n
        df = np.pi * np.cos(n * np.pi * x)
        
        axes[0].plot(x, f, label=f'n={n}')
        axes[1].plot(x, df, label=f'n={n}')
    
    axes[0].set_title('fₙ(x) = sin(nπx)/n → 0', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("f'ₙ(x) = πcos(nπx) ↛ 0", fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("f'(x)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Differentiation is NOT continuous (unbounded)', fontsize=14)
    plt.tight_layout()
    plt.savefig('unbounded_differentiation.png', dpi=150)
    plt.show()


unbounded_operator_example()
```

---

## 📏 4. Space of Bounded Operators

### Definition

```
B(X, Y) = L(X, Y) = {T: X → Y : T linear and bounded}

With operator norm ‖T‖, this is a normed space.
```

### Key Theorem

```
If Y is a Banach space, then B(X, Y) is also a Banach space.
```

### Special Case: Dual Space

```
X* = X' = B(X, 𝕂) = {bounded linear functionals on X}

f ∈ X* means f: X → ℝ (or ℂ) is linear and bounded.
```

### Properties of Operator Norm

```
1. ‖T‖ ≥ 0, and ‖T‖ = 0 ⟺ T = 0
2. ‖αT‖ = |α| · ‖T‖
3. ‖S + T‖ ≤ ‖S‖ + ‖T‖
4. ‖ST‖ ≤ ‖S‖ · ‖T‖  (submultiplicative)
5. ‖Tx‖ ≤ ‖T‖ · ‖x‖
```

```python
def operator_space_properties():
    """Demonstrate properties of operator norm."""
    
    print("=== Properties of Operator Norm ===\n")
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[0, 1], [1, 0]])
    
    norm_A = np.linalg.norm(A, 2)
    norm_B = np.linalg.norm(B, 2)
    norm_AB = np.linalg.norm(A @ B, 2)
    norm_AplusB = np.linalg.norm(A + B, 2)
    
    print(f"A = \n{A}\n")
    print(f"B = \n{B}\n")
    
    print(f"‖A‖ = {norm_A:.4f}")
    print(f"‖B‖ = {norm_B:.4f}")
    
    print(f"\nSubmultiplicativity: ‖AB‖ ≤ ‖A‖·‖B‖")
    print(f"‖AB‖ = {norm_AB:.4f}")
    print(f"‖A‖·‖B‖ = {norm_A * norm_B:.4f}")
    print(f"Check: {norm_AB:.4f} ≤ {norm_A * norm_B:.4f} ✓")
    
    print(f"\nTriangle inequality: ‖A+B‖ ≤ ‖A‖ + ‖B‖")
    print(f"‖A+B‖ = {norm_AplusB:.4f}")
    print(f"‖A‖ + ‖B‖ = {norm_A + norm_B:.4f}")
    print(f"Check: {norm_AplusB:.4f} ≤ {norm_A + norm_B:.4f} ✓")


operator_space_properties()
```

---

## 🎯 5. Kernel and Range

### Kernel (Kern)

```
ker(T) = N(T) = {x ∈ X : Tx = 0}

Always a subspace of X.
T injective ⟺ ker(T) = {0}
```

### Range (Bild)

```
ran(T) = R(T) = {Tx : x ∈ X}

Always a subspace of Y.
T surjective ⟺ ran(T) = Y
```

### Closed Range

```
For bounded T:
- ker(T) is always closed
- ran(T) is NOT always closed!
```

### Python Example

```python
def kernel_range_example():
    """Analyze kernel and range of operators."""
    
    print("=== Kernel and Range ===\n")
    
    # Example: Projection operator
    # P: ℝ³ → ℝ³, P(x,y,z) = (x,y,0)
    P = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]])
    
    print("Projection P(x,y,z) = (x,y,0)")
    print(f"P = \n{P}\n")
    
    # Kernel
    print("ker(P) = {(0,0,z) : z ∈ ℝ} = span{(0,0,1)}")
    print(f"Verify: P·(0,0,1) = {P @ np.array([0,0,1])}")
    
    # Range
    print("\nran(P) = {(x,y,0) : x,y ∈ ℝ} = span{(1,0,0), (0,1,0)}")
    
    # Rank-nullity theorem
    rank = np.linalg.matrix_rank(P)
    nullity = 3 - rank
    print(f"\nRank-Nullity: dim(ker) + dim(ran) = dim(X)")
    print(f"             {nullity} + {rank} = 3 ✓")
    
    # Example 2: Non-surjective bounded operator
    print("\n" + "="*50)
    print("\nExample: T: l² → l², T(x₁,x₂,x₃,...) = (0,x₁,x₂,...)")
    print("(Right shift operator)")
    print("\nker(T) = {0} (injective)")
    print("ran(T) = {(0,y₂,y₃,...)} ≠ l² (not surjective)")
    print("ran(T) is closed in this case.")


kernel_range_example()
```

---

## 🔀 6. Inverse Operators

### Invertibility

```
T: X → Y is invertible if:
∃T⁻¹: Y → X such that T⁻¹T = Iₓ and TT⁻¹ = I_Y
```

### Bounded Inverse Theorem (Satz vom beschränkten Inversen)

```
If X, Y are Banach spaces and T ∈ B(X,Y) is bijective,
then T⁻¹ is also bounded (T⁻¹ ∈ B(Y,X)).
```

### Neumann Series

```
If ‖T‖ < 1, then (I - T) is invertible and:

(I - T)⁻¹ = Σₙ₌₀^∞ Tⁿ = I + T + T² + T³ + ...

‖(I - T)⁻¹‖ ≤ 1/(1 - ‖T‖)
```

```python
def neumann_series_example():
    """Demonstrate Neumann series for operator inverse."""
    
    print("=== Neumann Series ===\n")
    print("If ‖T‖ < 1, then (I - T)⁻¹ = Σₙ Tⁿ\n")
    
    # Example matrix with ‖T‖ < 1
    T = np.array([[0.1, 0.2],
                  [0.15, 0.1]])
    
    norm_T = np.linalg.norm(T, 2)
    print(f"T = \n{T}")
    print(f"\n‖T‖ = {norm_T:.4f} < 1 ✓\n")
    
    # Compute (I - T)⁻¹ via Neumann series
    I = np.eye(2)
    
    neumann_sum = np.zeros_like(T)
    T_power = I.copy()
    
    print("Neumann series partial sums:")
    for n in range(10):
        neumann_sum += T_power
        T_power = T_power @ T
        
        if n in [0, 1, 2, 5, 9]:
            error = np.linalg.norm(neumann_sum - np.linalg.inv(I - T))
            print(f"  n={n}: error = {error:.2e}")
    
    # Compare with direct inverse
    direct_inv = np.linalg.inv(I - T)
    
    print(f"\nDirect (I-T)⁻¹ = \n{direct_inv}")
    print(f"\nNeumann series = \n{neumann_sum}")
    print(f"\nDifference: {np.linalg.norm(direct_inv - neumann_sum):.2e}")
    
    # Bound on inverse norm
    print(f"\n‖(I-T)⁻¹‖ ≤ 1/(1-‖T‖) = {1/(1-norm_T):.4f}")
    print(f"Actual ‖(I-T)⁻¹‖ = {np.linalg.norm(direct_inv, 2):.4f}")


neumann_series_example()
```

---

## 📐 7. Compact Operators (Kompakte Operatoren)

### Definition

```
T: X → Y is compact if:
T(B₁(0)) has compact closure in Y

Equivalently:
For every bounded sequence (xₙ), (Txₙ) has a convergent subsequence.
```

### Properties

```
1. Compact operators are bounded
2. Finite rank operators are compact
3. Limits of compact operators are compact
4. T compact, S bounded ⟹ ST and TS compact
```

### Examples

```
Compact:
- Finite-dimensional operators (matrices)
- Integral operators with continuous kernel
- Hilbert-Schmidt operators

NOT Compact:
- Identity on infinite-dimensional space
- Shift operators on l²
```

```python
def compact_operator_example():
    """Example of compact vs non-compact operators."""
    
    print("=== Compact Operators ===\n")
    
    print("Example 1: Finite rank operator (COMPACT)")
    print("-" * 40)
    print("T: l² → l², T(x₁,x₂,x₃,...) = (x₁,x₂,0,0,...)")
    print("ran(T) is 2-dimensional, so T is compact.\n")
    
    print("Example 2: Integral operator (COMPACT)")
    print("-" * 40)
    print("(Kf)(x) = ∫₀¹ k(x,t)f(t)dt")
    print("If k is continuous, K: C[0,1] → C[0,1] is compact.")
    print("(Arzelà-Ascoli theorem)\n")
    
    print("Example 3: Identity on l² (NOT COMPACT)")
    print("-" * 40)
    print("Consider eₙ = (0,...,0,1,0,...) (1 in n-th position)")
    print("‖eₙ‖ = 1 (bounded sequence)")
    print("But ‖eₙ - eₘ‖ = √2 for n ≠ m")
    print("No convergent subsequence! So I is not compact.\n")
    
    print("Example 4: Diagonal operator")
    print("-" * 40)
    print("T: l² → l², T(x₁,x₂,...) = (x₁/1, x₂/2, x₃/3, ...)")
    print("Diagonal entries λₙ = 1/n → 0")
    print("⟹ T is compact (limit of finite rank operators)")


compact_operator_example()
```

---

## 🔢 8. Spectrum (Spektrum)

### Definition for Bounded Operators

```
Resolvent set: ρ(T) = {λ ∈ ℂ : (T - λI) is bijective with bounded inverse}
Spectrum: σ(T) = ℂ \ ρ(T)
```

### Parts of the Spectrum

```
Point spectrum (Punktspektrum):
σₚ(T) = {λ : ker(T - λI) ≠ {0}} = eigenvalues

Continuous spectrum:
σ_c(T) = {λ : ker(T-λI) = {0}, ran(T-λI) dense but ≠ Y}

Residual spectrum:
σᵣ(T) = {λ : ker(T-λI) = {0}, ran(T-λI) not dense}
```

### Spectral Radius

```
r(T) = sup{|λ| : λ ∈ σ(T)}

r(T) = lim_{n→∞} ‖Tⁿ‖^(1/n) ≤ ‖T‖
```

```python
def spectrum_example():
    """Compute spectrum of matrices."""
    
    print("=== Spectrum ===\n")
    
    # Example matrix
    A = np.array([[4, -1, 1],
                  [2, 1, 1],
                  [-2, 1, 1]])
    
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eigenvalues))
    operator_norm = np.linalg.norm(A, 2)
    
    print(f"A = \n{A}\n")
    print(f"Eigenvalues (= σ(A) for matrices): {eigenvalues}")
    print(f"\nSpectral radius r(A) = {spectral_radius:.4f}")
    print(f"Operator norm ‖A‖ = {operator_norm:.4f}")
    print(f"r(A) ≤ ‖A‖: {spectral_radius:.4f} ≤ {operator_norm:.4f} ✓")
    
    # Verify spectral radius formula
    print("\nSpectral radius formula: r(T) = lim ‖Tⁿ‖^(1/n)")
    for n in [1, 2, 5, 10, 20]:
        A_n = np.linalg.matrix_power(A, n)
        estimate = np.linalg.norm(A_n, 2) ** (1/n)
        print(f"  n={n:2d}: ‖A^n‖^(1/n) = {estimate:.4f}")


spectrum_example()
```

---

## 📋 9. Summary Table

| Concept | Definition | Key Property |
|---------|------------|--------------|
| Linear operator | T(αx + βy) = αTx + βTy | Preserves vector structure |
| Bounded operator | ‖Tx‖ ≤ M‖x‖ | Equivalent to continuous |
| Operator norm | sup{‖Tx‖ : ‖x‖ = 1} | ‖Tx‖ ≤ ‖T‖·‖x‖ |
| Compact operator | T(B₁) has compact closure | "Almost finite-dimensional" |
| Spectrum σ(T) | {λ : T-λI not invertible} | Generalizes eigenvalues |

---

## 📋 10. Exam Checklist (Klausur)

### Definitions to Know

- [ ] Linear operator
- [ ] Bounded operator and operator norm
- [ ] Compact operator
- [ ] Spectrum, resolvent, spectral radius

### Key Theorems

- [ ] Bounded ⟺ Continuous for linear operators
- [ ] B(X,Y) is Banach if Y is Banach
- [ ] Neumann series: (I-T)⁻¹ = ΣTⁿ for ‖T‖ < 1
- [ ] Bounded inverse theorem

### Common Exam Tasks

- [ ] Compute operator norm
- [ ] Show operator is (un)bounded
- [ ] Find kernel and range
- [ ] Apply Neumann series
- [ ] Determine if operator is compact

### Standard Examples

- [ ] Differentiation is unbounded
- [ ] Integration operator has norm 1
- [ ] Shift operators
- [ ] Diagonal operators on l²

---

## 🔗 Related Documents

- [01-metric-normed-spaces.md](./01-metric-normed-spaces.md) - Metric and normed spaces
- [03-hilbert-spaces.md](./03-hilbert-spaces.md) - Inner product spaces
- [04-fundamental-theorems.md](./04-fundamental-theorems.md) - Big theorems

---

## 📚 References

- Werner, "Funktionalanalysis", Kapitel III-IV
- Kreyszig, "Introductory Functional Analysis", Chapters 2-3
- Conway, "A Course in Functional Analysis"

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: February 3, 2026*
