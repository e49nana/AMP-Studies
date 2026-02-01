# Expectation and Variance (Erwartungswert und Varianz)

## 📐 Introduction

Expected value and variance are the two most fundamental characteristics of a random variable's distribution. This document covers definitions, properties, and computational techniques essential for your Stochastik exam.

---

## 🎯 1. Expected Value (Erwartungswert)

### Definition

**Discrete:**
```
E[X] = Σₓ x · P(X = x) = Σₓ x · pₓ(x)
```

**Continuous:**
```
E[X] = ∫₋∞^∞ x · fₓ(x) dx
```

### Intuition

- "Long-run average" or "center of mass"
- μ = E[X] is where the distribution balances

### Python Implementation

```python
import numpy as np
from scipy import integrate
from scipy.stats import binom, poisson, norm, expon
import matplotlib.pyplot as plt

class ExpectationCalculator:
    """Calculate expectation for discrete and continuous RVs."""
    
    @staticmethod
    def discrete(values, probabilities):
        """E[X] for discrete RV."""
        return sum(x * p for x, p in zip(values, probabilities))
    
    @staticmethod
    def continuous(pdf, a=-np.inf, b=np.inf):
        """E[X] for continuous RV."""
        integrand = lambda x: x * pdf(x)
        result, _ = integrate.quad(integrand, a, b)
        return result
    
    @staticmethod
    def discrete_function(g, values, probabilities):
        """E[g(X)] for discrete RV."""
        return sum(g(x) * p for x, p in zip(values, probabilities))
    
    @staticmethod
    def continuous_function(g, pdf, a=-np.inf, b=np.inf):
        """E[g(X)] for continuous RV."""
        integrand = lambda x: g(x) * pdf(x)
        result, _ = integrate.quad(integrand, a, b)
        return result


# Example: Fair die
values = [1, 2, 3, 4, 5, 6]
probs = [1/6] * 6

E_die = ExpectationCalculator.discrete(values, probs)
print(f"E[Fair Die] = {E_die:.4f}")
print(f"Formula: (1+2+3+4+5+6)/6 = {sum(values)/6:.4f}")

# Example: Continuous uniform [0, 2]
pdf_uniform = lambda x: 0.5 if 0 <= x <= 2 else 0
E_uniform = ExpectationCalculator.continuous(pdf_uniform, 0, 2)
print(f"\nE[Uniform(0,2)] = {E_uniform:.4f}")
print(f"Formula: (a+b)/2 = {(0+2)/2:.4f}")
```

---

## 📊 2. Properties of Expectation

### Linearity (VERY IMPORTANT!)

```
E[aX + b] = aE[X] + b
E[X + Y] = E[X] + E[Y]  (ALWAYS, even if dependent!)
E[Σᵢ Xᵢ] = Σᵢ E[Xᵢ]
```

### Monotonicity

```
X ≤ Y  ⟹  E[X] ≤ E[Y]
X ≥ 0  ⟹  E[X] ≥ 0
```

### Product (only for independent!)

```
X, Y independent  ⟹  E[XY] = E[X]·E[Y]
```

### Python Demonstration

```python
def demonstrate_linearity():
    """Show linearity of expectation."""
    
    # Two dice
    die1 = np.array([1, 2, 3, 4, 5, 6])
    die2 = np.array([1, 2, 3, 4, 5, 6])
    
    # All outcomes for sum
    sums = []
    for d1 in die1:
        for d2 in die2:
            sums.append(d1 + d2)
    
    E_sum_direct = np.mean(sums)
    E_die1 = np.mean(die1)
    E_die2 = np.mean(die2)
    
    print("=== Linearity of Expectation ===")
    print(f"E[Die1] = {E_die1:.4f}")
    print(f"E[Die2] = {E_die2:.4f}")
    print(f"E[Die1] + E[Die2] = {E_die1 + E_die2:.4f}")
    print(f"E[Die1 + Die2] (direct) = {E_sum_direct:.4f}")
    print("→ Linearity holds!")
    
    # Linear transformation
    a, b = 3, 5
    E_aX_b = np.mean(a * die1 + b)
    print(f"\nE[{a}X + {b}] = {E_aX_b:.4f}")
    print(f"{a}·E[X] + {b} = {a * E_die1 + b:.4f}")


demonstrate_linearity()
```

---

## 🔢 3. Law of the Unconscious Statistician (LOTUS)

### Theorem

To find E[g(X)], you don't need the distribution of g(X):

**Discrete:**
```
E[g(X)] = Σₓ g(x) · P(X = x)
```

**Continuous:**
```
E[g(X)] = ∫₋∞^∞ g(x) · fₓ(x) dx
```

### Examples

```python
def lotus_examples():
    """LOTUS in action."""
    
    # Discrete: E[X²] for fair die
    values = np.array([1, 2, 3, 4, 5, 6])
    probs = np.array([1/6] * 6)
    
    E_X = np.sum(values * probs)
    E_X2 = np.sum(values**2 * probs)
    
    print("=== LOTUS Examples ===")
    print(f"\nFair Die:")
    print(f"E[X] = {E_X:.4f}")
    print(f"E[X²] = {E_X2:.4f}")
    print(f"(E[X])² = {E_X**2:.4f}")
    print(f"Note: E[X²] ≠ (E[X])²!")
    
    # Continuous: E[X²] for Exp(λ)
    lam = 2
    pdf_exp = lambda x: lam * np.exp(-lam * x)
    
    E_X_exp, _ = integrate.quad(lambda x: x * pdf_exp(x), 0, np.inf)
    E_X2_exp, _ = integrate.quad(lambda x: x**2 * pdf_exp(x), 0, np.inf)
    
    print(f"\nExponential(λ={lam}):")
    print(f"E[X] = 1/λ = {1/lam:.4f}, computed = {E_X_exp:.4f}")
    print(f"E[X²] = 2/λ² = {2/lam**2:.4f}, computed = {E_X2_exp:.4f}")


lotus_examples()
```

---

## 📈 4. Variance (Varianz)

### Definition

```
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²

Where μ = E[X]
```

### Computational Formula (wichtig!)

```
Var(X) = E[X²] - (E[X])²
```

This is usually easier to compute!

### Standard Deviation (Standardabweichung)

```
σ = SD(X) = √Var(X)
```

### Python Implementation

```python
class VarianceCalculator:
    """Calculate variance for RVs."""
    
    @staticmethod
    def discrete(values, probabilities):
        """Var(X) for discrete RV."""
        E_X = sum(x * p for x, p in zip(values, probabilities))
        E_X2 = sum(x**2 * p for x, p in zip(values, probabilities))
        return E_X2 - E_X**2
    
    @staticmethod
    def continuous(pdf, a=-np.inf, b=np.inf):
        """Var(X) for continuous RV."""
        E_X, _ = integrate.quad(lambda x: x * pdf(x), a, b)
        E_X2, _ = integrate.quad(lambda x: x**2 * pdf(x), a, b)
        return E_X2 - E_X**2


# Fair die
values = [1, 2, 3, 4, 5, 6]
probs = [1/6] * 6

var_die = VarianceCalculator.discrete(values, probs)
print(f"\n=== Fair Die Variance ===")
print(f"Var(X) = {var_die:.4f}")
print(f"SD(X) = {np.sqrt(var_die):.4f}")

# Verify with formula: Var = (n²-1)/12 for uniform discrete on 1,...,n
n = 6
var_formula = (n**2 - 1) / 12
print(f"Formula (n²-1)/12 = {var_formula:.4f}")

# Exponential
lam = 2
pdf_exp = lambda x: lam * np.exp(-lam * x) if x >= 0 else 0
var_exp = VarianceCalculator.continuous(pdf_exp, 0, np.inf)
print(f"\nExponential(λ={lam}):")
print(f"Var(X) = 1/λ² = {1/lam**2:.4f}, computed = {var_exp:.4f}")
```

---

## 📊 5. Properties of Variance

### Scaling

```
Var(aX + b) = a²Var(X)

Note: Adding constant doesn't change variance!
```

### Sum (Independent Only!)

```
X, Y independent  ⟹  Var(X + Y) = Var(X) + Var(Y)
```

### General Sum

```
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)
```

### Python Demonstration

```python
def variance_properties():
    """Demonstrate variance properties."""
    
    # Die example
    X = np.array([1, 2, 3, 4, 5, 6])
    p = 1/6
    
    E_X = np.mean(X)
    Var_X = np.mean(X**2) - E_X**2
    
    # Scaling: Var(3X + 5)
    a, b = 3, 5
    Y = a * X + b
    Var_Y = np.mean(Y**2) - np.mean(Y)**2
    
    print("=== Variance Properties ===")
    print(f"Var(X) = {Var_X:.4f}")
    print(f"Var({a}X + {b}) = {Var_Y:.4f}")
    print(f"{a}²·Var(X) = {a**2 * Var_X:.4f}")
    print("→ Var(aX + b) = a²Var(X) ✓")
    
    # Sum of independent dice
    # Var(D1 + D2) = Var(D1) + Var(D2)
    print(f"\nTwo independent dice:")
    print(f"Var(D1) + Var(D2) = {2 * Var_X:.4f}")
    
    # Direct calculation
    sums = []
    for d1 in X:
        for d2 in X:
            sums.append(d1 + d2)
    sums = np.array(sums)
    Var_sum = np.mean(sums**2) - np.mean(sums)**2
    print(f"Var(D1 + D2) direct = {Var_sum:.4f}")


variance_properties()
```

---

## 🔗 6. Covariance (Kovarianz)

### Definition

```
Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)] = E[XY] - E[X]E[Y]
```

### Properties

```
Cov(X, X) = Var(X)
Cov(X, Y) = Cov(Y, X)  (symmetric)
Cov(aX, bY) = ab·Cov(X, Y)
Cov(X + Y, Z) = Cov(X, Z) + Cov(Y, Z)
X, Y independent  ⟹  Cov(X, Y) = 0
```

### Interpretation

```
Cov(X, Y) > 0: X and Y tend to move together
Cov(X, Y) < 0: X and Y tend to move oppositely
Cov(X, Y) = 0: No linear relationship
```

### Python Implementation

```python
def covariance_example():
    """Covariance calculation and interpretation."""
    
    # Example: Joint distribution
    # X = first die, Y = first die + second die
    
    # Simulate
    np.random.seed(42)
    n = 100000
    
    die1 = np.random.randint(1, 7, n)
    die2 = np.random.randint(1, 7, n)
    
    X = die1
    Y = die1 + die2
    
    E_X = np.mean(X)
    E_Y = np.mean(Y)
    E_XY = np.mean(X * Y)
    
    Cov_XY = E_XY - E_X * E_Y
    
    print("=== Covariance Example ===")
    print("X = Die1, Y = Die1 + Die2")
    print(f"E[X] = {E_X:.4f}")
    print(f"E[Y] = {E_Y:.4f}")
    print(f"E[XY] = {E_XY:.4f}")
    print(f"Cov(X, Y) = E[XY] - E[X]E[Y] = {Cov_XY:.4f}")
    
    # Theoretical: Cov(X, X+Z) = Cov(X,X) + Cov(X,Z) = Var(X) + 0
    Var_X = np.var(X, ddof=0)
    print(f"\nTheoretical: Cov(X, X+Z) = Var(X) = {Var_X:.4f}")
    print("(where Z = Die2 is independent of X)")
    
    # Independent case
    Z = die2
    Cov_XZ = np.mean(X * Z) - np.mean(X) * np.mean(Z)
    print(f"\nIndependent dice:")
    print(f"Cov(Die1, Die2) = {Cov_XZ:.4f} ≈ 0")


covariance_example()
```

---

## 📐 7. Correlation (Korrelation)

### Definition

```
ρ(X, Y) = Corr(X, Y) = Cov(X, Y) / (σₓ · σᵧ)

Where σₓ = √Var(X), σᵧ = √Var(Y)
```

### Properties

```
-1 ≤ ρ(X, Y) ≤ 1
ρ(X, Y) = 1  ⟺  Y = aX + b with a > 0
ρ(X, Y) = -1  ⟺  Y = aX + b with a < 0
ρ(X, Y) = 0  ⟺  X, Y uncorrelated
```

### Interpretation

```
|ρ| ≈ 1: Strong linear relationship
|ρ| ≈ 0.5: Moderate linear relationship
|ρ| ≈ 0: Weak/no linear relationship
```

### Independence vs Uncorrelated

```
Independent ⟹ Uncorrelated (ρ = 0)
Uncorrelated ⇏ Independent!
```

### Python Example

```python
def correlation_example():
    """Correlation examples and counterexample."""
    
    np.random.seed(42)
    n = 10000
    
    # Example 1: Positive correlation
    X1 = np.random.normal(0, 1, n)
    Y1 = 2 * X1 + np.random.normal(0, 0.5, n)
    
    # Example 2: Negative correlation
    X2 = np.random.normal(0, 1, n)
    Y2 = -1.5 * X2 + np.random.normal(0, 0.5, n)
    
    # Example 3: Uncorrelated but dependent!
    X3 = np.random.normal(0, 1, n)
    Y3 = X3 ** 2  # Y = X², clearly dependent on X
    
    def calc_correlation(X, Y):
        cov = np.mean(X * Y) - np.mean(X) * np.mean(Y)
        return cov / (np.std(X) * np.std(Y))
    
    print("=== Correlation Examples ===")
    print(f"Positive linear: ρ = {calc_correlation(X1, Y1):.4f}")
    print(f"Negative linear: ρ = {calc_correlation(X2, Y2):.4f}")
    print(f"Quadratic (Y=X²): ρ = {calc_correlation(X3, Y3):.4f}")
    print("\n→ Y = X² is dependent on X but uncorrelated!")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(X1[:500], Y1[:500], alpha=0.5, s=10)
    axes[0].set_title(f'Positive: ρ = {calc_correlation(X1, Y1):.2f}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    axes[1].scatter(X2[:500], Y2[:500], alpha=0.5, s=10)
    axes[1].set_title(f'Negative: ρ = {calc_correlation(X2, Y2):.2f}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    axes[2].scatter(X3[:500], Y3[:500], alpha=0.5, s=10)
    axes[2].set_title(f'Quadratic (dependent!): ρ = {calc_correlation(X3, Y3):.2f}')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y = X²')
    
    plt.tight_layout()
    plt.savefig('correlation_examples.png', dpi=150)
    plt.show()


correlation_example()
```

---

## 📋 8. Summary: E[X] and Var(X) for Common Distributions

### Discrete

| Distribution | E[X] | Var(X) |
|--------------|------|--------|
| Bernoulli(p) | p | p(1-p) |
| Binomial(n,p) | np | np(1-p) |
| Geometric(p) | 1/p | (1-p)/p² |
| Poisson(λ) | λ | λ |
| Uniform{1,...,n} | (n+1)/2 | (n²-1)/12 |

### Continuous

| Distribution | E[X] | Var(X) |
|--------------|------|--------|
| Uniform(a,b) | (a+b)/2 | (b-a)²/12 |
| Exponential(λ) | 1/λ | 1/λ² |
| Normal(μ,σ²) | μ | σ² |
| Gamma(α,β) | α/β | α/β² |

### Derivation Example

```python
def derive_binomial_moments():
    """Derive E[X] and Var(X) for Binomial using linearity."""
    
    print("=== Binomial Moments via Linearity ===")
    print("\nX ~ Bin(n, p) = X₁ + X₂ + ... + Xₙ")
    print("where Xᵢ ~ Bernoulli(p) are independent")
    print("\nE[Xᵢ] = p")
    print("Var(Xᵢ) = p(1-p)")
    print("\nBy linearity:")
    print("E[X] = E[X₁] + ... + E[Xₙ] = np")
    print("\nBy independence:")
    print("Var(X) = Var(X₁) + ... + Var(Xₙ) = np(1-p)")
    
    # Verify numerically
    n, p = 20, 0.3
    X = binom(n, p)
    
    print(f"\nVerification for n={n}, p={p}:")
    print(f"E[X] = np = {n*p}")
    print(f"scipy mean = {X.mean()}")
    print(f"Var(X) = np(1-p) = {n*p*(1-p)}")
    print(f"scipy var = {X.var()}")


derive_binomial_moments()
```

---

## 🧮 9. Moment Generating Functions (MGF)

### Definition

```
Mₓ(t) = E[e^(tX)]

Discrete: Mₓ(t) = Σₓ e^(tx) P(X = x)
Continuous: Mₓ(t) = ∫₋∞^∞ e^(tx) fₓ(x) dx
```

### Key Property

```
E[Xⁿ] = Mₓ⁽ⁿ⁾(0) = dⁿ/dtⁿ Mₓ(t)|ₜ₌₀

E[X] = M'ₓ(0)
E[X²] = M''ₓ(0)
```

### Common MGFs

| Distribution | Mₓ(t) |
|--------------|-------|
| Bernoulli(p) | 1-p+pe^t |
| Binomial(n,p) | (1-p+pe^t)ⁿ |
| Poisson(λ) | e^(λ(e^t-1)) |
| Exponential(λ) | λ/(λ-t), t<λ |
| Normal(μ,σ²) | e^(μt+σ²t²/2) |

```python
def mgf_example():
    """MGF to find moments."""
    
    print("=== MGF Example: Exponential(λ) ===")
    print("\nMₓ(t) = λ/(λ-t) for t < λ")
    print("\nM'(t) = λ/(λ-t)²")
    print("M'(0) = λ/λ² = 1/λ = E[X] ✓")
    print("\nM''(t) = 2λ/(λ-t)³")
    print("M''(0) = 2λ/λ³ = 2/λ² = E[X²]")
    print("\nVar(X) = E[X²] - (E[X])² = 2/λ² - 1/λ² = 1/λ² ✓")


mgf_example()
```

---

## 📊 10. Chebyshev's Inequality (Tschebyscheff-Ungleichung)

### Theorem

For any random variable with finite variance:

```
P(|X - μ| ≥ kσ) ≤ 1/k²

Equivalently:
P(|X - μ| < kσ) ≥ 1 - 1/k²
```

### Applications

```
k = 2: P(|X - μ| < 2σ) ≥ 75%
k = 3: P(|X - μ| < 3σ) ≥ 88.9%
k = 4: P(|X - μ| < 4σ) ≥ 93.75%
```

### Python Demonstration

```python
def chebyshev_demo():
    """Compare Chebyshev bound with actual probabilities."""
    
    print("=== Chebyshev's Inequality ===")
    print("\nP(|X - μ| < kσ) ≥ 1 - 1/k²")
    print("\n" + "="*60)
    print(f"{'k':<6} {'Chebyshev':<15} {'Normal':<15} {'Exponential':<15}")
    print("="*60)
    
    # Normal(0, 1)
    Z = norm(0, 1)
    
    # Exponential(1): μ = 1, σ = 1
    E = expon(scale=1)
    mu_exp, sigma_exp = 1, 1
    
    for k in [1, 1.5, 2, 2.5, 3, 4]:
        chebyshev = max(0, 1 - 1/k**2)
        
        # Normal: P(|Z| < k)
        p_normal = Z.cdf(k) - Z.cdf(-k)
        
        # Exponential: P(|X-1| < k)
        p_exp = E.cdf(mu_exp + k*sigma_exp) - E.cdf(max(0, mu_exp - k*sigma_exp))
        
        print(f"{k:<6.1f} {chebyshev:<15.4f} {p_normal:<15.4f} {p_exp:<15.4f}")
    
    print("\n→ Chebyshev gives weak but universal bounds!")


chebyshev_demo()
```

---

## 📋 11. Exam Checklist (Klausur)

### Formulas to Know

- [ ] E[X] = Σ x·P(X=x) or ∫ x·f(x)dx
- [ ] Var(X) = E[X²] - (E[X])²
- [ ] Var(aX + b) = a²Var(X)
- [ ] Cov(X,Y) = E[XY] - E[X]E[Y]
- [ ] ρ(X,Y) = Cov(X,Y)/(σₓσᵧ)
- [ ] Chebyshev: P(|X-μ| ≥ kσ) ≤ 1/k²

### Key Properties

- [ ] Linearity of expectation (always!)
- [ ] Var(X+Y) = Var(X) + Var(Y) only if independent
- [ ] Independent ⟹ Uncorrelated, but not reverse
- [ ] E[X²] ≥ (E[X])² (always)

### E[X] and Var(X) by Heart

- [ ] Binomial: np, np(1-p)
- [ ] Poisson: λ, λ
- [ ] Exponential: 1/λ, 1/λ²
- [ ] Normal: μ, σ²

### Common Exam Tasks

- [ ] Calculate E[X] from PMF/PDF
- [ ] Use Var = E[X²] - (E[X])² formula
- [ ] Apply linearity for sums
- [ ] Calculate Cov and Corr
- [ ] Apply Chebyshev inequality

---

## 🔗 Related Documents

- [01-probability-foundations.md](./01-probability-foundations.md) - Probability basics
- [02-random-variables.md](./02-random-variables.md) - Distributions
- [04-limit-theorems.md](./04-limit-theorems.md) - LLN and CLT

---

## 📚 References

- Georgii, "Stochastik", Kapitel 5
- Ross, "A First Course in Probability", Chapter 7
- Meintrup & Schäffler, "Stochastik"

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: January 31, 2026*
