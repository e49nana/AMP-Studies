# Random Variables (Zufallsvariablen)

## 📐 Introduction

A random variable maps outcomes to numbers, enabling mathematical analysis of random phenomena. This document covers discrete and continuous random variables with their distributions, essential for your Stochastik exam.

---

## 🎯 1. Definition and Types

### Random Variable (Zufallsvariable)

```
X: Ω → ℝ

A function that assigns a real number to each outcome ω ∈ Ω.
```

### Types

```
Discrete (diskret): X takes countably many values
  Examples: coin flips, dice rolls, number of customers

Continuous (stetig): X takes values in an interval
  Examples: time, temperature, height
```

### Notation

```
{X = x} = {ω ∈ Ω : X(ω) = x}     Event "X equals x"
{X ≤ x} = {ω ∈ Ω : X(ω) ≤ x}    Event "X at most x"
{a < X ≤ b} = {ω ∈ Ω : a < X(ω) ≤ b}
```

---

## 📊 2. Discrete Random Variables

### Probability Mass Function (PMF / Zähldichte)

```
pₓ(x) = P(X = x)

Properties:
1. pₓ(x) ≥ 0 for all x
2. Σₓ pₓ(x) = 1
```

### Cumulative Distribution Function (CDF / Verteilungsfunktion)

```
Fₓ(x) = P(X ≤ x) = Σₜ≤ₓ pₓ(t)

Properties:
1. 0 ≤ F(x) ≤ 1
2. F is monotonically increasing (monoton wachsend)
3. lim(x→-∞) F(x) = 0
4. lim(x→+∞) F(x) = 1
5. F is right-continuous (rechtsseitig stetig)
```

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from fractions import Fraction

class DiscreteRV:
    """
    Discrete random variable.
    """
    def __init__(self, pmf_dict):
        """
        Parameters:
            pmf_dict: {value: probability} dictionary
        """
        self.pmf = pmf_dict
        self.values = sorted(pmf_dict.keys())
        
        # Verify valid PMF
        total = sum(pmf_dict.values())
        assert abs(total - 1) < 1e-10, f"PMF sums to {total}, not 1"
    
    def P(self, x):
        """P(X = x)."""
        return self.pmf.get(x, 0)
    
    def CDF(self, x):
        """F(x) = P(X ≤ x)."""
        return sum(p for v, p in self.pmf.items() if v <= x)
    
    def P_range(self, a, b, inclusive='both'):
        """P(a ≤ X ≤ b) or variants."""
        if inclusive == 'both':
            return sum(p for v, p in self.pmf.items() if a <= v <= b)
        elif inclusive == 'left':
            return sum(p for v, p in self.pmf.items() if a <= v < b)
        elif inclusive == 'right':
            return sum(p for v, p in self.pmf.items() if a < v <= b)
        else:  # 'neither'
            return sum(p for v, p in self.pmf.items() if a < v < b)
    
    def plot(self, title="PMF and CDF"):
        """Plot PMF and CDF."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PMF
        axes[0].bar(self.values, [self.P(x) for x in self.values], 
                    width=0.3, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('P(X = x)')
        axes[0].set_title('PMF (Zähldichte)')
        axes[0].grid(True, alpha=0.3)
        
        # CDF (step function)
        x_plot = np.linspace(min(self.values) - 1, max(self.values) + 1, 500)
        y_plot = [self.CDF(x) for x in x_plot]
        axes[1].step(x_plot, y_plot, where='post', color='darkred', linewidth=2)
        axes[1].scatter(self.values, [self.CDF(x) for x in self.values], 
                       color='darkred', s=50, zorder=5)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('F(x) = P(X ≤ x)')
        axes[1].set_title('CDF (Verteilungsfunktion)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-0.05, 1.05)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig('pmf_cdf.png', dpi=150)
        plt.show()


# Example: Fair die
die_pmf = {i: Fraction(1, 6) for i in range(1, 7)}
X = DiscreteRV({k: float(v) for k, v in die_pmf.items()})

print("=== Fair Die ===")
print(f"P(X = 3) = {X.P(3):.4f}")
print(f"P(X ≤ 4) = {X.CDF(4):.4f}")
print(f"P(2 ≤ X ≤ 5) = {X.P_range(2, 5):.4f}")
print(f"P(X > 4) = {1 - X.CDF(4):.4f}")

X.plot("Fair Die Distribution")
```

---

## 🌊 3. Continuous Random Variables

### Probability Density Function (PDF / Dichtefunktion)

```
fₓ(x) such that P(a ≤ X ≤ b) = ∫ₐᵇ fₓ(x)dx

Properties:
1. fₓ(x) ≥ 0 for all x
2. ∫₋∞^∞ fₓ(x)dx = 1

Note: P(X = x) = 0 for continuous X!
```

### CDF for Continuous RV

```
Fₓ(x) = P(X ≤ x) = ∫₋∞ˣ fₓ(t)dt

Relationship:
fₓ(x) = d/dx Fₓ(x)  (where F is differentiable)
```

### Python Implementation

```python
from scipy import integrate
from scipy.stats import norm, expon, uniform

class ContinuousRV:
    """
    Continuous random variable defined by PDF.
    """
    def __init__(self, pdf, support=(-np.inf, np.inf)):
        """
        Parameters:
            pdf: Function f(x)
            support: (a, b) interval where pdf > 0
        """
        self.pdf = pdf
        self.support = support
        
        # Verify PDF integrates to 1
        total, _ = integrate.quad(pdf, support[0], support[1])
        assert abs(total - 1) < 1e-6, f"PDF integrates to {total}, not 1"
    
    def f(self, x):
        """PDF at x."""
        return self.pdf(x)
    
    def CDF(self, x):
        """F(x) = P(X ≤ x)."""
        if x <= self.support[0]:
            return 0
        if x >= self.support[1]:
            return 1
        result, _ = integrate.quad(self.pdf, self.support[0], x)
        return result
    
    def P_range(self, a, b):
        """P(a ≤ X ≤ b)."""
        result, _ = integrate.quad(self.pdf, a, b)
        return result
    
    def quantile(self, p):
        """Find x such that F(x) = p (inverse CDF)."""
        from scipy.optimize import brentq
        
        def target(x):
            return self.CDF(x) - p
        
        # Find bounds
        a, b = self.support
        if a == -np.inf:
            a = -1000
        if b == np.inf:
            b = 1000
        
        return brentq(target, a, b)
    
    def plot(self, title="PDF and CDF"):
        """Plot PDF and CDF."""
        a, b = self.support
        if a == -np.inf:
            a = self.quantile(0.001)
        if b == np.inf:
            b = self.quantile(0.999)
        
        x = np.linspace(a, b, 500)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PDF
        axes[0].plot(x, [self.f(xi) for xi in x], 'b-', linewidth=2)
        axes[0].fill_between(x, [self.f(xi) for xi in x], alpha=0.3)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('f(x)')
        axes[0].set_title('PDF (Dichtefunktion)')
        axes[0].grid(True, alpha=0.3)
        
        # CDF
        axes[1].plot(x, [self.CDF(xi) for xi in x], 'r-', linewidth=2)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('F(x)')
        axes[1].set_title('CDF (Verteilungsfunktion)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-0.05, 1.05)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig('pdf_cdf_continuous.png', dpi=150)
        plt.show()


# Example: Uniform distribution on [0, 2]
uniform_pdf = lambda x: 0.5 if 0 <= x <= 2 else 0
X_uniform = ContinuousRV(uniform_pdf, support=(0, 2))

print("\n=== Uniform[0, 2] ===")
print(f"P(0.5 ≤ X ≤ 1.5) = {X_uniform.P_range(0.5, 1.5):.4f}")
print(f"P(X ≤ 1) = {X_uniform.CDF(1):.4f}")
print(f"Median (F⁻¹(0.5)) = {X_uniform.quantile(0.5):.4f}")

X_uniform.plot("Uniform Distribution [0, 2]")
```

---

## 📈 4. Common Discrete Distributions

### Bernoulli Distribution

```
X ~ Bernoulli(p)

P(X = 1) = p       (success)
P(X = 0) = 1 - p   (failure)

E[X] = p
Var(X) = p(1-p)
```

### Binomial Distribution (Binomialverteilung)

```
X ~ Bin(n, p)

P(X = k) = C(n,k) · pᵏ · (1-p)ⁿ⁻ᵏ,  k = 0, 1, ..., n

"Number of successes in n independent Bernoulli trials"

E[X] = np
Var(X) = np(1-p)
```

```python
from scipy.stats import binom

def binomial_distribution(n, p):
    """Create and analyze Binomial(n, p)."""
    X = binom(n, p)
    
    print(f"\n=== Binomial(n={n}, p={p}) ===")
    print(f"E[X] = np = {n*p}")
    print(f"Var(X) = np(1-p) = {n*p*(1-p)}")
    print(f"σ = √Var = {np.sqrt(n*p*(1-p)):.4f}")
    
    # Probabilities
    k_values = np.arange(0, n + 1)
    probs = X.pmf(k_values)
    
    print(f"\nP(X = {n//2}) = {X.pmf(n//2):.4f}")
    print(f"P(X ≤ {n//2}) = {X.cdf(n//2):.4f}")
    print(f"P(X ≥ {n//2}) = {1 - X.cdf(n//2 - 1):.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(k_values, probs, color='steelblue', edgecolor='black')
    plt.axvline(n*p, color='red', linestyle='--', label=f'E[X] = {n*p}')
    plt.xlabel('k')
    plt.ylabel('P(X = k)')
    plt.title(f'Binomial Distribution: n={n}, p={p}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('binomial.png', dpi=150)
    plt.show()
    
    return X


# Example: 10 coin flips, fair coin
X = binomial_distribution(10, 0.5)
```

### Geometric Distribution (Geometrische Verteilung)

```
X ~ Geo(p)

P(X = k) = (1-p)ᵏ⁻¹ · p,  k = 1, 2, 3, ...

"Number of trials until first success"

E[X] = 1/p
Var(X) = (1-p)/p²
```

```python
from scipy.stats import geom

def geometric_distribution(p):
    """Geometric distribution analysis."""
    X = geom(p)
    
    print(f"\n=== Geometric(p={p}) ===")
    print(f"E[X] = 1/p = {1/p:.4f}")
    print(f"Var(X) = (1-p)/p² = {(1-p)/p**2:.4f}")
    
    # Memoryless property
    print(f"\nMemoryless: P(X > s+t | X > s) = P(X > t)")
    s, t = 3, 2
    p_gt_s = 1 - X.cdf(s)
    p_gt_st = 1 - X.cdf(s + t)
    p_gt_t = 1 - X.cdf(t)
    print(f"P(X > {s+t} | X > {s}) = {p_gt_st / p_gt_s:.4f}")
    print(f"P(X > {t}) = {p_gt_t:.4f}")
    
    return X


X_geo = geometric_distribution(0.3)
```

### Poisson Distribution (Poisson-Verteilung)

```
X ~ Poisson(λ)

P(X = k) = λᵏe⁻λ / k!,  k = 0, 1, 2, ...

"Number of events in fixed interval" (rare events)

E[X] = λ
Var(X) = λ
```

```python
from scipy.stats import poisson

def poisson_distribution(lam):
    """Poisson distribution analysis."""
    X = poisson(lam)
    
    print(f"\n=== Poisson(λ={lam}) ===")
    print(f"E[X] = λ = {lam}")
    print(f"Var(X) = λ = {lam}")
    
    # Plot
    k_max = int(lam + 4*np.sqrt(lam))
    k_values = np.arange(0, k_max + 1)
    probs = X.pmf(k_values)
    
    plt.figure(figsize=(10, 6))
    plt.bar(k_values, probs, color='forestgreen', edgecolor='black')
    plt.axvline(lam, color='red', linestyle='--', label=f'E[X] = λ = {lam}')
    plt.xlabel('k')
    plt.ylabel('P(X = k)')
    plt.title(f'Poisson Distribution: λ={lam}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('poisson.png', dpi=150)
    plt.show()
    
    return X


# Example: Average 5 customers per hour
X_pois = poisson_distribution(5)
print(f"P(X = 5) = {X_pois.pmf(5):.4f}")
print(f"P(X ≤ 3) = {X_pois.cdf(3):.4f}")
print(f"P(X ≥ 8) = {1 - X_pois.cdf(7):.4f}")
```

---

## 🌊 5. Common Continuous Distributions

### Uniform Distribution (Gleichverteilung)

```
X ~ Uniform(a, b)

f(x) = 1/(b-a)  for a ≤ x ≤ b
F(x) = (x-a)/(b-a)  for a ≤ x ≤ b

E[X] = (a+b)/2
Var(X) = (b-a)²/12
```

### Exponential Distribution (Exponentialverteilung)

```
X ~ Exp(λ)

f(x) = λe⁻λˣ,  x ≥ 0
F(x) = 1 - e⁻λˣ,  x ≥ 0

E[X] = 1/λ
Var(X) = 1/λ²

Memoryless: P(X > s+t | X > s) = P(X > t)
```

```python
from scipy.stats import expon

def exponential_distribution(lam):
    """Exponential distribution (scipy uses scale = 1/λ)."""
    X = expon(scale=1/lam)
    
    print(f"\n=== Exponential(λ={lam}) ===")
    print(f"E[X] = 1/λ = {1/lam:.4f}")
    print(f"Var(X) = 1/λ² = {1/lam**2:.4f}")
    
    # Plot
    x = np.linspace(0, 5/lam, 500)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(x, X.pdf(x), 'b-', linewidth=2)
    axes[0].fill_between(x, X.pdf(x), alpha=0.3)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title(f'PDF: f(x) = {lam}e^(-{lam}x)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x, X.cdf(x), 'r-', linewidth=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('F(x)')
    axes[1].set_title(f'CDF: F(x) = 1 - e^(-{lam}x)')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Exponential Distribution: λ={lam}')
    plt.tight_layout()
    plt.savefig('exponential.png', dpi=150)
    plt.show()
    
    # Memoryless property demo
    s, t = 2, 3
    p_gt_s = 1 - X.cdf(s)
    p_gt_st = 1 - X.cdf(s + t)
    p_gt_t = 1 - X.cdf(t)
    
    print(f"\nMemoryless property:")
    print(f"P(X > {s+t} | X > {s}) = {p_gt_st / p_gt_s:.6f}")
    print(f"P(X > {t}) = {p_gt_t:.6f}")
    
    return X


X_exp = exponential_distribution(0.5)
```

### Normal Distribution (Normalverteilung)

```
X ~ N(μ, σ²)

f(x) = 1/(σ√(2π)) · exp(-(x-μ)²/(2σ²))

E[X] = μ
Var(X) = σ²

Standard Normal: Z ~ N(0, 1)
Standardization: Z = (X - μ)/σ
```

```python
from scipy.stats import norm

def normal_distribution(mu, sigma):
    """Normal distribution analysis."""
    X = norm(loc=mu, scale=sigma)
    
    print(f"\n=== Normal(μ={mu}, σ={sigma}) ===")
    print(f"E[X] = μ = {mu}")
    print(f"Var(X) = σ² = {sigma**2}")
    
    # Important quantiles
    print(f"\nQuantiles:")
    for p in [0.025, 0.05, 0.5, 0.95, 0.975]:
        print(f"  F⁻¹({p}) = {X.ppf(p):.4f}")
    
    # 68-95-99.7 rule
    print(f"\n68-95-99.7 Rule:")
    print(f"  P(μ-σ < X < μ+σ) = {X.cdf(mu+sigma) - X.cdf(mu-sigma):.4f}")
    print(f"  P(μ-2σ < X < μ+2σ) = {X.cdf(mu+2*sigma) - X.cdf(mu-2*sigma):.4f}")
    print(f"  P(μ-3σ < X < μ+3σ) = {X.cdf(mu+3*sigma) - X.cdf(mu-3*sigma):.4f}")
    
    # Plot
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(x, X.pdf(x), 'b-', linewidth=2)
    axes[0].fill_between(x, X.pdf(x), alpha=0.3)
    axes[0].axvline(mu, color='red', linestyle='--', label=f'μ = {mu}')
    axes[0].axvline(mu - sigma, color='orange', linestyle=':', alpha=0.7)
    axes[0].axvline(mu + sigma, color='orange', linestyle=':', alpha=0.7)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('PDF (Dichtefunktion)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x, X.cdf(x), 'r-', linewidth=2)
    axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[1].axvline(mu, color='red', linestyle='--')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('F(x)')
    axes[1].set_title('CDF (Verteilungsfunktion)')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Normal Distribution: μ={mu}, σ={sigma}')
    plt.tight_layout()
    plt.savefig('normal.png', dpi=150)
    plt.show()
    
    return X


X_norm = normal_distribution(100, 15)

# Standard normal table values
print("\n=== Standard Normal Z ~ N(0,1) ===")
Z = norm(0, 1)
print("Common values:")
print(f"  Φ(0) = {Z.cdf(0):.4f}")
print(f"  Φ(1) = {Z.cdf(1):.4f}")
print(f"  Φ(1.645) = {Z.cdf(1.645):.4f}")
print(f"  Φ(1.96) = {Z.cdf(1.96):.4f}")
print(f"  Φ(2.576) = {Z.cdf(2.576):.4f}")
```

---

## 🔄 6. Transformations of Random Variables

### Linear Transformation

```
If Y = aX + b, then:
E[Y] = aE[X] + b
Var(Y) = a²Var(X)
```

### Standardization

```
Z = (X - μ)/σ

E[Z] = 0
Var(Z) = 1
```

### General Transformation (Continuous)

```
If Y = g(X) and g is monotonic:
fᵧ(y) = fₓ(g⁻¹(y)) · |d/dy g⁻¹(y)|
```

```python
def transformation_example():
    """Example: Y = X² where X ~ N(0,1)."""
    
    # X ~ N(0,1)
    X = norm(0, 1)
    
    # Simulate Y = X²
    n_samples = 100000
    x_samples = X.rvs(n_samples)
    y_samples = x_samples ** 2
    
    # Y = X² follows Chi-squared with df=1
    from scipy.stats import chi2
    Y_theory = chi2(df=1)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(x_samples, bins=50, density=True, alpha=0.7, label='Simulated X')
    x_plot = np.linspace(-4, 4, 200)
    axes[0].plot(x_plot, X.pdf(x_plot), 'r-', linewidth=2, label='N(0,1)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].set_title('X ~ N(0,1)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(y_samples, bins=50, density=True, alpha=0.7, label='Simulated Y=X²')
    y_plot = np.linspace(0.01, 10, 200)
    axes[1].plot(y_plot, Y_theory.pdf(y_plot), 'r-', linewidth=2, label='χ²(1)')
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Y = X² ~ χ²(1)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 10)
    
    plt.tight_layout()
    plt.savefig('transformation.png', dpi=150)
    plt.show()


transformation_example()
```

---

## 📊 7. Distribution Summary Table

### Discrete Distributions

| Distribution | PMF | E[X] | Var(X) | Use Case |
|--------------|-----|------|--------|----------|
| Bernoulli(p) | p^x(1-p)^(1-x) | p | p(1-p) | Single trial |
| Binomial(n,p) | C(n,k)p^k(1-p)^(n-k) | np | np(1-p) | n trials, count successes |
| Geometric(p) | (1-p)^(k-1)p | 1/p | (1-p)/p² | Trials until success |
| Poisson(λ) | λ^k e^(-λ)/k! | λ | λ | Rare events in interval |

### Continuous Distributions

| Distribution | PDF | E[X] | Var(X) | Use Case |
|--------------|-----|------|--------|----------|
| Uniform(a,b) | 1/(b-a) | (a+b)/2 | (b-a)²/12 | Equal likelihood |
| Exponential(λ) | λe^(-λx) | 1/λ | 1/λ² | Waiting time |
| Normal(μ,σ²) | (2πσ²)^(-1/2)e^(-(x-μ)²/2σ²) | μ | σ² | Natural phenomena |

---

## 📋 8. Exam Checklist (Klausur)

### Formulas to Know

- [ ] Binomial: P(X=k) = C(n,k)p^k(1-p)^(n-k)
- [ ] Poisson: P(X=k) = λ^k e^(-λ)/k!
- [ ] Exponential: f(x) = λe^(-λx), F(x) = 1-e^(-λx)
- [ ] Normal: Standardization Z = (X-μ)/σ
- [ ] 68-95-99.7 rule for Normal

### Key Concepts

- [ ] Difference between PMF and PDF
- [ ] CDF properties and interpretation
- [ ] When to use which distribution
- [ ] Memoryless property (Geometric, Exponential)
- [ ] Poisson approximation to Binomial

### Common Exam Tasks

- [ ] Calculate P(a ≤ X ≤ b) from PMF/PDF
- [ ] Find CDF from PDF (integration)
- [ ] Use standard normal table
- [ ] Identify appropriate distribution for problem
- [ ] Transform random variables

---

## 🔗 Related Documents

- [01-probability-foundations.md](./01-probability-foundations.md) - Probability basics
- [03-expectation-variance.md](./03-expectation-variance.md) - E[X] and Var(X)
- [04-limit-theorems.md](./04-limit-theorems.md) - LLN and CLT

---

## 📚 References

- Georgii, "Stochastik", Kapitel 3-4
- Ross, "A First Course in Probability", Chapters 4-5
- Meintrup & Schäffler, "Stochastik"

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: January 30, 2026*
