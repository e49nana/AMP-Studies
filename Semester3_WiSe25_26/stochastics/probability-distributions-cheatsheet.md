# Probability Distributions — Cheatsheet  

## Discrete Distributions  

### Bernoulli Distribution  
Single trial with success probability \(p\).

- **PMF**: \(P(X = k) = p^k (1-p)^{1-k}\), \(k \in \{0, 1\}\)  
- **E[X]**: \(p\)  
- **Var(X)**: \(p(1-p)\)  

---

### Binomial Distribution  
Number of successes in \(n\) independent Bernoulli trials.

- **PMF**: \(P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}\), \(k = 0,1,\dots,n\)  
- **E[X]**: \(np\)  
- **Var(X)**: \(np(1-p)\)  

```python
from scipy.stats import binom

P_X_eq_3 = binom.pmf(3, n=10, p=0.5)   # P(X = 3)
P_X_leq_3 = binom.cdf(3, n=10, p=0.5)  # P(X ≤ 3)
```

---

### Poisson Distribution  
Number of events in a fixed interval (rare events).

- **PMF**: \(P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}\), \(k = 0,1,2,\dots\)  
- **E[X]**: \(\lambda\)  
- **Var(X)**: \(\lambda\)  
- **Use case**: Approximates Binomial when \(n\) is large, \(p\) is small, and \(\lambda = np\) is moderate.

---

### Geometric Distribution  
Number of trials until the first success.

- **PMF**: \(P(X = k) = (1-p)^{k-1} p\), \(k = 1,2,\dots\)  
- **E[X]**: \(1/p\)  
- **Var(X)**: \((1-p)/p^2\)  

---

## Continuous Distributions  

### Uniform Distribution \(U(a,b)\)  
Equal probability on \([a,b]\).

- **PDF**: \(f(x) = \frac{1}{b-a}\) for \(x \in [a,b]\)  
- **E[X]**: \((a+b)/2\)  
- **Var(X)**: \((b-a)^2/12\)  

---

### Exponential Distribution \(\mathrm{Exp}(\lambda)\)  
Time until next event (memoryless), \(x \ge 0\).

- **PDF**: \(f(x) = \lambda e^{-\lambda x}\)  
- **E[X]**: \(1/\lambda\)  
- **Var(X)**: \(1/\lambda^2\)  
- **Memoryless**: \(P(X > s+t \mid X > s) = P(X > t)\)

---

### Normal Distribution \(\mathcal{N}(\mu,\sigma^2)\)

- **PDF**: \(f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(x-\mu)^2}{2\sigma^2}}\)  
- **E[X]**: \(\mu\)  
- **Var(X)**: \(\sigma^2\)  

**68–95–99.7 rule**  
- 68% within \(\mu \pm \sigma\)  
- 95% within \(\mu \pm 2\sigma\)  
- 99.7% within \(\mu \pm 3\sigma\)  

```python
from scipy.stats import norm

z_score = (x - mu) / sigma
p_value = 1 - norm.cdf(z_score)  # P(X > x)
```

---

### Chi-Square Distribution \(\chi^2(k)\)  
Sum of \(k\) squared standard normals.

- **E[X]**: \(k\)  
- **Var(X)**: \(2k\)  
- **Use**: Goodness-of-fit tests, variance testing.

---

### Student’s t-Distribution \(t(\nu)\)  
Used when sample size is small and \(\sigma\) is unknown.

- **E[X]**: 0 (for \(\nu > 1\))  
- Heavier tails than Normal  
- As \(\nu \to \infty\), \(t(\nu) \to \mathcal{N}(0,1)\)

---

## Quick Reference Table  

| Distribution | E[X] | Var(X) | Use Case |
|---|---:|---:|---|
| Bernoulli(p) | p | p(1-p) | Single yes/no |
| Binomial(n,p) | np | np(1-p) | Count of successes |
| Poisson(λ) | λ | λ | Rare events |
| Geometric(p) | 1/p | (1-p)/p² | Trials until success |
| Uniform(a,b) | (a+b)/2 | (b-a)²/12 | Equal probability |
| Exponential(λ) | 1/λ | 1/λ² | Wait times |
| Normal(μ,σ²) | μ | σ² | Many natural phenomena |

---

## Python Quick Reference  

```python
import numpy as np
from scipy import stats

# Generate samples
samples = stats.norm.rvs(loc=0, scale=1, size=1000)

# PDF and CDF
pdf_value = stats.norm.pdf(0, loc=0, scale=1)
cdf_value = stats.norm.cdf(1.96, loc=0, scale=1)  # ≈ 0.975

# Inverse CDF (quantile)
z_95 = stats.norm.ppf(0.975)  # ≈ 1.96
```
