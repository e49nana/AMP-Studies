# 📋 Exam Quick Reference Card

> **WiSe 25/26 — Last-minute formulas & reminders**  
> *Good luck Emmanuel! Du schaffst das! 🍀*

---

## 🧮 Numerik

### Iterative Methods Convergence

| Method | Converges if | Rate |
|--------|--------------|------|
| Jacobi | A strictly diagonally dominant | O(ρⁿ) |
| Gauss-Seidel | A symmetric positive definite | Faster than Jacobi |
| SOR | 0 < ω < 2 | Optimal ω needed |

**Spectral radius:** ρ(M) < 1 ⟹ convergence

### Error Formulas

| Method | Local Error | Global Error |
|--------|-------------|--------------|
| Euler | O(h²) | O(h) |
| Heun/Midpoint | O(h³) | O(h²) |
| RK4 | O(h⁵) | O(h⁴) |

### Interpolation

**Lagrange:**
$$P(x) = \sum_{i=0}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$$

**Newton divided differences:**
$$P(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1) + ...$$

**Interpolation error:**
$$|f(x) - P_n(x)| \leq \frac{M_{n+1}}{(n+1)!} \prod_{i=0}^{n}|x - x_i|$$

---

## 📊 Stochastik

### Distributions Cheatsheet

| Distribution | E[X] | Var(X) |
|--------------|------|--------|
| Bernoulli(p) | p | p(1-p) |
| Binomial(n,p) | np | np(1-p) |
| Poisson(λ) | λ | λ |
| Geometric(p) | 1/p | (1-p)/p² |
| Uniform(a,b) | (a+b)/2 | (b-a)²/12 |
| Exponential(λ) | 1/λ | 1/λ² |
| Normal(μ,σ²) | μ | σ² |

### Key Z-Values

```
90% CI → z = 1.645
95% CI → z = 1.960
99% CI → z = 2.576
```

### Confidence Interval (Mean)

$$\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

### Hypothesis Test

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

---

## 📐 Funktionale Analysis

### Norms

| Norm | Definition |
|------|------------|
| L¹ | ‖x‖₁ = Σ\|xᵢ\| |
| L² | ‖x‖₂ = √(Σxᵢ²) |
| L∞ | ‖x‖∞ = max\|xᵢ\| |
| Frobenius | ‖A‖_F = √(Σaᵢⱼ²) |

### Key Inequalities

**Cauchy-Schwarz:**
$$|\langle x, y \rangle| \leq \|x\| \cdot \|y\|$$

**Triangle:**
$$\|x + y\| \leq \|x\| + \|y\|$$

**Parallelogram (Hilbert space):**
$$\|x + y\|^2 + \|x - y\|^2 = 2(\|x\|^2 + \|y\|^2)$$

### Banach vs Hilbert

| Property | Banach | Hilbert |
|----------|--------|---------|
| Norm | ✓ | ✓ |
| Inner product | ✗ | ✓ |
| Complete | ✓ | ✓ |
| Parallelogram law | ✗ | ✓ |

---

## ⚡ Physik II — Thermodynamik

### Ideal Gas Law
$$PV = nRT = Nk_BT$$

### First Law
$$\Delta U = Q - W$$

### Entropy
$$\Delta S = \frac{Q_{rev}}{T}$$

### Heat Capacities
$$C_p - C_v = nR$$

### Carnot Efficiency
$$\eta = 1 - \frac{T_c}{T_h}$$

---

## ⚡ Physik III — E&M

### Maxwell's Equations

| Law | Differential | Integral |
|-----|--------------|----------|
| Gauss (E) | ∇·E = ρ/ε₀ | ∮E·dA = Q/ε₀ |
| Gauss (B) | ∇·B = 0 | ∮B·dA = 0 |
| Faraday | ∇×E = -∂B/∂t | ∮E·dl = -dΦ_B/dt |
| Ampère | ∇×B = μ₀J + μ₀ε₀∂E/∂t | ∮B·dl = μ₀I |

### Key Constants
```
ε₀ = 8.85 × 10⁻¹² F/m
μ₀ = 4π × 10⁻⁷ H/m
c = 3 × 10⁸ m/s
e = 1.6 × 10⁻¹⁹ C
```

---

## 🔢 Diskrete Mathematik

### Combinatorics

| Type | Formula |
|------|---------|
| Permutations | n! |
| k-Permutations | n!/(n-k)! |
| Combinations | C(n,k) = n!/[k!(n-k)!] |
| With repetition | C(n+k-1, k) |

### Graph Theory

**Handshaking Lemma:**
$$\sum_{v \in V} \deg(v) = 2|E|$$

**Euler path exists if:** 0 or 2 vertices of odd degree

**Euler circuit exists if:** All vertices have even degree

---

## 💻 Programmierung (C#)

### OOP Principles
```
- Encapsulation (private fields, public methods)
- Inheritance (: base class)
- Polymorphism (virtual/override)
- Abstraction (abstract class, interface)
```

### SOLID
```
S - Single Responsibility
O - Open/Closed
L - Liskov Substitution
I - Interface Segregation
D - Dependency Inversion
```

---

## 🎯 Exam Strategy

1. **Read all questions first** — start with easiest
2. **Show your work** — partial credit matters
3. **Check units** — especially in Physics
4. **Manage time** — don't get stuck on one problem
5. **Review at end** — catch silly mistakes

---

## 💪 Final Reminder

```
Tu as travaillé dur.
Tu connais la matière.
Tu as implémenté chaque concept.

Maintenant, montre ce que tu sais.

VIEL ERFOLG! 🍀
```

---

*AMP-Studies — WiSe 25/26*  
*Created: January 19, 2026*  
*Exams: January 20 — February 13, 2026*
