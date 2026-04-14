"""
Stochastik — Essential Functions
=================================
Probability distributions, statistical tests, and stochastic utilities.
Quick reference implementations for exam preparation.

Author: Emmanuel Nana Nana
Date: January 18, 2026
Repo: AMP-Studies
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy import stats


# =============================================================================
# DISCRETE DISTRIBUTIONS
# =============================================================================

def bernoulli(p: float, size: int = 1) -> np.ndarray:
    """
    Bernoulli distribution.
    X ∈ {0, 1}, P(X=1) = p
    
    E[X] = p
    Var(X) = p(1-p)
    """
    return np.random.binomial(1, p, size)


def binomial_pmf(k: int, n: int, p: float) -> float:
    """
    Binomial PMF: P(X = k)
    X = number of successes in n trials
    
    P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
    E[X] = np
    Var(X) = np(1-p)
    """
    return stats.binom.pmf(k, n, p)


def poisson_pmf(k: int, lam: float) -> float:
    """
    Poisson PMF: P(X = k)
    Models rare events in fixed interval.
    
    P(X=k) = (λ^k * e^(-λ)) / k!
    E[X] = λ
    Var(X) = λ
    """
    return stats.poisson.pmf(k, lam)


def geometric_pmf(k: int, p: float) -> float:
    """
    Geometric PMF: P(X = k)
    X = number of trials until first success
    
    P(X=k) = (1-p)^(k-1) * p
    E[X] = 1/p
    Var(X) = (1-p)/p²
    """
    return stats.geom.pmf(k, p)


# =============================================================================
# CONTINUOUS DISTRIBUTIONS
# =============================================================================

def uniform_pdf(x: float, a: float, b: float) -> float:
    """
    Uniform PDF: f(x) = 1/(b-a) for x ∈ [a,b]
    
    E[X] = (a+b)/2
    Var(X) = (b-a)²/12
    """
    return stats.uniform.pdf(x, loc=a, scale=b-a)


def exponential_pdf(x: float, lam: float) -> float:
    """
    Exponential PDF: f(x) = λe^(-λx) for x ≥ 0
    Models waiting times (memoryless).
    
    E[X] = 1/λ
    Var(X) = 1/λ²
    """
    return stats.expon.pdf(x, scale=1/lam)


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Normal (Gaussian) PDF.
    
    f(x) = (1/(σ√(2π))) * exp(-(x-μ)²/(2σ²))
    E[X] = μ
    Var(X) = σ²
    """
    return stats.norm.pdf(x, loc=mu, scale=sigma)


def standard_normal_cdf(z: float) -> float:
    """
    Standard normal CDF: Φ(z) = P(Z ≤ z)
    """
    return stats.norm.cdf(z)


def normal_quantile(p: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Normal quantile (inverse CDF).
    Returns x such that P(X ≤ x) = p
    """
    return stats.norm.ppf(p, loc=mu, scale=sigma)


# =============================================================================
# KEY FORMULAS & RULES
# =============================================================================

def z_score(x: float, mu: float, sigma: float) -> float:
    """
    Z-score: standardize a value.
    z = (x - μ) / σ
    """
    return (x - mu) / sigma


def chebyshev_bound(k: float) -> float:
    """
    Chebyshev's inequality.
    P(|X - μ| ≥ kσ) ≤ 1/k²
    
    Returns upper bound on probability.
    """
    return 1 / (k ** 2)


def law_of_large_numbers_demo(p: float, n_trials: int, n_simulations: int = 1000) -> np.ndarray:
    """
    Demonstrate Law of Large Numbers.
    Sample mean → population mean as n → ∞
    """
    means = []
    for n in range(1, n_trials + 1):
        samples = np.random.binomial(1, p, (n_simulations, n))
        means.append(np.mean(samples))
    return np.array(means)


def central_limit_theorem_demo(dist_func: Callable, n_samples: int = 30, 
                                n_simulations: int = 1000) -> np.ndarray:
    """
    Demonstrate Central Limit Theorem.
    Sum of iid random variables → Normal distribution.
    
    Parameters
    ----------
    dist_func : Callable
        Function that generates random samples
    n_samples : int
        Number of samples to sum
    n_simulations : int
        Number of simulations
    
    Returns
    -------
    Array of sample means (should be approximately normal)
    """
    sample_means = []
    for _ in range(n_simulations):
        samples = dist_func(n_samples)
        sample_means.append(np.mean(samples))
    return np.array(sample_means)


# =============================================================================
# STATISTICAL INFERENCE
# =============================================================================

def confidence_interval_mean(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Confidence interval for population mean.
    Uses t-distribution for small samples.
    
    Parameters
    ----------
    data : Sample data
    confidence : Confidence level (default 95%)
    
    Returns
    -------
    (lower, upper) bounds
    """
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error
    
    # t critical value
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin = t_crit * se
    return (mean - margin, mean + margin)


def confidence_interval_proportion(successes: int, n: int, 
                                    confidence: float = 0.95) -> Tuple[float, float]:
    """
    Confidence interval for proportion (Wald interval).
    
    p̂ ± z * √(p̂(1-p̂)/n)
    """
    p_hat = successes / n
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha/2)
    
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    margin = z * se
    
    return (p_hat - margin, p_hat + margin)


def hypothesis_test_mean(data: np.ndarray, mu_0: float, 
                         alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    One-sample t-test for population mean.
    
    H0: μ = μ_0
    H1: μ ≠ μ_0 (two-sided) / μ > μ_0 (greater) / μ < μ_0 (less)
    
    Returns
    -------
    (t_statistic, p_value)
    """
    return stats.ttest_1samp(data, mu_0, alternative=alternative)


def hypothesis_test_proportion(successes: int, n: int, p_0: float,
                               alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    One-sample z-test for proportion.
    
    H0: p = p_0
    
    Returns
    -------
    (z_statistic, p_value)
    """
    p_hat = successes / n
    se = np.sqrt(p_0 * (1 - p_0) / n)
    z = (p_hat - p_0) / se
    
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z)
    else:  # less
        p_value = stats.norm.cdf(z)
    
    return z, p_value


# =============================================================================
# EXPECTED VALUE & VARIANCE RULES
# =============================================================================

def expected_value_rules():
    """
    Key rules for expected value (print summary).
    """
    rules = """
    ═══════════════════════════════════════════════════════════
    EXPECTED VALUE RULES
    ═══════════════════════════════════════════════════════════
    
    Linearity:
        E[aX + b] = a·E[X] + b
        E[X + Y] = E[X] + E[Y]  (always, even if dependent!)
    
    Product (independent only):
        E[XY] = E[X]·E[Y]  (if X, Y independent)
    
    Conditional:
        E[X] = E[E[X|Y]]  (Law of Total Expectation)
    
    Common distributions:
        Bernoulli(p):     E[X] = p
        Binomial(n,p):    E[X] = np
        Poisson(λ):       E[X] = λ
        Geometric(p):     E[X] = 1/p
        Uniform(a,b):     E[X] = (a+b)/2
        Exponential(λ):   E[X] = 1/λ
        Normal(μ,σ²):     E[X] = μ
    ═══════════════════════════════════════════════════════════
    """
    print(rules)


def variance_rules():
    """
    Key rules for variance (print summary).
    """
    rules = """
    ═══════════════════════════════════════════════════════════
    VARIANCE RULES
    ═══════════════════════════════════════════════════════════
    
    Definition:
        Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
    
    Scaling:
        Var(aX + b) = a²·Var(X)
    
    Sum (independent):
        Var(X + Y) = Var(X) + Var(Y)  (if independent)
    
    Sum (general):
        Var(X + Y) = Var(X) + Var(Y) + 2·Cov(X,Y)
    
    Common distributions:
        Bernoulli(p):     Var(X) = p(1-p)
        Binomial(n,p):    Var(X) = np(1-p)
        Poisson(λ):       Var(X) = λ
        Geometric(p):     Var(X) = (1-p)/p²
        Uniform(a,b):     Var(X) = (b-a)²/12
        Exponential(λ):   Var(X) = 1/λ²
        Normal(μ,σ²):     Var(X) = σ²
    ═══════════════════════════════════════════════════════════
    """
    print(rules)


# =============================================================================
# QUICK REFERENCE TABLES
# =============================================================================

def print_z_table():
    """Print common z-values for quick reference."""
    table = """
    ═══════════════════════════════════════════════════════════
    COMMON Z-VALUES (Standard Normal)
    ═══════════════════════════════════════════════════════════
    
    Confidence Level    α       z_{α/2}
    ─────────────────────────────────────
    90%                 0.10    1.645
    95%                 0.05    1.960
    99%                 0.01    2.576
    
    Percentiles:
    ─────────────────────────────────────
    Φ(z) = 0.90  →  z = 1.282
    Φ(z) = 0.95  →  z = 1.645
    Φ(z) = 0.975 →  z = 1.960
    Φ(z) = 0.99  →  z = 2.326
    Φ(z) = 0.995 →  z = 2.576
    
    68-95-99.7 Rule:
    ─────────────────────────────────────
    P(|Z| < 1) ≈ 68.3%
    P(|Z| < 2) ≈ 95.4%
    P(|Z| < 3) ≈ 99.7%
    ═══════════════════════════════════════════════════════════
    """
    print(table)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STOCHASTIK ESSENTIALS — EXAM PREP")
    print("=" * 60)
    
    # --- Distribution Examples ---
    print("\n1. Distribution Examples")
    print("-" * 40)
    
    print(f"   Binomial(n=10, p=0.3):")
    print(f"     P(X=3) = {binomial_pmf(3, 10, 0.3):.4f}")
    print(f"     E[X] = {10 * 0.3}, Var(X) = {10 * 0.3 * 0.7:.2f}")
    
    print(f"\n   Poisson(λ=5):")
    print(f"     P(X=3) = {poisson_pmf(3, 5):.4f}")
    print(f"     E[X] = 5, Var(X) = 5")
    
    print(f"\n   Normal(μ=100, σ=15):")
    print(f"     P(X < 115) = Φ((115-100)/15) = Φ(1) = {standard_normal_cdf(1):.4f}")
    print(f"     95th percentile = {normal_quantile(0.95, 100, 15):.2f}")
    
    # --- Confidence Interval ---
    print("\n2. Confidence Interval")
    print("-" * 40)
    
    data = np.random.normal(50, 10, 30)  # Sample from N(50, 100)
    ci = confidence_interval_mean(data, 0.95)
    print(f"   Sample mean: {np.mean(data):.2f}")
    print(f"   95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
    
    # --- Hypothesis Test ---
    print("\n3. Hypothesis Test")
    print("-" * 40)
    
    t_stat, p_val = hypothesis_test_mean(data, mu_0=50)
    print(f"   H0: μ = 50")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value: {p_val:.4f}")
    print(f"   Decision (α=0.05): {'Reject H0' if p_val < 0.05 else 'Fail to reject H0'}")
    
    # --- Quick Reference ---
    print("\n4. Quick Reference Tables")
    print("-" * 40)
    expected_value_rules()
    print_z_table()
    
    print("\n" + "=" * 60)
    print("Viel Erfolg bei der Prüfung! 🍀")
    print("=" * 60)
