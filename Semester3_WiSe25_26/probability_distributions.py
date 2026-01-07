"""Probability Distributions — Python Reference

Companion Python utilities for the probability-distributions-cheatsheet.md file.
Provides simple helper functions, examples, and SciPy wrappers for common
discrete and continuous probability distributions.
"""

import numpy as np
from scipy import stats


# =====================
# Discrete Distributions
# =====================

def bernoulli_pmf(k: int, p: float) -> float:
    """Bernoulli PMF: P(X=k), k in {0,1}."""
    if k not in (0, 1):
        return 0.0
    return p**k * (1 - p)**(1 - k)


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Binomial PMF using SciPy."""
    return stats.binom.pmf(k, n=n, p=p)


def binomial_cdf(k: int, n: int, p: float) -> float:
    """Binomial CDF using SciPy."""
    return stats.binom.cdf(k, n=n, p=p)


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF."""
    return stats.poisson.pmf(k, mu=lam)


def geometric_pmf(k: int, p: float) -> float:
    """Geometric PMF (number of trials until first success)."""
    if k < 1:
        return 0.0
    return (1 - p)**(k - 1) * p


# =======================
# Continuous Distributions
# =======================

def uniform_pdf(x: float, a: float, b: float) -> float:
    """Uniform PDF on [a, b]."""
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def exponential_pdf(x: float, lam: float) -> float:
    """Exponential PDF with rate lambda."""
    if x < 0:
        return 0.0
    return lam * np.exp(-lam * x)


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal PDF."""
    return stats.norm.pdf(x, loc=mu, scale=sigma)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal CDF."""
    return stats.norm.cdf(x, loc=mu, scale=sigma)


def normal_quantile(q: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Inverse CDF (quantile) of Normal distribution."""
    return stats.norm.ppf(q, loc=mu, scale=sigma)


# =====================
# Sampling Utilities
# =====================

def sample_normal(mu: float = 0.0, sigma: float = 1.0, size: int = 1000):
    """Generate samples from N(mu, sigma^2)."""
    return stats.norm.rvs(loc=mu, scale=sigma, size=size)


def sample_poisson(lam: float, size: int = 1000):
    """Generate Poisson samples."""
    return stats.poisson.rvs(mu=lam, size=size)


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("Bernoulli PMF P(X=1), p=0.3:", bernoulli_pmf(1, 0.3))
    print("Binomial P(X<=3), n=10, p=0.5:", binomial_cdf(3, 10, 0.5))
    print("Poisson P(X=2), lambda=4:", poisson_pmf(2, 4))
    print("Geometric P(X=5), p=0.2:", geometric_pmf(5, 0.2))

    print("Normal PDF at x=0:", normal_pdf(0))
    print("Normal CDF at x=1.96:", normal_cdf(1.96))
    print("Normal 97.5% quantile:", normal_quantile(0.975))

    samples = sample_normal()
    print("Generated normal samples:", samples[:5])
