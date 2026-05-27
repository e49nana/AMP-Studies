# Stochastik / Wahrscheinlichkeitsrechnung — Python Implementations

**Course:** Stochastik für AMP  
**Semester:** WiSe 2025/2026 (S3)  
**Author:** Emmanuel Nanan — TH Nürnberg

Probability, statistics, and Monte Carlo simulation in Python.

## Structure

```
code/
├── ch1-grundlagen/             # Combinatorics, probability, Bayes, Monte Carlo
├── ch2-diskrete-verteilungen/  # Bernoulli, binomial, Poisson, expected value (coming)
├── ch3-stetige-verteilungen/   # Normal, exponential, CLT (coming)
├── ch4-statistik/              # Estimation, hypothesis testing, regression (coming)
└── requirements.txt
```

## Chapter 1 — Grundlagen (4 modules)

| Module | Topics |
|---|---|
| `combinatorics.py` | Permutations, combinations, Pascal, binomial theorem, derangements |
| `probability_basics.py` | Kolmogorov axioms, Laplace, Bayes, independence |
| `conditional_probability.py` | Monty Hall, birthday paradox, Bayesian updating |
| `random_simulation.py` | Law of large numbers, Monte Carlo π, random walks |

## Quick Start

```bash
pip install numpy matplotlib
cd ch1-grundlagen
python random_simulation.py
```
