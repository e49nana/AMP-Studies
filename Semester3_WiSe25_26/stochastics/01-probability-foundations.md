# Probability Foundations (Wahrscheinlichkeitsgrundlagen)

## 📐 Introduction

Probability theory provides the mathematical framework for modeling uncertainty. This document covers axiomatic foundations, combinatorics, and fundamental concepts essential for your Stochastik exam.

---

## 🎯 1. Axiomatic Probability (Kolmogorov-Axiome)

### Sample Space and Events

```
Ω = Sample space (Ergebnisraum) - set of all possible outcomes
ω ∈ Ω = Elementary event (Elementarereignis)
A ⊆ Ω = Event (Ereignis)
𝓕 = σ-algebra of events (Ereignisalgebra)
```

### Kolmogorov Axioms

```
1. Non-negativity: P(A) ≥ 0 for all A ∈ 𝓕

2. Normalization: P(Ω) = 1

3. σ-Additivity: For disjoint A₁, A₂, ... ∈ 𝓕:
   P(⋃ᵢ₌₁^∞ Aᵢ) = Σᵢ₌₁^∞ P(Aᵢ)
```

### Consequences

```
P(∅) = 0
P(Aᶜ) = 1 - P(A)
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
A ⊆ B ⟹ P(A) ≤ P(B)  (Monotonie)
P(A) ≤ 1
```

### Python Implementation

```python
import numpy as np
from itertools import combinations, permutations
from fractions import Fraction

class ProbabilitySpace:
    """
    Discrete probability space.
    """
    def __init__(self, outcomes, probabilities=None):
        """
        Parameters:
            outcomes: List of elementary events
            probabilities: Dict {outcome: probability} or None for uniform
        """
        self.omega = set(outcomes)
        
        if probabilities is None:
            # Uniform distribution (Laplace)
            p = Fraction(1, len(outcomes))
            self.P = {omega: p for omega in outcomes}
        else:
            self.P = probabilities
            
        # Verify axioms
        assert all(p >= 0 for p in self.P.values()), "Negative probability!"
        assert sum(self.P.values()) == 1, "Probabilities don't sum to 1!"
    
    def prob(self, event):
        """P(A) for event A ⊆ Ω."""
        if callable(event):
            # Event defined by predicate
            event = {omega for omega in self.omega if event(omega)}
        return sum(self.P[omega] for omega in event if omega in self.P)
    
    def complement(self, event):
        """Aᶜ = Ω \ A."""
        return self.omega - event
    
    def prob_complement(self, event):
        """P(Aᶜ) = 1 - P(A)."""
        return 1 - self.prob(event)


# Example: Fair die (Würfel)
die = ProbabilitySpace([1, 2, 3, 4, 5, 6])

# P(even number)
even = {2, 4, 6}
print(f"P(even) = {die.prob(even)} = {float(die.prob(even)):.4f}")

# P(greater than 4)
gt4 = {5, 6}
print(f"P(X > 4) = {die.prob(gt4)} = {float(die.prob(gt4)):.4f}")

# P(even OR > 4) using inclusion-exclusion
even_or_gt4 = even | gt4
print(f"P(even ∪ X>4) = {die.prob(even_or_gt4)}")
```

---

## 🔢 2. Combinatorics (Kombinatorik)

### Counting Principles

| Type | Ordered? | Replacement? | Formula | Name (German) |
|------|----------|--------------|---------|---------------|
| Permutation | Yes | No | n! | Permutation |
| k-Permutation | Yes | No | n!/(n-k)! | Variation ohne Wdh. |
| k-Permutation | Yes | Yes | nᵏ | Variation mit Wdh. |
| Combination | No | No | C(n,k) = n!/(k!(n-k)!) | Kombination ohne Wdh. |
| Combination | No | Yes | C(n+k-1,k) | Kombination mit Wdh. |

### Binomial Coefficient

```
(n)   n!        n(n-1)...(n-k+1)
( ) = ―――――― = ――――――――――――――――
(k)   k!(n-k)!        k!

Properties:
(n) = (  n  )     Symmetry
(k)   (n-k)

(n) + ( n ) = (n+1)   Pascal's rule
(k)   (k+1)   (k+1)

Σₖ₌₀ⁿ (n) = 2ⁿ        Sum of row
      (k)
```

### Python Implementation

```python
from math import factorial, comb
from functools import lru_cache

@lru_cache(maxsize=None)
def binomial(n, k):
    """Binomial coefficient C(n,k)."""
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))


def permutations_count(n, k=None):
    """
    Number of k-permutations of n elements.
    P(n,k) = n!/(n-k)!
    """
    if k is None:
        k = n
    return factorial(n) // factorial(n - k)


def combinations_with_replacement(n, k):
    """
    Combinations with replacement.
    C(n+k-1, k) = (n+k-1)! / (k!(n-1)!)
    """
    return binomial(n + k - 1, k)


# Examples
print("=== Combinatorics Examples ===\n")

# How many ways to arrange 5 books on a shelf?
print(f"Arrange 5 books: {factorial(5)} = 5!")

# How many 3-letter codes from 26 letters (no repetition)?
print(f"3-letter codes (no rep): {permutations_count(26, 3)}")

# How many 3-letter codes from 26 letters (with repetition)?
print(f"3-letter codes (with rep): {26**3}")

# How many ways to choose 5 cards from 52?
print(f"Choose 5 from 52: {binomial(52, 5)}")

# How many ways to distribute 10 identical balls into 4 boxes?
print(f"10 balls → 4 boxes: {combinations_with_replacement(4, 10)}")
```

### Multinomial Coefficient

```
(    n    )        n!
(         ) = ――――――――――――
(k₁,k₂,...,kᵣ)   k₁!k₂!...kᵣ!

Where k₁ + k₂ + ... + kᵣ = n
```

```python
def multinomial(n, groups):
    """
    Multinomial coefficient.
    Number of ways to partition n items into groups of sizes k₁, k₂, ...
    """
    assert sum(groups) == n, "Groups must sum to n"
    
    result = factorial(n)
    for k in groups:
        result //= factorial(k)
    return result


# How many ways to arrange MISSISSIPPI?
# M:1, I:4, S:4, P:2
print(f"Arrangements of MISSISSIPPI: {multinomial(11, [1, 4, 4, 2])}")
```

---

## 🎲 3. Laplace Probability

### Definition

For finite Ω with equally likely outcomes:

```
P(A) = |A| / |Ω| = favorable outcomes / total outcomes
```

### Classic Problems

```python
def laplace_probability(favorable, total):
    """P(A) = |A|/|Ω| for equally likely outcomes."""
    return Fraction(favorable, total)


# Example 1: Two dice sum to 7
omega_2dice = [(i, j) for i in range(1, 7) for j in range(1, 7)]
sum_7 = [(i, j) for i, j in omega_2dice if i + j == 7]
p_sum7 = laplace_probability(len(sum_7), len(omega_2dice))
print(f"P(sum = 7 with 2 dice) = {p_sum7} = {float(p_sum7):.4f}")

# Example 2: At least one 6 with 3 dice
omega_3dice = [(i, j, k) for i in range(1, 7) 
                         for j in range(1, 7) 
                         for k in range(1, 7)]
at_least_one_6 = [x for x in omega_3dice if 6 in x]
p_at_least_6 = laplace_probability(len(at_least_one_6), len(omega_3dice))
print(f"P(at least one 6 with 3 dice) = {p_at_least_6} ≈ {float(p_at_least_6):.4f}")

# Complement method: 1 - P(no 6)
p_no_6 = Fraction(5, 6) ** 3
p_at_least_6_v2 = 1 - p_no_6
print(f"Via complement: 1 - (5/6)³ = {p_at_least_6_v2} ≈ {float(p_at_least_6_v2):.4f}")
```

### Birthday Problem (Geburtstagsproblem)

```python
def birthday_probability(n, days=365):
    """
    Probability that at least 2 of n people share a birthday.
    
    P(at least one match) = 1 - P(all different)
    P(all different) = 365/365 × 364/365 × ... × (365-n+1)/365
    """
    if n > days:
        return 1.0
    
    p_all_different = 1.0
    for i in range(n):
        p_all_different *= (days - i) / days
    
    return 1 - p_all_different


# Find n where P > 0.5
print("\n=== Birthday Problem ===")
for n in [10, 20, 23, 30, 50, 70]:
    p = birthday_probability(n)
    print(f"n = {n}: P(match) = {p:.4f}")
```

---

## 🔄 4. Conditional Probability (Bedingte Wahrscheinlichkeit)

### Definition

```
P(A|B) = P(A ∩ B) / P(B),  provided P(B) > 0

"Probability of A given B"
```

### Multiplication Rule

```
P(A ∩ B) = P(A|B) · P(B) = P(B|A) · P(A)

Chain rule:
P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = P(A₁) · P(A₂|A₁) · P(A₃|A₁∩A₂) · ...
```

### Python Implementation

```python
def conditional_probability(p_a_and_b, p_b):
    """P(A|B) = P(A ∩ B) / P(B)."""
    if p_b == 0:
        raise ValueError("P(B) = 0, conditional probability undefined")
    return p_a_and_b / p_b


# Example: Card drawing without replacement
# Draw 2 cards. P(2nd is Ace | 1st is Ace)?

p_first_ace = Fraction(4, 52)
p_both_aces = Fraction(4, 52) * Fraction(3, 51)
p_second_ace_given_first = conditional_probability(p_both_aces, p_first_ace)

print(f"P(2nd Ace | 1st Ace) = {p_second_ace_given_first} = {float(p_second_ace_given_first):.4f}")
# Direct: 3/51 since one Ace is gone
print(f"Direct calculation: 3/51 = {Fraction(3, 51)}")
```

---

## 📊 5. Law of Total Probability (Satz von der totalen Wahrscheinlichkeit)

### Theorem

If B₁, B₂, ..., Bₙ partition Ω (disjoint, union = Ω):

```
P(A) = Σᵢ₌₁ⁿ P(A|Bᵢ) · P(Bᵢ)
```

### Visualization

```
        Ω
   ┌────┬────┬────┐
   │ B₁ │ B₂ │ B₃ │
   │    │    │    │
   │ A∩B₁│A∩B₂│A∩B₃│  ← Event A crosses partitions
   └────┴────┴────┘

P(A) = P(A∩B₁) + P(A∩B₂) + P(A∩B₃)
     = P(A|B₁)P(B₁) + P(A|B₂)P(B₂) + P(A|B₃)P(B₃)
```

### Example

```python
def total_probability(conditionals, partition_probs):
    """
    Law of total probability.
    
    Parameters:
        conditionals: [P(A|B₁), P(A|B₂), ...]
        partition_probs: [P(B₁), P(B₂), ...]
    
    Returns:
        P(A)
    """
    assert abs(sum(partition_probs) - 1) < 1e-10, "Not a valid partition"
    return sum(p_a_bi * p_bi for p_a_bi, p_bi in zip(conditionals, partition_probs))


# Example: Factory with 3 machines
# Machine 1: 30% of production, 2% defective
# Machine 2: 45% of production, 3% defective
# Machine 3: 25% of production, 5% defective
# P(defective item)?

production = [0.30, 0.45, 0.25]  # P(Bᵢ)
defect_rates = [0.02, 0.03, 0.05]  # P(Defective|Bᵢ)

p_defective = total_probability(defect_rates, production)
print(f"P(defective) = {p_defective:.4f}")

# Detailed calculation
for i, (prod, defect) in enumerate(zip(production, defect_rates), 1):
    print(f"  Machine {i}: {prod:.0%} × {defect:.0%} = {prod * defect:.4f}")
```

---

## 🔀 6. Bayes' Theorem (Satz von Bayes)

### Formula

```
P(Bⱼ|A) = P(A|Bⱼ) · P(Bⱼ) / P(A)
        = P(A|Bⱼ) · P(Bⱼ) / Σᵢ P(A|Bᵢ) · P(Bᵢ)
```

### Terminology

```
P(Bⱼ)     = Prior probability (a priori)
P(Bⱼ|A)   = Posterior probability (a posteriori)
P(A|Bⱼ)   = Likelihood
P(A)      = Evidence (Normalization)
```

### Python Implementation

```python
def bayes_theorem(likelihood, prior, evidence=None, all_likelihoods=None, all_priors=None):
    """
    Bayes' theorem: P(B|A) = P(A|B)P(B) / P(A)
    
    If evidence not given, compute via total probability.
    """
    if evidence is None:
        if all_likelihoods is None or all_priors is None:
            raise ValueError("Need evidence or all likelihoods/priors")
        evidence = total_probability(all_likelihoods, all_priors)
    
    return likelihood * prior / evidence


# Example: Medical test
# Disease prevalence: 1%
# Test sensitivity (true positive): 95%
# Test specificity (true negative): 90%
# If test is positive, what's P(disease)?

p_disease = 0.01  # Prior
p_healthy = 0.99

p_pos_given_disease = 0.95  # Sensitivity
p_pos_given_healthy = 0.10  # 1 - Specificity (false positive)

# P(positive)
p_positive = total_probability(
    [p_pos_given_disease, p_pos_given_healthy],
    [p_disease, p_healthy]
)

# P(disease | positive)
p_disease_given_pos = bayes_theorem(
    likelihood=p_pos_given_disease,
    prior=p_disease,
    evidence=p_positive
)

print("\n=== Medical Test (Bayes) ===")
print(f"P(Disease) = {p_disease:.2%} (prior)")
print(f"P(Positive) = {p_positive:.4f}")
print(f"P(Disease | Positive) = {p_disease_given_pos:.2%} (posterior)")
print("\nEven with 95% sensitivity, low prevalence means most positives are false!")
```

### Bayes Update Table

```python
def bayes_table(hypotheses, priors, likelihoods):
    """
    Create Bayes update table.
    
    Parameters:
        hypotheses: Names of hypotheses
        priors: P(Hᵢ)
        likelihoods: P(Evidence|Hᵢ)
    """
    evidence = sum(l * p for l, p in zip(likelihoods, priors))
    posteriors = [l * p / evidence for l, p in zip(likelihoods, priors)]
    
    print(f"{'Hypothesis':<15} {'Prior':<10} {'Likelihood':<12} {'Prior×Like':<12} {'Posterior':<10}")
    print("-" * 60)
    
    for h, prior, like, post in zip(hypotheses, priors, likelihoods, posteriors):
        print(f"{h:<15} {prior:<10.4f} {like:<12.4f} {prior*like:<12.4f} {post:<10.4f}")
    
    print("-" * 60)
    print(f"{'Total':<15} {sum(priors):<10.4f} {'':<12} {evidence:<12.4f} {sum(posteriors):<10.4f}")
    
    return posteriors


# Example: Which machine produced a defective item?
print("\n=== Bayes Table: Which Machine? ===")
posteriors = bayes_table(
    hypotheses=["Machine 1", "Machine 2", "Machine 3"],
    priors=[0.30, 0.45, 0.25],
    likelihoods=[0.02, 0.03, 0.05]
)
```

---

## 🔗 7. Independence (Unabhängigkeit)

### Definition

Events A and B are independent if:

```
P(A ∩ B) = P(A) · P(B)

Equivalently:
P(A|B) = P(A)  (knowing B doesn't change A)
P(B|A) = P(B)
```

### Mutual Independence

Events A₁, ..., Aₙ are mutually independent if:

```
P(Aᵢ₁ ∩ Aᵢ₂ ∩ ... ∩ Aᵢₖ) = P(Aᵢ₁) · P(Aᵢ₂) · ... · P(Aᵢₖ)

for ALL subsets {i₁, ..., iₖ} ⊆ {1, ..., n}
```

### Pairwise vs Mutual Independence

```python
# Example: Pairwise independent but NOT mutually independent

# Two fair coin flips
# A = "First coin is Heads"
# B = "Second coin is Heads"  
# C = "Both coins show the same"

omega = [('H', 'H'), ('H', 'T'), ('T', 'H'), ('T', 'T')]
p = {x: Fraction(1, 4) for x in omega}

A = {('H', 'H'), ('H', 'T')}  # First is H
B = {('H', 'H'), ('T', 'H')}  # Second is H
C = {('H', 'H'), ('T', 'T')}  # Same

def prob(event):
    return sum(p[x] for x in event)

print("\n=== Independence Check ===")
print(f"P(A) = {prob(A)}")
print(f"P(B) = {prob(B)}")
print(f"P(C) = {prob(C)}")

print(f"\nP(A∩B) = {prob(A & B)}, P(A)P(B) = {prob(A) * prob(B)}")
print(f"P(A∩C) = {prob(A & C)}, P(A)P(C) = {prob(A) * prob(C)}")
print(f"P(B∩C) = {prob(B & C)}, P(B)P(C) = {prob(B) * prob(C)}")
print("→ Pairwise independent!")

print(f"\nP(A∩B∩C) = {prob(A & B & C)}, P(A)P(B)P(C) = {prob(A) * prob(B) * prob(C)}")
print("→ NOT mutually independent!")
```

### Independent Trials

```python
def independent_trials(p_success, n_trials, k_successes):
    """
    P(exactly k successes in n independent trials).
    
    This is the Binomial probability (preview).
    """
    return binomial(n_trials, k_successes) * (p_success ** k_successes) * ((1 - p_success) ** (n_trials - k_successes))


# Example: Flip fair coin 10 times, P(exactly 6 heads)?
p = independent_trials(0.5, 10, 6)
print(f"\nP(6 heads in 10 flips) = {p:.4f}")

# P(at least 8 heads)?
p_at_least_8 = sum(independent_trials(0.5, 10, k) for k in range(8, 11))
print(f"P(at least 8 heads in 10 flips) = {p_at_least_8:.4f}")
```

---

## 📋 8. Important Formulas Summary

### Set Operations

```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
P(A ∪ B ∪ C) = P(A) + P(B) + P(C) - P(A∩B) - P(A∩C) - P(B∩C) + P(A∩B∩C)
P(Aᶜ) = 1 - P(A)
```

### Conditional Probability

```
P(A|B) = P(A ∩ B) / P(B)
P(A ∩ B) = P(A|B) · P(B)
```

### Total Probability & Bayes

```
P(A) = Σᵢ P(A|Bᵢ) · P(Bᵢ)

P(Bⱼ|A) = P(A|Bⱼ) · P(Bⱼ) / Σᵢ P(A|Bᵢ) · P(Bᵢ)
```

### Independence

```
A, B independent ⟺ P(A ∩ B) = P(A) · P(B)
```

### Combinatorics

```
n! = n × (n-1) × ... × 1
P(n,k) = n!/(n-k)!
C(n,k) = n!/(k!(n-k)!)
```

---

## 📋 9. Exam Checklist (Klausur)

### Formulas to Know

- [ ] Kolmogorov axioms
- [ ] Inclusion-exclusion: P(A∪B) = P(A) + P(B) - P(A∩B)
- [ ] Conditional probability: P(A|B) = P(A∩B)/P(B)
- [ ] Total probability
- [ ] Bayes' theorem
- [ ] Binomial coefficient

### Key Concepts

- [ ] When to use combinations vs permutations
- [ ] Difference between pairwise and mutual independence
- [ ] How to set up a Bayes problem
- [ ] Complement rule for "at least one"

### Common Exam Tasks

- [ ] Count arrangements/selections
- [ ] Calculate conditional probabilities
- [ ] Apply Bayes' theorem to diagnostic problems
- [ ] Check independence of events
- [ ] Use total probability with partitions

---

## 🔗 Related Documents

- [02-random-variables.md](./02-random-variables.md) - Random variables & distributions
- [03-expectation-variance.md](./03-expectation-variance.md) - Expected value & variance
- [04-distributions.md](./04-distributions.md) - Common distributions

---

## 📚 References

- Georgii, "Stochastik", Kapitel 1-2
- Ross, "A First Course in Probability"
- Meintrup & Schäffler, "Stochastik"

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: January 29, 2026*
