# Chapitre 1 — Grundlagen der Wahrscheinlichkeit

> **Résumé de révision** — Stochastik für AMP, S3

## 1.1 Combinatoire

| Formule | Expression | Usage |
|---|---|---|
| Factorielle | $n! = 1 \cdot 2 \cdots n$ | permutations |
| Arrangements | $A(n,k) = n!/(n-k)!$ | $k$-uplets ordonnés |
| Combinaisons | $\binom{n}{k} = n!/(k!(n-k)!)$ | sous-ensembles |
| Avec répétition | $\binom{n+k-1}{k}$ | tirages avec remise |

**Triangle de Pascal :** $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$

**Binôme de Newton :** $(a+b)^n = \sum_{k=0}^n \binom{n}{k} a^k b^{n-k}$

**Inclusion-exclusion :** $|A \cup B| = |A| + |B| - |A \cap B|$

**Dérangements :** $D_n = n! \sum_{k=0}^n (-1)^k/k! \approx n!/e$

## 1.2 Probabilités

**Axiomes de Kolmogorov :**
1. $P(A) \geq 0$
2. $P(\Omega) = 1$
3. $P(A \cup B) = P(A) + P(B)$ si $A \cap B = \emptyset$

**Laplace :** $P(A) = |A|/|\Omega|$ (équiprobabilité)

## 1.3 Probabilités conditionnelles et Bayes

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Formule des probabilités totales :** $P(B) = \sum_i P(B|A_i) \cdot P(A_i)$

**Bayes :** $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$

**Indépendance :** $A \perp B \iff P(A \cap B) = P(A) \cdot P(B)$

**Paradoxes classiques :**
- **Monty Hall :** changer gagne $2/3$ du temps
- **Anniversaires :** $n = 23$ suffit pour $P(\text{collision}) > 50\%$
- **Test médical :** même un bon test a un faible PPV si la prévalence est basse

## 1.4 Simulation Monte Carlo

- **Loi des grands nombres :** $\bar{X}_n \to E[X]$ quand $n \to \infty$
- **Monte Carlo π :** ratio points dans le quart de cercle $\times 4$
- **Convergence :** erreur $\propto 1/\sqrt{n}$ (lente mais universelle)
- **Marche aléatoire :** $S_n = \sum X_i$, $E[S_n] = 0$, $\text{Var}(S_n) = n$

## Programmes associés

| Module | Contenu |
|---|---|
| `combinatorics.py` | Permutations, Pascal, binôme, dérangements |
| `probability_basics.py` | Kolmogorov, Laplace, Bayes, dés/urnes |
| `conditional_probability.py` | Monty Hall, anniversaires, mise à jour bayésienne |
| `random_simulation.py` | LGN, Monte Carlo π, intégration, marches |
