# Chapitre 3 — Stetige Verteilungen

> **Résumé de révision** — Stochastik für AMP, S3

## 3.1 Distributions continues

| Distribution | Densité $f(x)$ | $E[X]$ | $\text{Var}(X)$ |
|---|---|---|---|
| Uniforme $U(a,b)$ | $1/(b-a)$ sur $[a,b]$ | $(a+b)/2$ | $(b-a)^2/12$ |
| Exponentielle $\text{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$ pour $x \geq 0$ | $1/\lambda$ | $1/\lambda^2$ |
| Normale $N(\mu,\sigma^2)$ | $\frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu$ | $\sigma^2$ |

**Exponentielle :** seule distribution continue **sans mémoire** : $P(X > t+s | X > t) = P(X > s)$

## 3.2 Loi normale

**Standardisation :** $Z = (X - \mu)/\sigma \sim N(0,1)$

**Règle 68-95-99.7 :**
- $P(|Z| < 1) \approx 68.3\%$
- $P(|Z| < 2) \approx 95.4\%$
- $P(|Z| < 3) \approx 99.7\%$

**CDF :** $\Phi(x) = \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$ (pas de formule fermée)

**Somme :** $X \sim N(\mu_1,\sigma_1^2), Y \sim N(\mu_2,\sigma_2^2)$ indép. $\Rightarrow X+Y \sim N(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$

**Approx. de la binomiale :** $B(n,p) \approx N(np, np(1-p))$ pour $n$ grand. Correction de continuité : $P(X \leq k) \approx \Phi((k+0.5-np)/\sqrt{np(1-p)})$.

## 3.3 Théorème central limite (TCL)

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0,1)$$

Quelle que soit la distribution de départ (pourvu que $\sigma < \infty$). C'est **la** raison pour laquelle la normale est omniprésente.

**Application :** IC pour la moyenne $\bar{X} \pm z_{\alpha/2} \cdot \sigma/\sqrt{n}$.

## 3.4 Ajustement de distributions

- **QQ-plot :** quantiles observés vs théoriques — droite = bon ajustement
- **Test KS :** $D_n = \sup|F_n(x) - F(x)|$, compare ECDF à CDF théorique
- **Test $\chi^2$ :** $\sum (O_i - E_i)^2/E_i$, compare fréquences observées/attendues
- **MLE :** $\hat{\theta} = \arg\max \prod f(x_i|\theta)$

## Programmes associés

| Module | Contenu |
|---|---|
| `continuous_distributions.py` | Uniforme, exponentielle, normale, 68-95-99.7 |
| `normal_distribution.py` | Φ from-scratch, table, IC, approx. binomiale |
| `central_limit_theorem.py` | TCL pour 6 distributions, vitesse de convergence |
| `distribution_fitting.py` | QQ-plot, KS, χ², MLE, comparaison |
