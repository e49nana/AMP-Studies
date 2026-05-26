# Chapitre 2 — Dynamik (Newton)

> **Résumé de révision** — Physik für AMP

## 2.1 Lois de Newton

1. **Inertie :** un corps reste en MRU si $\sum \vec{F} = \vec{0}$
2. **Fondamentale :** $\vec{F} = m\vec{a}$
3. **Action-réaction :** $\vec{F}_{12} = -\vec{F}_{21}$

## 2.2 Forces courantes

| Force | Expression | Direction |
|---|---|---|
| Poids | $P = mg$ | vers le bas |
| Normale | $N$ | perpendiculaire à la surface |
| Frottement statique | $f_s \leq \mu_s N$ | oppose le mouvement potentiel |
| Frottement cinétique | $f_k = \mu_k N$ | oppose le mouvement |
| Tension | $T$ | le long du fil |
| Ressort | $F = -kx$ | vers la position d'équilibre |

## 2.3 Plan incliné

$$a = g(\sin\theta - \mu_k\cos\theta), \qquad \theta_c = \arctan(\mu_s)$$

## 2.4 Machine d'Atwood

$$a = \frac{(m_2 - m_1)g}{m_1 + m_2}, \qquad T = \frac{2m_1 m_2 g}{m_1 + m_2}$$

## 2.5 Oscillations

**Harmonique :** $x'' + \omega_0^2 x = 0 \to x = A\cos(\omega_0 t + \phi)$, $T = 2\pi/\omega_0$

**Amorti :** $x'' + 2\gamma x' + \omega_0^2 x = 0$

| Régime | Condition | Solution |
|---|---|---|
| Sous-amorti | $\gamma < \omega_0$ | $e^{-\gamma t}\cos(\omega_d t)$ |
| Critique | $\gamma = \omega_0$ | $(A + Bt)e^{-\gamma t}$ |
| Sur-amorti | $\gamma > \omega_0$ | $Ae^{r_1 t} + Be^{r_2 t}$ |

**Forcé :** résonance quand $\omega \approx \omega_0$. Amplitude $\propto 1/\gamma$ au pic. Facteur de qualité $Q = \omega_0/(2\gamma)$.

## 2.6 Gravitation

$$F = G\frac{m_1 m_2}{r^2}, \qquad g(r) = \frac{GM}{r^2}$$

| Vitesse | Formule | Signification |
|---|---|---|
| Orbitale $v_1$ | $\sqrt{GM/r}$ | orbite circulaire |
| Libération $v_2$ | $\sqrt{2GM/r} = v_1\sqrt{2}$ | quitter le champ |

**Kepler :** $T^2 \propto a^3$ (vérifié pour toutes les planètes).

## 2.7 Collisions

| Type | Conservation | $e$ |
|---|---|---|
| Élastique | $p$ et $E_{cin}$ | $1$ |
| Inélastique | $p$ seulement | $0$ |
| Partiel | $p$ seulement | $0 < e < 1$ |

Masses égales, choc élastique : échange complet des vitesses (billard).

## Programmes associés

| Module | Contenu |
|---|---|
| `newton_laws.py` | F=ma, plan incliné, Atwood, frottement |
| `oscillations.py` | Harmonique, amorti, forcé, résonance |
| `gravity.py` | Gravitation, orbites, Kepler |
| `collisions.py` | Chocs 1D/2D, restitution |
