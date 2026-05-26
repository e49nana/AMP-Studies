# Chapitre 6 — Wellen und Optik

> **Résumé de révision** — Physik für AMP

## 6.1 Ondes mécaniques

$$y(x,t) = A\sin(kx - \omega t + \varphi)$$

| Grandeur | Relation |
|---|---|
| Nombre d'onde | $k = 2\pi/\lambda$ |
| Pulsation | $\omega = 2\pi f$ |
| Vitesse | $v = \lambda f = \omega/k$ |

**Ondes stationnaires :** $y = 2A\sin(kx)\cos(\omega t)$. Sur une corde de longueur $L$ : $\lambda_n = 2L/n$, $f_n = nv/(2L)$.

**Battements :** superposition de $f_1 \approx f_2$ → enveloppe à $f_{batt} = |f_1 - f_2|$.

**Doppler :** $f_{obs} = f_s \cdot (v_{son} + v_{obs})/(v_{son} + v_s)$. Source qui s'approche → $f$ augmente.

## 6.2 Interférence et diffraction

**Young (2 fentes) :**
$$I = 4I_0\cos^2\left(\frac{\pi d\sin\theta}{\lambda}\right)$$

Maxima : $d\sin\theta = m\lambda$. Interfrange : $\Delta y = \lambda L/d$.

**Diffraction (1 fente) :**
$$I = I_0\left(\frac{\sin\beta}{\beta}\right)^2, \qquad \beta = \frac{\pi a\sin\theta}{\lambda}$$

Premier minimum : $\sin\theta = \lambda/a$.

**Réseau ($N$ fentes) :** pics très fins, $N^2 \times$ plus intenses. Pouvoir de résolution $R = mN$.

## 6.3 Optique géométrique

**Snell-Descartes :** $n_1\sin\theta_1 = n_2\sin\theta_2$

**Angle critique :** $\theta_c = \arcsin(n_2/n_1)$ (réflexion totale si $n_1 > n_2$ et $\theta > \theta_c$).

**Lentille mince :** $\frac{1}{f} = \frac{1}{p} + \frac{1}{q}$, grandissement $\gamma = -q/p$.

| Cas | $p$ vs $f$ | Image |
|---|---|---|
| $p > 2f$ | — | réelle, réduite, renversée |
| $p = 2f$ | — | réelle, taille 1, renversée |
| $f < p < 2f$ | — | réelle, agrandie, renversée |
| $p = f$ | — | à l'infini |
| $p < f$ | — | virtuelle, agrandie, droite |

**Miroir sphérique :** $f = R/2$, même formule de conjugaison.

**Fibre optique :** $ON = \sqrt{n_{coeur}^2 - n_{gaine}^2}$, $\theta_{max} = \arcsin(ON)$.

## Programmes associés

| Module | Contenu |
|---|---|
| `waves.py` | Propagation, stationnaires, battements, Doppler |
| `interference.py` | Young, diffraction, réseau |
| `optics.py` | Snell, lentilles, miroirs, fibres |
