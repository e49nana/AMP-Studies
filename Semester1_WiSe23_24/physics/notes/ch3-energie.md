# Chapitre 3 — Energie und Arbeit

> **Résumé de révision** — Physik für AMP

## 3.1 Travail et énergie cinétique

$$W = \int_a^b \vec{F} \cdot d\vec{s}, \qquad E_{cin} = \tfrac{1}{2}mv^2$$

**Théorème travail-énergie :** $W_{net} = \Delta E_{cin}$

| Force | Travail |
|---|---|
| Constante | $W = Fd\cos\theta$ |
| Gravité | $W = mgh$ (descente > 0) |
| Ressort | $W = \tfrac{1}{2}k(x_1^2 - x_2^2)$ |

## 3.2 Conservation de l'énergie

$$E_{mec} = E_{cin} + E_{pot} = \text{const.}$$

(Si toutes les forces sont conservatives — pas de frottement.)

**Avec frottement :** $E_{mec,f} = E_{mec,i} - W_{frott}$ ($W_{frott} > 0$ toujours).

## 3.3 Diagramme de potentiel $E_p(x)$

- Mouvement possible là où $E_p(x) \leq E_{total}$
- Points de rebroussement : $E_p = E_{total}$
- Équilibre stable : minimum de $E_p$ ($E_p'' > 0$)
- Équilibre instable : maximum de $E_p$ ($E_p'' < 0$)

**Looping :** hauteur minimale $h = 5R/2$ pour passer un looping de rayon $R$.

## 3.4 Puissance et rendement

$$P = \frac{dW}{dt} = \vec{F} \cdot \vec{v}, \qquad \eta = \frac{P_{utile}}{P_{fournie}}$$

**Machines simples :**

| Machine | Avantage mécanique |
|---|---|
| Levier | $AM = d_{effort}/d_{charge}$ |
| Poulie ($n$ brins) | $AM = n$ |
| Plan incliné | $AM = L/h$ |

Toujours : $W_{fourni} \geq W_{utile}$ (pas de gain d'énergie, seulement de force).

**Cycliste :** $P = P_{aéro} + P_{roulement} + P_{gravité}$ avec $P_{aéro} \propto v^3$ (domine à haute vitesse).

## Programmes associés

| Module | Contenu |
|---|---|
| `work_energy.py` | Travail, théorème W-E, ressort |
| `conservation.py` | Conservation, diagrammes de potentiel, looping |
| `power_efficiency.py` | Puissance, machines simples, cycliste |
