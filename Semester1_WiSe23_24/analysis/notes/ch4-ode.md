# Chapitre 4 — Gewöhnliche Differentialgleichungen

> **Résumé de révision** — Analysis / Mathematik 1 für AMP, S1

## 4.1 Problème de Cauchy

$$y' = f(t, y), \qquad y(t_0) = y_0$$

**Existence et unicité** (Picard-Lindelöf) : si $f$ est Lipschitz en $y$, la solution existe et est unique.

## 4.2 Méthode d'Euler

**Explicite :** $y_{k+1} = y_k + h \cdot f(t_k, y_k)$. Erreur globale $O(h)$.

**Implicite :** $y_{k+1} = y_k + h \cdot f(t_{k+1}, y_{k+1})$. Plus stable pour problèmes raides.

**Stabilité :** pour $y' = \lambda y$, Euler explicite stable ssi $|1 + h\lambda| < 1$, soit $h < 2/|\lambda|$.

## 4.3 Runge-Kutta

| Méthode | Ordre | Évaluations de $f$/pas | Usage |
|---|---|---|---|
| Euler | 1 | 1 | pédagogique |
| Heun (RK2) | 2 | 2 | rapide |
| **RK4** | **4** | **4** | **standard** |

**RK4 classique :**
$$k_1 = f(t, y), \quad k_2 = f(t+h/2, y+hk_1/2)$$
$$k_3 = f(t+h/2, y+hk_2/2), \quad k_4 = f(t+h, y+hk_3)$$
$$y^+ = y + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

## 4.4 Systèmes d'EDO

Toute EDO d'ordre $n$ se ramène à un système d'ordre 1 : $y'' + \omega^2 y = 0 \to \begin{cases} y_1' = y_2 \\ y_2' = -\omega^2 y_1 \end{cases}$

**Classification des points fixes (2D) :**

| $\lambda$ | Type |
|---|---|
| Réels négatifs | nœud stable |
| Réels de signes opposés | selle (instable) |
| Complexes, Re < 0 | spirale stable |
| Imaginaires purs | centre |

## 4.5 Applications physiques

| Système | EDO | Solution / comportement |
|---|---|---|
| Croissance exp. | $y' = ry$ | $y = y_0 e^{rt}$ |
| Logistique | $y' = ry(1-y/K)$ | $y \to K$ (saturation) |
| Pendule (lin.) | $\theta'' + (g/L)\theta = 0$ | $\theta = \theta_0 \cos(\omega t)$ |
| Pendule (non-lin.) | $\theta'' + (g/L)\sin\theta = 0$ | période dépend de $\theta_0$ |
| Circuit RC | $V' = (V_s - V)/(RC)$ | $V = V_s(1 - e^{-t/\tau})$ |

## Programmes associés

| Module | Contenu |
|---|---|
| `ode_euler.py` | Euler explicite/implicite, stabilité |
| `ode_runge_kutta.py` | RK2, RK4, comparaison des ordres |
| `ode_systems.py` | Systèmes, portraits de phase, Lotka-Volterra |
| `ode_applications.py` | Pendule, logistique, circuit RC |
