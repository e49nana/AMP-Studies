# Chapitre 4 — Integraltransformationen

> **Résumé de révision** — Angewandte Analysis für AMP, S3-S4

## 4.1 Transformée de Fourier

$$\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t)\,e^{-i\omega t}\,dt, \qquad f(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} \hat{f}(\omega)\,e^{i\omega t}\,d\omega$$

**Propriétés fondamentales :**

| Propriété | Temporel | Fréquentiel |
|---|---|---|
| Linéarité | $af + bg$ | $a\hat{f} + b\hat{g}$ |
| Décalage | $f(t - t_0)$ | $e^{-i\omega t_0}\hat{f}(\omega)$ |
| Modulation | $e^{i\omega_0 t}f(t)$ | $\hat{f}(\omega - \omega_0)$ |
| Convolution | $(f * g)(t)$ | $\hat{f}(\omega) \cdot \hat{g}(\omega)$ |
| Dérivation | $f'(t)$ | $i\omega\hat{f}(\omega)$ |

**Parseval :** $\int|f(t)|^2\,dt = \frac{1}{2\pi}\int|\hat{f}(\omega)|^2\,d\omega$

**Principe d'incertitude :** $\sigma_t \cdot \sigma_\omega \geq \frac{1}{2}$ (on ne peut pas localiser en temps ET en fréquence).

**TF de la gaussienne :** gaussienne → gaussienne (étroite en $t$ → large en $\omega$).

## 4.2 DFT et FFT

$$X[k] = \sum_{n=0}^{N-1} x[n]\,e^{-2\pi i kn/N}$$

| Algorithme | Complexité | Usage |
|---|---|---|
| DFT directe | $O(N^2)$ | pédagogique |
| FFT (Cooley-Tukey) | $O(N\log N)$ | standard ($N = 2^p$) |

**Fréquences :** $f_k = k/(N\Delta t)$. **Nyquist :** $f_{max} = 1/(2\Delta t)$.

**Aliasing :** si $f_{signal} > f_{Nyquist}$, la fréquence apparaît repliée (fausse fréquence).

## 4.3 Transformée de Laplace

$$F(s) = \int_0^{\infty} f(t)\,e^{-st}\,dt, \qquad s \in \mathbb{C}$$

| $f(t)$ | $F(s)$ |
|---|---|
| $1$ | $1/s$ |
| $t^n$ | $n!/s^{n+1}$ |
| $e^{at}$ | $1/(s-a)$ |
| $\sin(\omega t)$ | $\omega/(s^2+\omega^2)$ |
| $\cos(\omega t)$ | $s/(s^2+\omega^2)$ |

**Résolution d'EDO :** $\mathcal{L}\{y'\} = sY(s) - y(0)$ → l'EDO devient algébrique en $s$.

**Fonction de transfert :** $H(s) = Y(s)/U(s)$. **Pôles** = racines du dénominateur. **Stable** ssi $\text{Re}(p_i) < 0$ pour tous les pôles.

## 4.4 Traitement du signal

- **Filtrage fréquentiel :** FFT → masquer les fréquences → IFFT
- **Fenêtrage :** Hann, Hamming, Blackman — réduit le leakage spectral
- **Spectrogramme (STFT) :** représentation temps-fréquence (FFT par fenêtres glissantes)
- **Shannon :** $f_s > 2f_{max}$ pour reconstruire le signal sans aliasing

## Programmes associés

| Module | Contenu |
|---|---|
| `fourier_transform.py` | TF continue, convolution, Parseval |
| `dft_fft.py` | DFT from-scratch, FFT Cooley-Tukey, benchmark |
| `laplace_transform.py` | TL, résolution d'EDO, Bode, pôles |
| `signal_processing.py` | Filtrage, fenêtrage, spectrogramme, aliasing |
