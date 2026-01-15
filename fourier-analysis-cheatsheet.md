# Fourier Analysis — Quick Reference

> *"Fourier's theorem is not only one of the most beautiful results of modern analysis, but it is also one of the most useful."* — Lord Kelvin

---

## 🌊 Fourier Series

Any periodic function f(x) with period 2L can be represented as:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\left(\frac{n\pi x}{L}\right) + b_n \sin\left(\frac{n\pi x}{L}\right) \right]$$

### Coefficients

$$a_0 = \frac{1}{L} \int_{-L}^{L} f(x) \, dx$$

$$a_n = \frac{1}{L} \int_{-L}^{L} f(x) \cos\left(\frac{n\pi x}{L}\right) dx$$

$$b_n = \frac{1}{L} \int_{-L}^{L} f(x) \sin\left(\frac{n\pi x}{L}\right) dx$$

### Complex Form (Exponential)

$$f(x) = \sum_{n=-\infty}^{\infty} c_n e^{in\pi x/L}$$

where:

$$c_n = \frac{1}{2L} \int_{-L}^{L} f(x) e^{-in\pi x/L} dx$$

---

## 🔄 Fourier Transform

For non-periodic functions, we use the continuous Fourier Transform:

### Forward Transform
$$\hat{f}(\omega) = \mathcal{F}\{f(t)\} = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt$$

### Inverse Transform
$$f(t) = \mathcal{F}^{-1}\{\hat{f}(\omega)\} = \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(\omega) e^{i\omega t} d\omega$$

---

## 📊 Common Transform Pairs

| f(t) | F(ω) |
|------|------|
| δ(t) | 1 |
| 1 | 2πδ(ω) |
| e^{-at}u(t), a>0 | 1/(a + iω) |
| e^{-a\|t\|} | 2a/(a² + ω²) |
| rect(t/τ) | τ sinc(ωτ/2) |
| sinc(at) | (π/a) rect(ω/2a) |
| cos(ω₀t) | π[δ(ω-ω₀) + δ(ω+ω₀)] |
| sin(ω₀t) | (π/i)[δ(ω-ω₀) - δ(ω+ω₀)] |
| e^{-t²/2} | √(2π) e^{-ω²/2} |

---

## ⚡ Key Properties

| Property | Time Domain | Frequency Domain |
|----------|-------------|------------------|
| **Linearity** | af(t) + bg(t) | aF(ω) + bG(ω) |
| **Time Shift** | f(t - t₀) | F(ω)e^{-iωt₀} |
| **Freq Shift** | f(t)e^{iω₀t} | F(ω - ω₀) |
| **Scaling** | f(at) | (1/\|a\|)F(ω/a) |
| **Derivative** | f'(t) | iωF(ω) |
| **Convolution** | f * g | F(ω)G(ω) |
| **Multiplication** | f(t)g(t) | (1/2π)(F * G) |
| **Parseval** | ∫\|f(t)\|²dt | (1/2π)∫\|F(ω)\|²dω |

---

## 💻 DFT (Discrete Fourier Transform)

For sampled signals with N points:

### Forward DFT
$$X_k = \sum_{n=0}^{N-1} x_n e^{-i2\pi kn/N}, \quad k = 0, 1, ..., N-1$$

### Inverse DFT
$$x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{i2\pi kn/N}, \quad n = 0, 1, ..., N-1$$

### FFT Complexity
- DFT: O(N²)
- FFT: O(N log N) ← **Much faster!**

---

## 🐍 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def fourier_series_coefficients(f, L, N_terms):
    """
    Compute Fourier series coefficients numerically.
    
    Parameters
    ----------
    f : callable - Function to approximate
    L : float - Half-period (function has period 2L)
    N_terms : int - Number of terms
    
    Returns
    -------
    a0, an, bn : coefficients
    """
    x = np.linspace(-L, L, 1000)
    dx = x[1] - x[0]
    
    # a0
    a0 = (1/L) * np.trapz(f(x), x)
    
    # an, bn
    an = np.zeros(N_terms)
    bn = np.zeros(N_terms)
    
    for n in range(1, N_terms + 1):
        an[n-1] = (1/L) * np.trapz(f(x) * np.cos(n * np.pi * x / L), x)
        bn[n-1] = (1/L) * np.trapz(f(x) * np.sin(n * np.pi * x / L), x)
    
    return a0, an, bn


def reconstruct_fourier(x, L, a0, an, bn):
    """Reconstruct function from Fourier coefficients."""
    result = a0 / 2
    for n in range(1, len(an) + 1):
        result += an[n-1] * np.cos(n * np.pi * x / L)
        result += bn[n-1] * np.sin(n * np.pi * x / L)
    return result


# Example: Square wave
def square_wave(x):
    return np.sign(np.sin(x))

# Compute coefficients
L = np.pi
a0, an, bn = fourier_series_coefficients(square_wave, L, N_terms=20)

# Plot
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y_original = square_wave(x)
y_approx = reconstruct_fourier(x, L, a0, an, bn)

plt.figure(figsize=(10, 6))
plt.plot(x, y_original, 'b-', label='Original', alpha=0.5)
plt.plot(x, y_approx, 'r-', label=f'Fourier ({len(an)} terms)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Fourier Series Approximation of Square Wave')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('fourier_square_wave.png', dpi=150)
plt.show()
```

---

## 🎵 FFT with NumPy

```python
import numpy as np
import matplotlib.pyplot as plt

# Sample signal: sum of two frequencies
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# Compute FFT
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1/fs)

# Plot magnitude spectrum (positive frequencies only)
positive_mask = frequencies >= 0
plt.figure(figsize=(10, 4))
plt.plot(frequencies[positive_mask], np.abs(fft_result[positive_mask]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Spectrum')
plt.xlim(0, 200)
plt.grid(True, alpha=0.3)
plt.savefig('fft_spectrum.png', dpi=150)
plt.show()
```

---

## 🎓 Applications

| Domain | Application |
|--------|-------------|
| **Signal Processing** | Audio filtering, noise removal |
| **Image Processing** | Compression (JPEG), edge detection |
| **Physics** | Wave analysis, quantum mechanics |
| **Trading** | Cycle detection, spectral analysis |
| **PDEs** | Heat equation, wave equation solutions |

---

## 🔗 Connections to Other Topics

- **Funktionale Analysis** — Fourier series in L² space, Hilbert basis
- **Numerik** — FFT algorithms, spectral methods
- **Physik** — Wave equations, quantum mechanics
- **Trading** — Fourier analysis for cycle detection

---

## 📚 Key Theorems

### Parseval's Theorem (Energy Conservation)
$$\int_{-\infty}^{\infty} |f(t)|^2 dt = \frac{1}{2\pi} \int_{-\infty}^{\infty} |\hat{f}(\omega)|^2 d\omega$$

### Convolution Theorem
$$\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$$

### Uncertainty Principle
$$\Delta t \cdot \Delta \omega \geq \frac{1}{2}$$

*You cannot simultaneously localize a signal in both time and frequency.*

---

*AMP-Studies — Funktionale Analysis / Numerik*  
*TH Nürnberg — WiSe 25/26*  

---

### 🎂 Commit spécial

```
Created on: January 15, 2026
A birthday gift to myself: knowledge that lasts forever.
```
