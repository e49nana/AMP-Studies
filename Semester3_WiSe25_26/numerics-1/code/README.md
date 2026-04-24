# Numerische Mathematik 1 — Python Implementations

**Course:** Numerische Mathematik 1 für AMP  
**Semester:** WiSe 2024/2025 (S3)  
**Reference:** Tim Kröger, *Numerische Mathematik 1 für AMP*  
**Author:** Emmanuel Nanan — TH Nürnberg

Every algorithm from the lecture script, implemented from scratch in Python and compared with NumPy/SciPy.

## Structure

Each chapter has its own folder. Modules (`.py`) contain reusable implementations with type hints and French docstrings. Notebooks (`.ipynb`) provide interactive demos with visualizations.

```
code/
├── ch1-grundbegriffe/           # Norms, conditioning, floating-point, cancellation
├── ch2-lineare-gleichungssysteme/  # Gauss, LR, Cholesky, Jacobi, Gauss-Seidel
├── ch3-nichtlineare-gleichungen/   # Newton (scalar + systems), secant, bisection
├── ch4-interpolation/           # Lagrange, Newton, Neville, Runge, splines
├── ch5-ausgleichsrechnung/      # Householder QR, least squares fitting
├── ch6-eigenwertprobleme/       # Power iteration, Wielandt, Jacobi, Gershgorin, QR
├── figures/                     # Generated plots
└── requirements.txt
```

## Programmes (18 modules)

### Chapter 1 — Grundbegriffe der Numerik
| Module | Topics | Script ref. |
|---|---|---|
| `vector_norms.py` | p-norms, unit balls, norm equivalence | §1.2, Übung 1.3, Beispiel 1.6 |
| `cancellation.py` | Catastrophic cancellation, stable reformulations | §1.3.3, §1.4, Beispiel 1.13 |
| `floating_point.py` | IEEE 754, ε_mach, Kahan summation | §1.3.4, Beispiel 1.12 |

### Chapter 2 — Lineare Gleichungssysteme
| Module | Topics | Script ref. |
|---|---|---|
| `matrix_norms.py` | Induced norms, Frobenius, spectral radius, κ(A) | §2.1, §2.2, Satz 2.5–2.17 |
| `gauss_elimination.py` | Gauss with 3 pivot strategies, LR decomposition | §2.3, Übung 2.21 |
| `lu_decomposition.py` | Reusable LR, Nachiteration, Cholesky | §2.3.2, §2.3.7, §2.4 |
| `jacobi_gauss_seidel.py` | Iterative solvers, heat equation 2D | §2.5, Beispiel 2.24 |

### Chapter 3 — Nichtlineare Gleichungen
| Module | Topics | Script ref. |
|---|---|---|
| `newton_scalar.py` | Newton, secant, bisection, convergence order | §3.1, Übung 3.1 |
| `newton_systems.py` | Multivariate Newton with Jacobian | §3.2, Übung 3.10 |
| `derivative_free.py` | Regula Falsi, Illinois, hybrid strategy | §3.1.6, §3.1.7 |

### Chapter 4 — Interpolation
| Module | Topics | Script ref. |
|---|---|---|
| `runge_phenomenon.py` | Lagrange, Newton, Neville, Runge, Chebyshev nodes | §4.2, Satz 4.10 |
| `splines.py` | Natural cubic splines, Thomas algorithm | §4.3, Def. 4.12 |

### Chapter 5 — Lineare Ausgleichsrechnung
| Module | Topics | Script ref. |
|---|---|---|
| `householder_qr.py` | Householder QR, least squares via QR | §5.3, §5.4 |
| `least_squares_fitting.py` | Polynomial, exponential, power law fitting | §5.5.1–5.5.3 |

### Chapter 6 — Eigenwertprobleme
| Module | Topics | Script ref. |
|---|---|---|
| `power_iteration.py` | Von Mises, Wielandt inverse iteration | §6.4, Übung 6.8 |
| `gershgorin_circles.py` | Gershgorin circles visualization | §6.6, Beispiel 6.16–6.18 |
| `jacobi_eigenvalue.py` | Jacobi eigenvalue method (classic + cyclic) | §6.5 |
| `qr_algorithm.py` | QR algorithm with Hessenberg + shift | §6.7 |

## Quick Start

```bash
pip install numpy scipy matplotlib
cd ch1-grundbegriffe
python vector_norms.py          # run any module standalone
jupyter notebook vector_norms_demo.ipynb  # interactive demo
```

## Design Principles

- **From scratch first:** every algorithm is implemented without NumPy linear algebra, then compared with `np.linalg` / `scipy.linalg`
- **Intermediate Python:** classes, dataclasses, type hints, enums — no advanced metaprogramming
- **French docstrings:** to double as revision material
- **Script-aligned:** every Satz, Übung and Beispiel is referenced by number
