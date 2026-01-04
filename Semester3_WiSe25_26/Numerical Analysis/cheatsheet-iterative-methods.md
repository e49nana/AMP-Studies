# Iterative Methods — Quick Reference  

## Jacobi Method  

**Update rule**  

\[
x_i^{(k+1)} = \frac{1}{A_{ii}}
\left(b_i - \sum_{j \neq i} A_{ij} x_j^{(k)}\right)
\]

**Convergence**  
Guaranteed if \(A\) is **strictly diagonally dominant**.

```python
def jacobi(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()

    for _ in range(max_iter):
        x_new = np.zeros(n)

        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        if np.linalg.norm(x_new - x) < tol:
            return x_new

        x = x_new

    return x
```

---

## Gauss–Seidel Method  

**Update rule**  

\[
x_i^{(k+1)} =
\frac{1}{A_{ii}}
\left(
b_i
- \sum_{j < i} A_{ij} x_j^{(k+1)}
- \sum_{j > i} A_{ij} x_j^{(k)}
\right)
\]

```python
def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()

    for _ in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x - x_old) < tol:
            return x

    return x
```

---

## SOR (Successive Over-Relaxation)  

```python
def sor(A, b, x0, omega, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()

    for _ in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x_gs = (b[i] - s1 - s2) / A[i, i]
            x[i] = (1 - omega) * x_old[i] + omega * x_gs

        if np.linalg.norm(x - x_old) < tol:
            return x

    return x
```

---

## NumPy Vectorized Versions  

### Jacobi (fully vectorized)

```python
def jacobi_vec(A, b, x0, tol=1e-6, max_iter=100):
    A = np.asarray(A)
    b = np.asarray(b)
    x = np.asarray(x0, dtype=float).copy()

    D = np.diag(A)
    R = A - np.diagflat(D)

    for _ in range(max_iter):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new

    return x
```

### Gauss–Seidel (NumPy-friendly)

```python
def gauss_seidel_np(A, b, x0, tol=1e-6, max_iter=100):
    A = np.asarray(A)
    b = np.asarray(b)
    x = np.asarray(x0, dtype=float).copy()

    n = b.size
    for _ in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            s1 = A[i, :i] @ x[:i]
            s2 = A[i, i+1:] @ x_old[i+1:]
            x[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x - x_old) < tol:
            return x

    return x
```

### SOR (NumPy-friendly)

```python
def sor_np(A, b, x0, omega, tol=1e-6, max_iter=100):
    A = np.asarray(A)
    b = np.asarray(b)
    x = np.asarray(x0, dtype=float).copy()

    n = b.size
    for _ in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            s1 = A[i, :i] @ x[:i]
            s2 = A[i, i+1:] @ x_old[i+1:]
            x_gs = (b[i] - s1 - s2) / A[i, i]
            x[i] = (1 - omega) * x_old[i] + omega * x_gs

        if np.linalg.norm(x - x_old) < tol:
            return x

    return x
```
