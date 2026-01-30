# Numerical Linear Algebra (Numerische Lineare Algebra)

## 📐 Introduction

Solving linear systems Ax = b and eigenvalue problems are fundamental to scientific computing. This document covers direct and iterative methods with conditioning analysis, essential for your Numerik exam.

---

## 🎯 1. Direct Methods for Ax = b

### Gaussian Elimination (Gauß-Elimination)

Transform A to upper triangular form U, then back-substitute.

```
Forward elimination:
A⁽⁰⁾ = A → A⁽¹⁾ → ... → A⁽ⁿ⁻¹⁾ = U

For k = 1, ..., n-1:
    For i = k+1, ..., n:
        lᵢₖ = aᵢₖ⁽ᵏ⁻¹⁾ / aₖₖ⁽ᵏ⁻¹⁾
        aᵢⱼ⁽ᵏ⁾ = aᵢⱼ⁽ᵏ⁻¹⁾ - lᵢₖ · aₖⱼ⁽ᵏ⁻¹⁾
```

### Python Implementation

```python
import numpy as np

def gauss_elimination(A, b):
    """
    Gaussian elimination without pivoting.
    
    Parameters:
        A: Coefficient matrix (n×n)
        b: Right-hand side vector
    
    Returns:
        x: Solution vector
    """
    n = len(b)
    # Augmented matrix
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    
    # Forward elimination
    for k in range(n - 1):
        for i in range(k + 1, n):
            if Ab[k, k] == 0:
                raise ValueError("Zero pivot encountered")
            
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
    
    return x


# Example
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

x = gauss_elimination(A, b)
print(f"Solution: {x}")
print(f"Residual: {np.linalg.norm(A @ x - b):.2e}")
```

### Complexity

- Gaussian elimination: O(n³/3) flops
- Back substitution: O(n²) flops

---

## 🔷 2. LU Decomposition (LU-Zerlegung)

### Concept

Factor A = LU where:
- L = lower triangular with 1s on diagonal
- U = upper triangular

```
A = LU

L = | 1       0    ...  0   |     U = | u₁₁  u₁₂  ...  u₁ₙ |
    | l₂₁    1    ...  0   |         | 0    u₂₂  ...  u₂ₙ |
    | :      :    ...  :   |         | :    :    ...  :   |
    | lₙ₁   lₙ₂   ...  1   |         | 0    0    ...  uₙₙ |
```

### Solving with LU

```
Ax = b  →  LUx = b  →  Ly = b, Ux = y

1. Factor A = LU (once)
2. Forward solve Ly = b: O(n²)
3. Back solve Ux = y: O(n²)
```

### Doolittle Algorithm

```python
def lu_decomposition(A):
    """
    LU decomposition without pivoting (Doolittle).
    
    Returns:
        L: Lower triangular (unit diagonal)
        U: Upper triangular
    """
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    
    for k in range(n):
        # Compute U row k
        for j in range(k, n):
            U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])
        
        # Compute L column k
        for i in range(k + 1, n):
            if U[k, k] == 0:
                raise ValueError("Zero pivot")
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]
    
    return L, U


def solve_lu(L, U, b):
    """Solve LUx = b."""
    n = len(b)
    
    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Back substitution: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x


# Example
A = np.array([[4, 3, 2],
              [2, 5, 3],
              [1, 2, 4]], dtype=float)
b = np.array([9, 10, 7], dtype=float)

L, U = lu_decomposition(A)
x = solve_lu(L, U, b)

print("L =")
print(L)
print("\nU =")
print(U)
print(f"\nSolution: {x}")
print(f"Verify LU = A: {np.allclose(L @ U, A)}")
```

---

## 🔄 3. Pivoting Strategies

### Why Pivoting?

Without pivoting:
- Zero pivots cause failure
- Small pivots cause numerical instability

### Partial Pivoting (Spaltenpivotisierung)

At step k, swap rows to get largest |aᵢₖ| in pivot position.

```
PA = LU

Where P is permutation matrix
```

```python
def lu_partial_pivoting(A):
    """
    LU decomposition with partial pivoting.
    
    Returns:
        P: Permutation matrix
        L: Lower triangular
        U: Upper triangular
        
    Such that PA = LU
    """
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)
    
    for k in range(n - 1):
        # Find pivot
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        
        if U[pivot_row, k] == 0:
            raise ValueError("Matrix is singular")
        
        # Swap rows in U, P, and L
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            # Swap already computed L entries
            L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        
        # Elimination
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return P, L, U


# Example with ill-conditioned case
A = np.array([[1e-20, 1],
              [1, 1]], dtype=float)
b = np.array([1, 2], dtype=float)

# Without pivoting (unstable)
try:
    L1, U1 = lu_decomposition(A)
    x1 = solve_lu(L1, U1, b)
    print(f"Without pivoting: {x1}")
except:
    print("Without pivoting: Failed")

# With pivoting (stable)
P, L, U = lu_partial_pivoting(A)
x = solve_lu(L, U, P @ b)
print(f"With pivoting: {x}")
print(f"Exact: [1, 1]")
```

---

## 📊 4. Cholesky Decomposition

### For Symmetric Positive Definite (SPD) Matrices

```
A = LLᵀ

Where L is lower triangular with positive diagonal
```

### Algorithm

```
lₖₖ = √(aₖₖ - Σⱼ₌₁ᵏ⁻¹ lₖⱼ²)

lᵢₖ = (aᵢₖ - Σⱼ₌₁ᵏ⁻¹ lᵢⱼlₖⱼ) / lₖₖ,  i > k
```

### Python Implementation

```python
def cholesky(A):
    """
    Cholesky decomposition A = LLᵀ.
    
    A must be symmetric positive definite.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    
    for k in range(n):
        # Diagonal element
        sum_sq = np.dot(L[k, :k], L[k, :k])
        val = A[k, k] - sum_sq
        
        if val <= 0:
            raise ValueError("Matrix is not positive definite")
        
        L[k, k] = np.sqrt(val)
        
        # Off-diagonal elements
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], L[k, :k])) / L[k, k]
    
    return L


def solve_cholesky(L, b):
    """Solve Ax = b where A = LLᵀ."""
    # Forward: Ly = b
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    
    # Backward: Lᵀx = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(L[i+1:, i], x[i+1:])) / L[i, i]
    
    return x


# Example: SPD matrix
A = np.array([[4, 2, 2],
              [2, 5, 1],
              [2, 1, 6]], dtype=float)
b = np.array([8, 8, 9], dtype=float)

L = cholesky(A)
x = solve_cholesky(L, b)

print("L =")
print(L)
print(f"\nVerify LLᵀ = A: {np.allclose(L @ L.T, A)}")
print(f"Solution: {x}")
```

### Advantages

- Half the operations of LU: O(n³/6)
- Numerically stable without pivoting
- Guaranteed for SPD matrices

---

## 📐 5. QR Decomposition

### Concept

```
A = QR

Q: Orthogonal (Qᵀ = Q⁻¹)
R: Upper triangular
```

### Methods

1. **Gram-Schmidt** (Classical/Modified)
2. **Householder reflections** (most stable)
3. **Givens rotations** (sparse matrices)

### Modified Gram-Schmidt

```python
def qr_gram_schmidt(A):
    """
    QR decomposition via Modified Gram-Schmidt.
    
    More numerically stable than classical GS.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v -= R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(v)
        
        if R[j, j] < 1e-14:
            raise ValueError("Columns are linearly dependent")
        
        Q[:, j] = v / R[j, j]
    
    return Q, R
```

### Householder Reflections

```
H = I - 2vvᵀ/vᵀv

Reflects vectors across hyperplane orthogonal to v
```

```python
def qr_householder(A):
    """
    QR decomposition via Householder reflections.
    
    Most numerically stable method.
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.astype(float).copy()
    
    for k in range(min(m - 1, n)):
        # Householder vector
        x = R[k:, k]
        v = x.copy()
        v[0] += np.sign(x[0]) * np.linalg.norm(x)
        v = v / np.linalg.norm(v)
        
        # Apply reflection to R
        R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])
        
        # Accumulate Q
        Q_k = np.eye(m)
        Q_k[k:, k:] -= 2 * np.outer(v, v)
        Q = Q @ Q_k
    
    return Q[:, :n], R[:n, :]


# Example
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q, R = qr_householder(A)

print("Q =")
print(Q)
print("\nR =")
print(R)
print(f"\nQᵀQ = I: {np.allclose(Q.T @ Q, np.eye(3))}")
print(f"QR = A: {np.allclose(Q @ R, A)}")
```

### Solving Ax = b with QR

```
Ax = b  →  QRx = b  →  Rx = Qᵀb
```

```python
def solve_qr(A, b):
    """Solve Ax = b via QR decomposition."""
    Q, R = qr_householder(A)
    
    # Rx = Qᵀb
    y = Q.T @ b
    
    # Back substitution
    n = R.shape[1]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    
    return x
```

---

## ⚖️ 6. Conditioning and Error Analysis

### Condition Number

```
κ(A) = ‖A‖ · ‖A⁻¹‖

For 2-norm: κ₂(A) = σₘₐₓ/σₘᵢₙ (ratio of singular values)
```

### Error Bound

```
‖δx‖/‖x‖ ≤ κ(A) · ‖δb‖/‖b‖

Small perturbation in b can cause κ(A) times larger error in x
```

### Python Analysis

```python
def analyze_conditioning(A):
    """Analyze matrix conditioning."""
    # Condition number
    cond_2 = np.linalg.cond(A, 2)
    cond_inf = np.linalg.cond(A, np.inf)
    
    # Singular values
    U, s, Vt = np.linalg.svd(A)
    
    print(f"Condition number (2-norm): {cond_2:.2e}")
    print(f"Condition number (∞-norm): {cond_inf:.2e}")
    print(f"Singular values: {s}")
    print(f"σ_max / σ_min = {s[0]/s[-1]:.2e}")
    
    if cond_2 > 1e10:
        print("WARNING: Matrix is ill-conditioned!")
    
    return cond_2


# Well-conditioned example
A_good = np.array([[4, 1],
                   [1, 3]], dtype=float)
print("Well-conditioned matrix:")
analyze_conditioning(A_good)

print("\n" + "="*50 + "\n")

# Ill-conditioned example (Hilbert matrix)
n = 5
A_bad = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])
print(f"Hilbert matrix ({n}x{n}):")
analyze_conditioning(A_bad)
```

### Residual vs Error

```
Residual: r = b - Ax̂  (computable)
Error: e = x - x̂      (unknown)

‖e‖ ≤ κ(A) · ‖r‖/‖A‖

Small residual ≠ small error if κ(A) is large!
```

---

## 🔄 7. Iterative Methods

### Jacobi Method

```
x⁽ᵏ⁺¹⁾ᵢ = (bᵢ - Σⱼ≠ᵢ aᵢⱼxⱼ⁽ᵏ⁾) / aᵢᵢ

Matrix form: x⁽ᵏ⁺¹⁾ = D⁻¹(b - (L+U)x⁽ᵏ⁾)
```

### Gauss-Seidel Method

```
x⁽ᵏ⁺¹⁾ᵢ = (bᵢ - Σⱼ<ᵢ aᵢⱼxⱼ⁽ᵏ⁺¹⁾ - Σⱼ>ᵢ aᵢⱼxⱼ⁽ᵏ⁾) / aᵢᵢ

Uses updated values immediately!
```

### SOR (Successive Over-Relaxation)

```
x⁽ᵏ⁺¹⁾ᵢ = (1-ω)x⁽ᵏ⁾ᵢ + ω/aᵢᵢ · (bᵢ - Σⱼ<ᵢ aᵢⱼxⱼ⁽ᵏ⁺¹⁾ - Σⱼ>ᵢ aᵢⱼxⱼ⁽ᵏ⁾)

ω ∈ (0, 2): relaxation parameter
ω = 1: Gauss-Seidel
ω > 1: over-relaxation (faster for some problems)
```

### Python Implementation

```python
def jacobi(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Jacobi iterative method.
    
    Convergence: Requires A to be strictly diagonally dominant
                 or spectral radius ρ(D⁻¹(L+U)) < 1
    """
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)
    x_new = np.zeros(n)
    
    D = np.diag(A)
    R = A - np.diag(D)  # L + U
    
    history = [np.linalg.norm(A @ x - b)]
    
    for k in range(max_iter):
        x_new = (b - R @ x) / D
        
        residual = np.linalg.norm(A @ x_new - b)
        history.append(residual)
        
        if residual < tol:
            return x_new, k + 1, history
        
        x = x_new.copy()
    
    return x, max_iter, history


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Gauss-Seidel iterative method.
    
    Usually converges faster than Jacobi.
    """
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)
    
    history = [np.linalg.norm(A @ x - b)]
    
    for k in range(max_iter):
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]
        
        residual = np.linalg.norm(A @ x - b)
        history.append(residual)
        
        if residual < tol:
            return x, k + 1, history
    
    return x, max_iter, history


def sor(A, b, omega=1.5, x0=None, tol=1e-10, max_iter=1000):
    """
    Successive Over-Relaxation (SOR).
    
    omega = 1: Gauss-Seidel
    omega in (1, 2): Over-relaxation
    """
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)
    
    history = [np.linalg.norm(A @ x - b)]
    
    for k in range(max_iter):
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
        
        residual = np.linalg.norm(A @ x - b)
        history.append(residual)
        
        if residual < tol:
            return x, k + 1, history
    
    return x, max_iter, history


# Compare methods
def compare_iterative():
    """Compare convergence of iterative methods."""
    # Diagonally dominant matrix
    n = 50
    A = np.diag(4 * np.ones(n)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
    b = np.ones(n)
    
    x_jac, iter_jac, hist_jac = jacobi(A, b, tol=1e-10)
    x_gs, iter_gs, hist_gs = gauss_seidel(A, b, tol=1e-10)
    x_sor, iter_sor, hist_sor = sor(A, b, omega=1.5, tol=1e-10)
    
    print(f"Jacobi: {iter_jac} iterations")
    print(f"Gauss-Seidel: {iter_gs} iterations")
    print(f"SOR (ω=1.5): {iter_sor} iterations")
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(hist_jac, 'b-', label=f'Jacobi ({iter_jac} iter)')
    plt.semilogy(hist_gs, 'r-', label=f'Gauss-Seidel ({iter_gs} iter)')
    plt.semilogy(hist_sor, 'g-', label=f'SOR ω=1.5 ({iter_sor} iter)')
    plt.xlabel('Iteration')
    plt.ylabel('Residual ‖Ax - b‖')
    plt.title('Iterative Methods Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('iterative_convergence.png', dpi=150)
    plt.show()


compare_iterative()
```

### Convergence Conditions

| Method | Converges if |
|--------|--------------|
| Jacobi | A strictly diagonally dominant |
| Gauss-Seidel | A symmetric positive definite OR strictly diagonally dominant |
| SOR | A SPD and 0 < ω < 2 |

---

## 🎯 8. Eigenvalue Problems

### Power Method

Find dominant eigenvalue (largest |λ|):

```
v⁽ᵏ⁺¹⁾ = Av⁽ᵏ⁾ / ‖Av⁽ᵏ⁾‖
λ ≈ (v⁽ᵏ⁾)ᵀAv⁽ᵏ⁾ / (v⁽ᵏ⁾)ᵀv⁽ᵏ⁾  (Rayleigh quotient)
```

```python
def power_method(A, tol=1e-10, max_iter=1000):
    """
    Power method for dominant eigenvalue.
    
    Returns:
        eigenvalue: Dominant eigenvalue
        eigenvector: Corresponding eigenvector
        iterations: Number of iterations
    """
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    lambda_old = 0
    
    for k in range(max_iter):
        w = A @ v
        v = w / np.linalg.norm(w)
        
        # Rayleigh quotient
        lambda_new = v @ A @ v
        
        if abs(lambda_new - lambda_old) < tol:
            return lambda_new, v, k + 1
        
        lambda_old = lambda_new
    
    return lambda_new, v, max_iter


# Example
A = np.array([[4, 1, 1],
              [1, 3, 1],
              [1, 1, 2]], dtype=float)

eigenvalue, eigenvector, iters = power_method(A)

print(f"Dominant eigenvalue: {eigenvalue:.10f}")
print(f"Eigenvector: {eigenvector}")
print(f"Iterations: {iters}")

# Verify
eigenvalues_np = np.linalg.eigvals(A)
print(f"\nAll eigenvalues (numpy): {sorted(eigenvalues_np, reverse=True)}")
```

### Inverse Power Method

Find smallest eigenvalue:

```python
def inverse_power_method(A, tol=1e-10, max_iter=1000):
    """
    Inverse power method for smallest eigenvalue.
    """
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    # LU factorization for efficiency
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A)
    
    lambda_old = 0
    
    for k in range(max_iter):
        w = lu_solve((lu, piv), v)  # Solve Aw = v
        v = w / np.linalg.norm(w)
        
        # Rayleigh quotient gives 1/λ_min
        lambda_new = v @ A @ v
        
        if abs(lambda_new - lambda_old) < tol:
            return lambda_new, v, k + 1
        
        lambda_old = lambda_new
    
    return lambda_new, v, max_iter
```

### Shifted Inverse Power

Find eigenvalue closest to μ:

```python
def shifted_inverse_power(A, mu, tol=1e-10, max_iter=1000):
    """
    Shifted inverse power method.
    
    Finds eigenvalue closest to shift μ.
    """
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    A_shifted = A - mu * np.eye(n)
    
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A_shifted)
    
    lambda_old = 0
    
    for k in range(max_iter):
        w = lu_solve((lu, piv), v)
        v = w / np.linalg.norm(w)
        
        lambda_new = v @ A @ v
        
        if abs(lambda_new - lambda_old) < tol:
            return lambda_new, v, k + 1
        
        lambda_old = lambda_new
    
    return lambda_new, v, max_iter


# Find eigenvalue closest to 3
eigenvalue, eigenvector, iters = shifted_inverse_power(A, mu=3.0)
print(f"Eigenvalue closest to 3: {eigenvalue:.10f}")
```

### QR Algorithm (Brief)

```
A₀ = A
For k = 0, 1, 2, ...:
    Qₖ, Rₖ = QR(Aₖ)
    Aₖ₊₁ = RₖQₖ

Aₖ → diagonal (or quasi-upper triangular)
Diagonal entries → eigenvalues
```

```python
def qr_algorithm(A, max_iter=100, tol=1e-10):
    """
    Basic QR algorithm for eigenvalues.
    """
    n = A.shape[0]
    Ak = A.astype(float).copy()
    
    for k in range(max_iter):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
        
        # Check convergence (off-diagonal elements)
        off_diag = np.sum(np.abs(np.tril(Ak, -1)))
        if off_diag < tol:
            break
    
    return np.diag(Ak), k + 1


eigenvalues, iters = qr_algorithm(A)
print(f"Eigenvalues (QR algorithm): {sorted(eigenvalues, reverse=True)}")
print(f"Iterations: {iters}")
```

---

## 📋 9. Method Summary

### Direct Methods

| Method | Complexity | Stability | Best For |
|--------|------------|-----------|----------|
| Gauss | O(n³/3) | Needs pivoting | General |
| LU | O(n³/3) | Needs pivoting | Multiple RHS |
| Cholesky | O(n³/6) | Stable | SPD matrices |
| QR | O(2n³/3) | Very stable | Least squares |

### Iterative Methods

| Method | Convergence | Per Iteration | Best For |
|--------|-------------|---------------|----------|
| Jacobi | Slow | O(n²) | Parallelizable |
| Gauss-Seidel | Medium | O(n²) | General sparse |
| SOR | Fast (with good ω) | O(n²) | Large sparse |
| CG | Fast for SPD | O(n²) | Large SPD |

---

## 📋 10. Exam Checklist (Klausur)

### Formulas to Know

- [ ] LU decomposition algorithm
- [ ] Cholesky: lₖₖ = √(aₖₖ - Σlₖⱼ²)
- [ ] Jacobi iteration formula
- [ ] Gauss-Seidel iteration formula
- [ ] Condition number: κ(A) = ‖A‖·‖A⁻¹‖
- [ ] Power method iteration

### Key Concepts

- [ ] When to use LU vs Cholesky vs QR
- [ ] Partial pivoting and why it's needed
- [ ] Conditioning and error amplification
- [ ] Convergence criteria for iterative methods
- [ ] Spectral radius and convergence rate

### Common Exam Tasks

- [ ] Perform LU decomposition by hand (3×3)
- [ ] One iteration of Jacobi/Gauss-Seidel
- [ ] Check diagonal dominance
- [ ] Calculate condition number
- [ ] Power method iteration

---

## 🔗 Related Documents

- [01-root-finding.md](./01-root-finding.md) - Root finding methods
- [02-interpolation.md](./02-interpolation.md) - Polynomial interpolation
- [03-integration.md](./03-integration.md) - Numerical integration
- [04-ode-solvers.md](./04-ode-solvers.md) - ODE solving methods

---

## 📚 References

- Stoer & Bulirsch, "Numerische Mathematik 1", Kapitel 4
- Golub & Van Loan, "Matrix Computations"
- Trefethen & Bau, "Numerical Linear Algebra"

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: January 28, 2026*
