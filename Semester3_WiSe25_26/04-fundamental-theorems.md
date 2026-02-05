# Fundamental Theorems (Fundamentalsätze)

## 📐 Introduction

The four pillars of functional analysis are the Hahn-Banach theorem, the Uniform Boundedness Principle, the Open Mapping theorem, and the Closed Graph theorem. These powerful results have far-reaching consequences and are essential for your Funktionale Analysis exam.

---

## 🎯 1. Hahn-Banach Theorem (Satz von Hahn-Banach)

### Extension Form

```
Let X be a real vector space, p: X → ℝ a sublinear functional:
  p(x + y) ≤ p(x) + p(y)
  p(αx) = αp(x) for α ≥ 0

Let U ⊆ X be a subspace and f: U → ℝ linear with f(u) ≤ p(u) ∀u ∈ U.

Then there exists F: X → ℝ linear such that:
  1. F|_U = f  (F extends f)
  2. F(x) ≤ p(x) ∀x ∈ X
```

### Normed Space Version

```
Let X be a normed space, U ⊆ X a subspace, f ∈ U*.

Then there exists F ∈ X* such that:
  1. F|_U = f
  2. ‖F‖_{X*} = ‖f‖_{U*}

"Every bounded functional on a subspace extends to the whole space 
without increasing the norm."
```

### Python Illustration

```python
import numpy as np
import matplotlib.pyplot as plt

def hahn_banach_illustration():
    """Illustrate Hahn-Banach extension."""
    
    print("=== Hahn-Banach Theorem ===\n")
    
    # Example: X = ℝ², U = {(x, 0) : x ∈ ℝ} (x-axis)
    # f: U → ℝ, f(x, 0) = 2x
    # Extend to F: ℝ² → ℝ
    
    print("X = ℝ², U = x-axis = {(x, 0)}")
    print("f(x, 0) = 2x on U")
    print("‖f‖ = sup{|f(u)|/‖u‖} = sup{|2x|/|x|} = 2")
    
    print("\nPossible extensions F(x, y) = 2x + cy:")
    print("‖F‖ = sup{|2x + cy|/√(x² + y²)}")
    
    # For extension to preserve norm, need |c| ≤ 2... actually more complex
    # The Hahn-Banach extension is not unique in general
    
    print("\nExtension F(x, y) = 2x preserves norm:")
    print("  ‖F‖ = 2 = ‖f‖ ✓")
    
    print("\nNote: Extension is generally NOT unique!")


def separation_theorem():
    """Geometric form: separation of convex sets."""
    
    print("\n" + "="*50)
    print("\n=== Geometric Hahn-Banach (Separation) ===\n")
    
    print("Let C be a closed convex set, x₀ ∉ C.")
    print("Then ∃f ∈ X* and α ∈ ℝ such that:")
    print("  f(c) ≤ α < f(x₀)  ∀c ∈ C")
    print("\n'A hyperplane separates x₀ from C'")
    
    # Visualize in 2D
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convex set (circle)
    theta = np.linspace(0, 2*np.pi, 100)
    C_x = np.cos(theta)
    C_y = np.sin(theta)
    ax.fill(C_x, C_y, alpha=0.3, color='blue', label='Convex set C')
    ax.plot(C_x, C_y, 'b-', linewidth=2)
    
    # Point outside
    x0 = np.array([2, 1])
    ax.scatter([x0[0]], [x0[1]], color='red', s=100, zorder=5, label='x₀ ∉ C')
    
    # Separating hyperplane (line in 2D)
    # Normal direction from center to x0
    direction = x0 / np.linalg.norm(x0)
    
    # Hyperplane at distance 1 (boundary of C)
    t = np.linspace(-2, 2, 100)
    perp = np.array([-direction[1], direction[0]])
    hyperplane = direction * 1.0 + np.outer(t, perp)
    
    ax.plot(hyperplane[:, 0], hyperplane[:, 1], 'g--', linewidth=2, 
            label='Separating hyperplane')
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Geometric Hahn-Banach: Separation Theorem')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.savefig('separation_theorem.png', dpi=150)
    plt.show()


hahn_banach_illustration()
separation_theorem()
```

### Consequences

```
1. X* separates points: x ≠ y ⟹ ∃f ∈ X*: f(x) ≠ f(y)

2. For x ∈ X: ‖x‖ = max{|f(x)| : f ∈ X*, ‖f‖ ≤ 1}

3. Existence of supporting functionals:
   ∀x ≠ 0 ∃f ∈ X*: ‖f‖ = 1 and f(x) = ‖x‖
```

---

## ⚡ 2. Uniform Boundedness Principle (Satz von Banach-Steinhaus)

### Theorem

```
Let X be a Banach space, Y a normed space.
Let {Tₐ}_{α∈A} ⊆ B(X, Y) be a family of bounded operators.

If  sup_α ‖Tₐx‖ < ∞  for all x ∈ X  (pointwise bounded)
Then  sup_α ‖Tₐ‖ < ∞  (uniformly bounded)
```

### Contrapositive (Resonance Theorem)

```
If sup_α ‖Tₐ‖ = ∞, then ∃x ∈ X such that sup_α ‖Tₐx‖ = ∞.

"Unboundedness must occur on a dense set"
```

### Python Illustration

```python
def uniform_boundedness_demo():
    """Demonstrate Uniform Boundedness Principle."""
    
    print("=== Uniform Boundedness Principle ===\n")
    
    print("If {Tₐ} is pointwise bounded on Banach space X,")
    print("then {Tₐ} is uniformly bounded in operator norm.\n")
    
    # Example: Partial sum operators for Fourier series
    print("Example: Fourier partial sums Sₙ on C[-π, π]")
    print("-" * 50)
    
    # Sₙf(x) = Σₖ₌₋ₙⁿ ĉₖ eⁱᵏˣ
    # ‖Sₙ‖ = Lₙ (Lebesgue constant) → ∞
    
    def lebesgue_constant(n):
        """Approximate Lebesgue constant."""
        # Lₙ ≈ (4/π²) log(n) for large n
        return (4/np.pi**2) * np.log(n + 1) + 1
    
    print("\nLebesgue constants (operator norms of Sₙ):")
    for n in [1, 10, 100, 1000]:
        Ln = lebesgue_constant(n)
        print(f"  L_{n} ≈ {Ln:.4f}")
    
    print("\n‖Sₙ‖ → ∞ as n → ∞")
    print("\nBy UBP contrapositive:")
    print("∃f ∈ C[-π, π] such that Sₙf does NOT converge uniformly!")
    print("(This is du Bois-Reymond's theorem)")
    
    # Visualization
    n_values = np.arange(1, 101)
    L_values = [lebesgue_constant(n) for n in n_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, L_values, 'b-', linewidth=2)
    plt.plot(n_values, (4/np.pi**2) * np.log(n_values + 1), 'r--', 
             linewidth=1.5, label='(4/π²)log(n)')
    plt.xlabel('n')
    plt.ylabel('Lₙ')
    plt.title('Lebesgue Constants: ‖Sₙ‖ → ∞')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lebesgue_constants.png', dpi=150)
    plt.show()


def ubp_application():
    """Application: Convergence of operators."""
    
    print("\n" + "="*50)
    print("\n=== Application: Tₙ → T pointwise ===\n")
    
    print("If Tₙ → T pointwise and X is Banach, then:")
    print("  1. sup_n ‖Tₙ‖ < ∞ (by UBP)")
    print("  2. T is bounded with ‖T‖ ≤ liminf ‖Tₙ‖")
    
    # Example: Matrix sequence
    print("\nExample: Aₙ = (1/n)I in ℝ²")
    print("Aₙ → 0 pointwise")
    print("‖Aₙ‖ = 1/n → 0")
    print("sup_n ‖Aₙ‖ = 1 < ∞ ✓")


uniform_boundedness_demo()
ubp_application()
```

---

## 🚪 3. Open Mapping Theorem (Satz von der offenen Abbildung)

### Theorem

```
Let X, Y be Banach spaces and T ∈ B(X, Y) surjective.
Then T is an open map: T(U) is open for every open U ⊆ X.
```

### Equivalent Formulation

```
∃δ > 0: B_Y(0, δ) ⊆ T(B_X(0, 1))

"The image of the unit ball contains a ball"
```

### Bounded Inverse Theorem (Corollary)

```
If T ∈ B(X, Y) is bijective (X, Y Banach), then T⁻¹ ∈ B(Y, X).

"Continuous bijection has continuous inverse"
```

### Python Illustration

```python
def open_mapping_demo():
    """Illustrate Open Mapping Theorem."""
    
    print("=== Open Mapping Theorem ===\n")
    
    print("If T: X → Y is bounded, linear, and SURJECTIVE")
    print("(X, Y Banach), then T maps open sets to open sets.\n")
    
    # Example: T: ℝ² → ℝ², T(x,y) = (2x+y, x+y) (invertible)
    T = np.array([[2, 1],
                  [1, 1]])
    
    print(f"T = \n{T}\n")
    print(f"det(T) = {np.linalg.det(T):.1f} ≠ 0 (bijective)")
    
    # Unit ball maps to...
    theta = np.linspace(0, 2*np.pi, 100)
    unit_ball = np.array([np.cos(theta), np.sin(theta)])
    image = T @ unit_ball
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original unit ball
    axes[0].fill(unit_ball[0], unit_ball[1], alpha=0.3, color='blue')
    axes[0].plot(unit_ball[0], unit_ball[1], 'b-', linewidth=2)
    axes[0].set_title('Unit Ball B(0,1) in X')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    
    # Image
    axes[1].fill(image[0], image[1], alpha=0.3, color='red')
    axes[1].plot(image[0], image[1], 'r-', linewidth=2)
    
    # Show that image contains a ball
    # Smallest singular value gives the radius
    _, s, _ = np.linalg.svd(T)
    delta = s[-1]  # Smallest singular value
    
    inner_ball = delta * unit_ball
    axes[1].plot(inner_ball[0], inner_ball[1], 'g--', linewidth=2,
                 label=f'B(0, δ), δ = {delta:.2f}')
    
    axes[1].set_title('T(B(0,1)) contains B(0, δ)')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-3, 3)
    
    plt.suptitle('Open Mapping Theorem: Image contains a ball', fontsize=14)
    plt.tight_layout()
    plt.savefig('open_mapping.png', dpi=150)
    plt.show()
    
    print(f"\nSmallest singular value σ_min = {delta:.4f}")
    print(f"B(0, {delta:.4f}) ⊆ T(B(0, 1)) ✓")


def bounded_inverse_demo():
    """Bounded Inverse Theorem."""
    
    print("\n" + "="*50)
    print("\n=== Bounded Inverse Theorem ===\n")
    
    print("If T: X → Y is bounded, linear, and BIJECTIVE")
    print("(X, Y Banach), then T⁻¹ is also bounded.\n")
    
    T = np.array([[2, 1],
                  [1, 1]])
    T_inv = np.linalg.inv(T)
    
    print(f"T = \n{T}")
    print(f"\nT⁻¹ = \n{T_inv}")
    print(f"\n‖T‖ = {np.linalg.norm(T, 2):.4f}")
    print(f"‖T⁻¹‖ = {np.linalg.norm(T_inv, 2):.4f}")
    
    # Condition number
    kappa = np.linalg.cond(T, 2)
    print(f"\nCondition number κ(T) = ‖T‖·‖T⁻¹‖ = {kappa:.4f}")


open_mapping_demo()
bounded_inverse_demo()
```

---

## 📊 4. Closed Graph Theorem (Satz vom abgeschlossenen Graphen)

### Definition: Closed Graph

```
The graph of T: X → Y is:
Γ(T) = {(x, Tx) : x ∈ X} ⊆ X × Y

T has closed graph ⟺ Γ(T) is closed in X × Y
                    ⟺ (xₙ → x and Txₙ → y) ⟹ Tx = y
```

### Theorem

```
Let X, Y be Banach spaces and T: X → Y linear.

T is bounded ⟺ T has closed graph
```

### Python Illustration

```python
def closed_graph_demo():
    """Illustrate Closed Graph Theorem."""
    
    print("=== Closed Graph Theorem ===\n")
    
    print("For linear T: X → Y between Banach spaces:")
    print("T bounded ⟺ Graph(T) is closed\n")
    
    print("Graph(T) = {(x, Tx) : x ∈ X} ⊆ X × Y")
    print("\nClosed graph means:")
    print("If xₙ → x AND Txₙ → y, then Tx = y\n")
    
    # Example: Bounded operator (closed graph)
    print("=" * 50)
    print("\nExample 1: T(x) = 2x on ℝ (bounded, closed graph)")
    print("-" * 50)
    
    # Sequence converging
    x_n = [1 + 1/n for n in range(1, 6)]
    Tx_n = [2 * x for x in x_n]
    
    print(f"xₙ = 1 + 1/n → 1")
    print(f"Txₙ = 2xₙ → 2")
    print(f"T(1) = 2 ✓ (graph is closed)")
    
    # Example: Unbounded operator (not closed graph on different domain)
    print("\n" + "=" * 50)
    print("\nExample 2: Differentiation (unbounded)")
    print("-" * 50)
    print("D: C¹[0,1] → C[0,1] with ‖·‖∞")
    print("\nConsider fₙ(x) = sin(nx)/n")
    print("fₙ → 0 uniformly (in C[0,1])")
    print("f'ₙ(x) = cos(nx) does NOT converge")
    print("\nThe sequences (fₙ, f'ₙ) don't satisfy closed graph condition")
    print("because we'd need f'ₙ → g AND then D(0) = g")
    print("But D(0) = 0 ≠ 'limit' of f'ₙ")


def closed_graph_verification():
    """How to use closed graph theorem."""
    
    print("\n" + "="*50)
    print("\n=== Using Closed Graph Theorem ===\n")
    
    print("To show T is bounded, verify:")
    print("  xₙ → x AND Txₙ → y  ⟹  Tx = y\n")
    
    print("This is often EASIER than finding ‖T‖ directly!")
    print("\nExample: Multiplication operator on L²")
    print("-" * 50)
    print("(Mf)(x) = g(x)·f(x) where g ∈ L∞")
    print("\nTo show M is bounded via closed graph:")
    print("If fₙ → f in L² and Mfₙ → h in L²")
    print("Then g·fₙ → g·f in L² (since g bounded)")
    print("So h = g·f = Mf ✓")
    print("\nTherefore M is bounded by Closed Graph Theorem.")


closed_graph_demo()
closed_graph_verification()
```

---

## 🔗 5. Relationships Between Theorems

### Logical Connections

```
                    Baire Category Theorem
                           ↓
              ┌───────────┴───────────┐
              ↓                       ↓
    Uniform Boundedness      Open Mapping Theorem
              ↓                       ↓
              └───────────┬───────────┘
                          ↓
              Closed Graph Theorem
              
    Hahn-Banach (independent, uses Zorn's Lemma)
```

### Summary Table

```python
def theorem_summary():
    """Summary of the four fundamental theorems."""
    
    print("=== Summary: Four Pillars of Functional Analysis ===\n")
    
    theorems = [
        ("Hahn-Banach", 
         "Extend bounded functionals",
         "X vector space, U subspace",
         "Existence of rich dual space"),
        
        ("Uniform Boundedness",
         "Pointwise bounded ⟹ uniformly bounded",
         "X Banach, Y normed",
         "Convergence of operator sequences"),
        
        ("Open Mapping",
         "Surjective bounded T is open",
         "X, Y Banach",
         "Bounded inverse theorem"),
        
        ("Closed Graph",
         "Bounded ⟺ closed graph",
         "X, Y Banach",
         "Alternative boundedness proof"),
    ]
    
    print(f"{'Theorem':<22} {'Statement':<40} {'Requires':<20}")
    print("=" * 82)
    
    for name, statement, requires, _ in theorems:
        print(f"{name:<22} {statement:<40} {requires:<20}")
    
    print("\n" + "=" * 82)
    print("\nKey Applications:")
    for name, _, _, application in theorems:
        print(f"  {name}: {application}")


theorem_summary()
```

---

## 📐 6. Important Applications

### Application 1: Weak Convergence

```python
def weak_convergence_demo():
    """Weak convergence in Hilbert spaces."""
    
    print("=== Weak Convergence ===\n")
    
    print("xₙ ⇀ x (weakly) ⟺ ⟨xₙ, y⟩ → ⟨x, y⟩ ∀y ∈ H")
    print("\nBy Uniform Boundedness:")
    print("If xₙ ⇀ x, then sup_n ‖xₙ‖ < ∞")
    print("\nWeak convergence implies boundedness!")
    
    print("\n" + "-" * 50)
    print("\nExample in l²:")
    print("eₙ = (0,...,0,1,0,...) (1 in n-th position)")
    print("⟨eₙ, y⟩ = yₙ → 0 for any y ∈ l²")
    print("\nSo eₙ ⇀ 0 weakly")
    print("But ‖eₙ‖ = 1 ↛ 0, so NOT strong convergence!")


weak_convergence_demo()
```

### Application 2: Equivalent Norms

```python
def equivalent_norms_demo():
    """Using Open Mapping for equivalent norms."""
    
    print("=== Equivalent Norms via Open Mapping ===\n")
    
    print("If ‖·‖₁ and ‖·‖₂ both make X complete,")
    print("and ‖x‖₁ ≤ C‖x‖₂ for some C,")
    print("then the norms are equivalent!\n")
    
    print("Proof:")
    print("Consider id: (X, ‖·‖₂) → (X, ‖·‖₁)")
    print("By assumption, id is bounded.")
    print("Both spaces are Banach, id is bijective.")
    print("By Bounded Inverse Theorem, id⁻¹ is bounded.")
    print("So ∃c: ‖x‖₂ ≤ c‖x‖₁")
    print("Therefore c‖x‖₁ ≤ ‖x‖₂ ≤ C‖x‖₁ ✓")


equivalent_norms_demo()
```

### Application 3: Closed Subspace Complementation

```python
def complementation_demo():
    """Closed subspace complementation."""
    
    print("=== Closed Subspace Complementation ===\n")
    
    print("Let M ⊆ X be a closed subspace of Banach space X.")
    print("\nIf ∃ closed subspace N with X = M ⊕ N (algebraic direct sum),")
    print("then the projection P: X → M is bounded.")
    print("\nProof: Use Closed Graph Theorem!")
    print("\nNote: Not every closed subspace has a closed complement!")
    print("(c₀ in l∞ is a counterexample)")


complementation_demo()
```

---

## 📋 7. Proof Techniques Summary

### When to Use Each Theorem

```
Hahn-Banach:
- Extend functionals
- Separate convex sets
- Show dual space is "large"

Uniform Boundedness:
- Show operator sequence is uniformly bounded
- Prove convergence of operators
- Resonance/unboundedness results

Open Mapping:
- Show inverse is continuous
- Prove equivalence of norms
- Quotient space arguments

Closed Graph:
- Alternative way to prove boundedness
- When directly estimating ‖T‖ is hard
```

---

## 📋 8. Exam Checklist (Klausur)

### Theorems to State

- [ ] Hahn-Banach (extension form)
- [ ] Hahn-Banach (separation/geometric form)
- [ ] Uniform Boundedness Principle
- [ ] Open Mapping Theorem
- [ ] Bounded Inverse Theorem
- [ ] Closed Graph Theorem

### Hypotheses Required

- [ ] Hahn-Banach: sublinear functional p, f ≤ p on subspace
- [ ] UBP: X Banach (complete!), pointwise bounded
- [ ] Open Mapping: X, Y Banach, T surjective
- [ ] Closed Graph: X, Y Banach, T linear

### Key Applications

- [ ] Dual space separates points (Hahn-Banach)
- [ ] Weak convergence implies boundedness (UBP)
- [ ] Continuous bijection has continuous inverse (Open Mapping)
- [ ] Alternative boundedness proofs (Closed Graph)

### Common Exam Tasks

- [ ] State theorem with correct hypotheses
- [ ] Apply theorem to specific operator
- [ ] Identify which theorem to use
- [ ] Prove a consequence using these theorems

---

## 🔗 Related Documents

- [01-metric-normed-spaces.md](./01-metric-normed-spaces.md) - Foundations
- [02-operators.md](./02-operators.md) - Linear operators
- [03-hilbert-spaces.md](./03-hilbert-spaces.md) - Hilbert spaces

---

## 📚 References

- Werner, "Funktionalanalysis", Kapitel III, IV
- Rudin, "Functional Analysis", Chapters 2-5
- Brezis, "Functional Analysis, Sobolev Spaces and PDEs"

---

*Part of the [AMP-Studies](https://github.com/e49nana/AMP-Studies) repository*

*Last updated: February 5, 2026*
