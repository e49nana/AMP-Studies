#!/usr/bin/env python3
"""
Program 3: Cancellation (Auslöschung) demonstration
We compute (1 + 10^(-k)) - 1 for k = 1, ..., 20
and compare with the exact value 10^(-k).

For large k, floating-point arithmetic cannot distinguish
1 + 10^(-k) from 1, so the result becomes 0 and
the relative error explodes.
"""

def main() -> None:
    print("=== Program 3: Cancellation (Auslöschung) ===")
    print("{:>3s}  {:>20s}  {:>20s}".format("k", "float result", "absolute error"))

    for k in range(1, 21):
        eps = 10.0 ** (-k)
        res = (1.0 + eps) - 1.0      # subject to rounding
        err = abs(res - eps)        # exact value would be eps
        print(f"{k:3d}  {res:20.12e}  {err:20.12e}")

    print("\nFor large k, (1 + 10^(-k)) is rounded to 1.0,")
    print("so (1 + 10^(-k)) - 1 becomes 0, showing catastrophic cancellation.")


if __name__ == "__main__":
    main()
