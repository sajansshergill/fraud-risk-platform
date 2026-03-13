from __future__ import annotations

"""
Compatibility wrapper so that:

    python -m src.features.build_payments_features

works by delegating to the actual implementation in
`build_payment_features.py`.
"""

from .build_payment_features import build_payments_features, main

__all__ = ["build_payments_features", "main"]


if __name__ == "__main__":
    main()

