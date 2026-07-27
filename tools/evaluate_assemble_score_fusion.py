#!/usr/bin/env python3
"""Mainline assemble-score evaluation entrypoint.

This is the named MMRec mainline wrapper around late score fusion:

    score_final = (1 - alpha) * norm(score_base) + alpha * norm(score_aux)

The implementation lives in evaluate_late_score_fusion.py to preserve
backward compatibility with earlier experiment scripts.
"""

from evaluate_late_score_fusion import main


if __name__ == "__main__":
    main()
