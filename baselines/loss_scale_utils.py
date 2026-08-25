# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Reward/loss scale-alignment helper (BudgetMem, Zhang et al. 2026, Eq.10).

BudgetMem found that combining a task loss/reward and a cost loss/reward of
mismatched variance lets the higher-variance term dominate the combined
objective's gradient, collapsing optimization toward whichever term has more
spread (their Fig.3 ablation) -- a low-cost-only policy in their RL setting.
This isn't RL-specific: it happens to any ``combined = task_loss + lambda_ *
cost_loss`` where the two terms live on different scales. Every score/cost
head trained in this repo so far (ridge/GBM/binomial-GLM score heads,
quantile cost heads -- 실험DD/N/Q/NN etc.) has been trained as fully separate
models with independent loss functions, so this hasn't come up yet.

Not wired into anything yet -- there is no current experiment with a joint
score+cost objective. Keep this ready for the day one is tried (e.g. a
multi-task extension of 실험SS's mu-conditional classifier that predicts cost
alongside the model choice): call this each time the recent task/cost loss
history is updated, and scale the cost term by its return value before adding
it to the task loss, so neither term structurally dominates just because it
happens to have larger variance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def scale_aligned_weight(
    task_losses_recent: Sequence[float],
    cost_losses_recent: Sequence[float],
    eps: float = 1e-6,
) -> float:
    """Ratio of recent task-loss spread to recent cost-loss spread.

    Multiply the cost loss by this before adding it to the task loss so the
    combined objective isn't dominated by whichever term has more variance:
    ``combined_loss = task_loss + lambda_ * scale_aligned_weight(...) * cost_loss``.
    ``lambda_`` still controls the actual trade-off strength; this only
    equalizes the two terms' scales first.
    """
    return float(np.std(task_losses_recent) / (np.std(cost_losses_recent) + eps))
