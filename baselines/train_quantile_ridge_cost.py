# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train linear quantile-regression cost heads (실험P: ensemble with N's GBM quantile head).

K found that averaging two independently-trained models (ridge + GBM) for
the *score* head reduced prediction noise and beat either alone. N never
applied that idea to the *cost* head -- it swapped the cost head entirely
for a single GBM quantile regressor. This script trains the linear
counterpart (sklearn.linear_model.QuantileRegressor, pinball loss + L1) at
the same quantile levels N uses, so the two can be mean-ensembled the same
way K ensembled ridge+GBM scores.

Run with `.venv-embed`'s python (needs scikit-learn >= 1.1 for QuantileRegressor).
"""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.linear_model import QuantileRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hash_regex as hr
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes, policy_sha256

ALPHA_LEVELS = ("0.65", "0.75")
L1_ALPHA_GRID = (0.0001, 0.001, 0.01, 0.1)


def _outcome_cost(outcome, policy) -> float:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    cost = (
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )
    return float(cost)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    policy = load_bundled_policy()
    train_inputs = load_input(root / "data/materialized/train/inputs.json")
    train_outcomes = load_outcomes(root / "data/train/outcomes.json")
    outcome_index = {(o.episode_id, o.model_id): o for o in train_outcomes.outcomes}

    print("building feature matrix...")
    episodes = list(train_inputs.episodes)
    X_raw = np.asarray([hr.raw_feature_vector(ep, 256) for ep in episodes], dtype=np.float64)
    mean = X_raw.mean(axis=0)
    scale = X_raw.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    X = (X_raw - mean) / scale

    log_cost = {
        model_id: np.asarray(
            [math.log(_outcome_cost(outcome_index[(ep.episode_id, model_id)], policy)) for ep in episodes]
        )
        for model_id in MODEL_IDS
    }

    # simple 5-fold CV to pick L1 alpha per (quantile, model) using pinball loss
    n = len(episodes)
    rng = np.random.default_rng(0)
    fold_ids = rng.permutation(n) % 5

    artifact: Mapping = {
        "artifact_type": "ossp-quantile-ridge-cost-v1",
        "schema_version": 1,
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "hash_bins": 256,
        "feature_mean": [float(v) for v in mean],
        "feature_scale": [float(v) for v in scale],
        "alpha_heads": {},
    }

    for q_key in ALPHA_LEVELS:
        q = float(q_key)
        print(f"quantile={q} ...")
        artifact["alpha_heads"][q_key] = {}
        for model_id in MODEL_IDS:
            y = log_cost[model_id]
            best = None
            for l1_alpha in L1_ALPHA_GRID:
                losses = []
                for fold in range(5):
                    valid = fold_ids == fold
                    fit = ~valid
                    model = QuantileRegressor(quantile=q, alpha=l1_alpha, solver="highs")
                    model.fit(X[fit], y[fit])
                    pred = model.predict(X[valid])
                    losses.append(_pinball_loss(y[valid], pred, q))
                mean_loss = float(np.mean(losses))
                if best is None or mean_loss < best[0]:
                    best = (mean_loss, l1_alpha)
            l1_alpha = best[1]
            model = QuantileRegressor(quantile=q, alpha=l1_alpha, solver="highs")
            model.fit(X, y)
            artifact["alpha_heads"][q_key][model_id] = {
                "intercept": float(model.intercept_),
                "coefficients": [float(v) for v in model.coef_],
                "l1_alpha": l1_alpha,
            }
            print(f"  {model_id}: l1_alpha={l1_alpha} cv_pinball={best[0]:.6f}")

    out_path = root / "artifacts/quantile-ridge-cost.v1.json"
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
