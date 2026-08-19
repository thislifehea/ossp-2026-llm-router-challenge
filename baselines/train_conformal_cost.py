# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train a median-cost GBM head + split-conformal calibration residuals (실험O).

Extends 실험N (quantile-GBM cost head, tier-wide alpha chosen by Dev grid
search) with a statistically justified margin: instead of grid-searching a
single quantile level per tier against Dev, this fits a *median* cost
predictor on a genuine fit/calibration split of Train, then stores the
calibration-set residuals. At evaluation/runtime, any desired miscoverage
rate delta can be converted into a per-model additive correction via the
standard split-conformal formula:

    correction(delta) = the ceil((n_calib + 1) * (1 - delta)) / n_calib
                         empirical quantile of {actual_cost_i - median_pred_i}
                         over the calibration set
    cost_upper(x) = median_pred(x) + correction(delta)

This gives a *marginal coverage guarantee*: P(actual_cost <= cost_upper(X))
>= 1 - delta for a fresh draw from the same distribution as the calibration
set (Lei et al. 2018 / conformalized regression) -- unlike N's tier-wide
safety_ratio or alpha, which were picked by grid-searching whatever passed
Dev's self-check, with no guarantee they would generalize to a different
sample.

Run with `.venv-embed`'s python (needs scikit-learn).
"""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hash_regex as hr
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes, policy_sha256

GBM_PARAMS = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}
CALIB_FRACTION = 0.2
SEED = 0
ALPHA_GRID = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)


def _outcome_cost(outcome, policy) -> float:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    cost = (
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )
    return float(cost)


def _export_tree(tree) -> Mapping[str, Any]:
    return {
        "children_left": [int(v) for v in tree.children_left],
        "children_right": [int(v) for v in tree.children_right],
        "feature": [int(v) for v in tree.feature],
        "threshold": [float(v) for v in tree.threshold],
        "value": [float(v) for v in tree.value.reshape(-1)],
    }


def _export_head(model: GradientBoostingRegressor) -> Mapping[str, Any]:
    trees = [_export_tree(est[0].tree_) for est in model.estimators_]
    init_value = float(model.init_.constant_.reshape(-1)[0])
    return {"init_value": init_value, "learning_rate": float(model.learning_rate), "trees": trees}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    policy = load_bundled_policy()
    train_inputs = load_input(root / "data/materialized/train/inputs.json")
    train_outcomes = load_outcomes(root / "data/train/outcomes.json")
    outcome_index = {(o.episode_id, o.model_id): o for o in train_outcomes.outcomes}

    print("building feature matrix...")
    episodes = list(train_inputs.episodes)
    X = np.asarray([hr.raw_feature_vector(ep, 256) for ep in episodes], dtype=np.float64)
    actual_cost = {
        model_id: np.asarray(
            [_outcome_cost(outcome_index[(ep.episode_id, model_id)], policy) for ep in episodes]
        )
        for model_id in MODEL_IDS
    }
    log_cost = {model_id: np.log(actual_cost[model_id]) for model_id in MODEL_IDS}

    rng = np.random.default_rng(SEED)
    n = len(episodes)
    perm = rng.permutation(n)
    n_calib = int(round(n * CALIB_FRACTION))
    calib_idx = perm[:n_calib]
    fit_idx = perm[n_calib:]
    print(f"fit={len(fit_idx)} calib={len(calib_idx)}")

    artifact = {
        "artifact_type": "ossp-conformal-gbm-cost-v1",
        "schema_version": 1,
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "gbm_params": GBM_PARAMS,
        "hash_bins": 256,
        "residual_space": "log",
        "calib_fraction": CALIB_FRACTION,
        "seed": SEED,
        "alpha_heads": {},
        "alpha_calib_residuals": {},
    }
    # Conformalized Quantile Regression (Romano et al. 2019): the base
    # predictor is itself a quantile-GBM (like 실험N), not a median -- so the
    # per-episode heteroscedasticity N already learned (e.g. axk1-think's
    # volatile output length getting a wider margin than predictable
    # episodes) is preserved, and conformal calibration only adds the small
    # correction needed to hit *exact* coverage instead of trusting the
    # GBM quantile loss to be perfectly calibrated on unseen data.
    for alpha in ALPHA_GRID:
        print(f"training alpha={alpha} quantile heads ...")
        artifact["alpha_heads"][str(alpha)] = {}
        artifact["alpha_calib_residuals"][str(alpha)] = {}
        for model_id in MODEL_IDS:
            model = GradientBoostingRegressor(
                loss="quantile", alpha=alpha, random_state=0, **GBM_PARAMS
            )
            model.fit(X[fit_idx], log_cost[model_id][fit_idx])
            artifact["alpha_heads"][str(alpha)][model_id] = _export_head(model)

            quantile_log_pred_calib = model.predict(X[calib_idx])
            residuals = log_cost[model_id][calib_idx] - quantile_log_pred_calib
            artifact["alpha_calib_residuals"][str(alpha)][model_id] = sorted(
                float(v) for v in residuals
            )

    out_path = root / "artifacts/conformal-gbm-cost.v1.json"
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
