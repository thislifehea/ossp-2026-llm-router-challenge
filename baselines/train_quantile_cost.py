# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train per-model cost heads as GBM quantile regressors (실험N).

Replaces the mean-cost-then-blanket-safety_ratio pattern with a directly
learned high-percentile cost estimate: quantiles are preserved under the
monotonic exp() retransform, so training GradientBoostingRegressor(loss=
"quantile") on log(cost) and exponentiating gives the alpha-th percentile of
actual cost per episode -- the per-episode uncertainty (e.g. axk1-think's
volatile output length) is baked into the prediction itself instead of a
single tier-wide multiplier.

Score heads are left untouched (reuses the adopted K ensemble's score
prediction, i.e. mean of the ridge and GBM score heads) -- this experiment
isolates the cost-head change only, same pattern as 실험J (Gamma GLM cost
head).

Run with `.venv-embed`'s python (needs scikit-learn); reuses
baselines/hash_regex.py's feature extraction (features/select logic do not
need NumPy, only the training step does).
"""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hash_regex as hr
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes, policy_sha256

ALPHA_GRID: Sequence[float] = (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
GBM_PARAMS = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}


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
    X = np.asarray(
        [hr.raw_feature_vector(ep, 256) for ep in train_inputs.episodes], dtype=np.float64
    )
    y = {
        model_id: np.asarray(
            [
                math.log(_outcome_cost(outcome_index[(ep.episode_id, model_id)], policy))
                for ep in train_inputs.episodes
            ]
        )
        for model_id in MODEL_IDS
    }
    print("feature matrix:", X.shape)

    artifact = {
        "artifact_type": "ossp-quantile-gbm-cost-v1",
        "schema_version": 1,
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "gbm_params": GBM_PARAMS,
        "hash_bins": 256,
        "alpha_heads": {},
    }
    for alpha in ALPHA_GRID:
        print(f"training alpha={alpha} ...")
        heads = {}
        for model_id in MODEL_IDS:
            model = GradientBoostingRegressor(
                loss="quantile", alpha=alpha, random_state=0, **GBM_PARAMS
            )
            model.fit(X, y[model_id])
            heads[model_id] = _export_head(model)
        artifact["alpha_heads"][str(alpha)] = heads

    out_path = root / "artifacts/quantile-gbm-cost.v1.json"
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
