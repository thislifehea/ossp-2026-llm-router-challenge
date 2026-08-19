# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train per-model GBM quantile regressors for SCORE (실험S).

N/O/Q applied quantile regression only to the *cost* head; the *score*
(quality) head has stayed a point estimate (mean of ridge+GBM) the whole
day. The Lagrangian selection (score - mu*cost) trusts that point estimate
exactly -- on an episode where the score prediction is itself uncertain
(e.g. borderline between two models), a point estimate can pick the cheap
model when the *true* quality gap could go either way. This trains a
*lower*-quantile score head (e.g. 15th percentile: "this is the quality
we're fairly confident we'll get, not just the average") as a more
risk-averse quality signal to feed the same selection logic.

Unlike cost (which is log-transformed because it's multiplicatively scaled
and always positive), score is already bounded in [0, 1], so no transform is
applied -- quantile regression targets it directly.

Run with `.venv-embed`'s python (needs scikit-learn).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hash_regex as hr
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes, policy_sha256

ALPHA_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
GBM_PARAMS = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}


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
    score = {
        model_id: np.asarray(
            [float(outcome_index[(ep.episode_id, model_id)].score) for ep in episodes]
        )
        for model_id in MODEL_IDS
    }
    print("feature matrix:", X.shape)

    artifact = {
        "artifact_type": "ossp-quantile-gbm-score-v1",
        "schema_version": 1,
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "gbm_params": GBM_PARAMS,
        "hash_bins": 256,
        "alpha_heads": {},
    }
    for alpha in ALPHA_GRID:
        print(f"training alpha={alpha} score heads ...")
        artifact["alpha_heads"][str(alpha)] = {}
        for model_id in MODEL_IDS:
            model = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=0, **GBM_PARAMS)
            model.fit(X, score[model_id])
            artifact["alpha_heads"][str(alpha)][model_id] = _export_head(model)

    out_path = root / "artifacts/quantile-gbm-score.v1.json"
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
