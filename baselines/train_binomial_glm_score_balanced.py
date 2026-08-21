# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train balanced-tier-specific binomial-GLM score heads for 실험LL.

DD's single binomial-GLM artifact uses per-model C selected via CV
log-likelihood (light=0.001, ax31=0.003, axk1-think=0.003) and was shared
by both fast and balanced tiers. A robustness-first C re-search per tier
(실험LL) found that balanced specifically does better with a UNIFORM
C=0.001 across all three models (+0.0011 tier_score vs the shared
artifact), while fast keeps DD's original mixed-C artifact unchanged.

Run with `.venv-embed`'s python (needs scikit-learn).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hash_regex as hr
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes, policy_sha256
from ossp_router.routing_ensemble import DENSE_FEATURE_NAMES

C = 0.001  # uniform across all three models, robustness-validated for balanced (실험LL)
HASH_BINS = 256


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    policy = load_bundled_policy()
    train_inputs = load_input(root / "data/materialized/train/inputs.json")
    train_outcomes = load_outcomes(root / "data/train/outcomes.json")
    outcome_index = {(o.episode_id, o.model_id): o for o in train_outcomes.outcomes}
    episodes = list(train_inputs.episodes)
    n_episodes = len(episodes)

    print("building feature matrix...")
    X = np.asarray([hr.raw_feature_vector(ep, HASH_BINS) for ep in episodes], dtype=np.float64)
    feature_mean = X.mean(axis=0)
    feature_scale = X.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    Xs = (X - feature_mean) / feature_scale

    score_heads: dict[str, Mapping[str, Any]] = {}
    for model_id in MODEL_IDS:
        n_arr = np.zeros(n_episodes, dtype=np.int64)
        k_arr = np.zeros(n_episodes, dtype=np.int64)
        for i, ep in enumerate(episodes):
            outcome = outcome_index[(ep.episode_id, model_id)]
            n = int(outcome.num_generations)
            k = int(round(float(outcome.score) * n))
            n_arr[i] = n
            k_arr[i] = k

        rows_x, rows_y = [], []
        for i in range(n_episodes):
            n, k = n_arr[i], k_arr[i]
            if k > 0:
                rows_x.append(np.tile(Xs[i], (k, 1)))
                rows_y.append(np.ones(k))
            if n - k > 0:
                rows_x.append(np.tile(Xs[i], (n - k, 1)))
                rows_y.append(np.zeros(n - k))
        Xe = np.concatenate(rows_x, axis=0)
        ye = np.concatenate(rows_y, axis=0)
        print(f"  {model_id}: expanded rows = {len(ye)} (positives={ye.sum():.0f})")

        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(Xe, ye)
        score_heads[model_id] = {"intercept": float(clf.intercept_[0]), "coefficients": clf.coef_[0].tolist()}

    artifact = {
        "artifact_type": "ossp-binomial-glm-score-v1",
        "schema_version": 1,
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "hash_bins": HASH_BINS,
        "dense_feature_names": list(DENSE_FEATURE_NAMES),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "score_heads": score_heads,
    }
    out_path = root / "artifacts/binomial-glm-score-balanced.v1.json"
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
