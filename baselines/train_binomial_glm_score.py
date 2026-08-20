# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train binomial-GLM (logistic link) score heads for 실험DD.

outcomes.json's score is not an arbitrary [0,1] continuous value -- it is
k/n, the fraction of num_generations (2 or 4) sampled generations that were
correct. All prior score heads (ridge, GBM) treated this as a plain
regression target under squared-error loss, which implicitly assumes every
observation is equally reliable regardless of n. A binomial GLM instead
expands each (episode, model) observation into n Bernoulli trials (k
successes + (n-k) failures, same feature row repeated) and fits standard
L2-regularized logistic regression on the expanded rows -- this is
mathematically the weighted-binomial-likelihood MLE, so n=4 observations
naturally get more influence than n=2 ones.

Run with `.venv-embed`'s python (needs scikit-learn).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hash_regex as hr
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes, policy_sha256
from ossp_router.routing_ensemble import DENSE_FEATURE_NAMES

C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
HASH_BINS = 256


def _head_to_json(intercept: float, coefficients: np.ndarray) -> Mapping[str, Any]:
    return {"intercept": float(intercept), "coefficients": coefficients.tolist()}


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
    print("shape:", Xs.shape)

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

        rows_x, rows_y, rows_ep = [], [], []
        for i in range(n_episodes):
            n, k = n_arr[i], k_arr[i]
            if k > 0:
                rows_x.append(np.tile(Xs[i], (k, 1)))
                rows_y.append(np.ones(k))
                rows_ep.append(np.full(k, i))
            if n - k > 0:
                rows_x.append(np.tile(Xs[i], (n - k, 1)))
                rows_y.append(np.zeros(n - k))
                rows_ep.append(np.full(n - k, i))
        Xe = np.concatenate(rows_x, axis=0)
        ye = np.concatenate(rows_y, axis=0)
        epe = np.concatenate(rows_ep, axis=0)
        print(f"  {model_id}: expanded rows = {len(ye)} (positives={ye.sum():.0f})")

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        ep_idx = np.arange(n_episodes)
        best_c, best_ll = None, None
        for c in C_GRID:
            fold_lls = []
            for fit_ep, val_ep in kf.split(ep_idx):
                fit_mask = np.isin(epe, fit_ep)
                val_mask = np.isin(epe, val_ep)
                if ye[fit_mask].sum() == 0 or ye[fit_mask].sum() == fit_mask.sum():
                    continue
                clf = LogisticRegression(C=c, max_iter=2000)
                clf.fit(Xe[fit_mask], ye[fit_mask])
                p = np.clip(clf.predict_proba(Xe[val_mask])[:, 1], 1e-9, 1 - 1e-9)
                ll = (ye[val_mask] * np.log(p) + (1 - ye[val_mask]) * np.log(1 - p)).mean()
                fold_lls.append(ll)
            mean_ll = sum(fold_lls) / len(fold_lls)
            if best_ll is None or mean_ll > best_ll:
                best_ll = mean_ll
                best_c = c
        print(f"    selected C={best_c} (cv_loglik={best_ll:.5f})")

        final_clf = LogisticRegression(C=best_c, max_iter=3000)
        final_clf.fit(Xe, ye)
        score_heads[model_id] = _head_to_json(final_clf.intercept_[0], final_clf.coef_[0])

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
    out_path = root / "artifacts/binomial-glm-score.v1.json"
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
