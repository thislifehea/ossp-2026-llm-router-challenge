# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train GBM heads on hybrid (regex-dense + multilingual sentence embedding) features.

Untried combination flagged in EXPERIMENTS.md: ridge was already tried on
hash+regex features (H) and on embedding+regex features (실험 임베딩-hybrid,
0.686023, worse than baseline); GBM was already tried on hash+regex features
(실험I, 0.694375, worse than H). This script is the last untested corner:
GBM on embedding+regex features.

Reuses cached embeddings (see build/embedding-cache/, verified by SHA-256 of
the input file + model name to match this repo's current
data/materialized/{train,dev}/inputs.json) so it needs NumPy + scikit-learn
only -- no working torch import required, sidestepping this environment's
torch DLL loading failure (see EXPERIMENTS.md 실험E/F).

Run with `.venv-embed`'s python (numpy + scikit-learn installed).
"""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import embedding_regex as er
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
    policy_sha256,
)

DENSE_LEN = 14
EMBEDDING_DIM = 384
FEATURE_DIM = DENSE_LEN + EMBEDDING_DIM


def _cache_path(cache_dir: Path, input_path: Path, model_name: str) -> Path:
    import hashlib

    digest = hashlib.sha256()
    digest.update(input_path.read_bytes())
    digest.update(model_name.encode("utf-8"))
    return cache_dir / f"embeddings-{digest.hexdigest()}.npy"


def _load_cache(input_path: Path, model_name: str, cache_dir: Path) -> np.ndarray:
    path = _cache_path(cache_dir, input_path, model_name)
    if not path.is_file():
        raise FileNotFoundError(f"임베딩 캐시가 없습니다: {path}")
    return np.load(path)


def _outcome_cost(outcome, policy) -> float:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    cost = (
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )
    return float(cost)


def _targets(inputs, outcomes, policy) -> np.ndarray:
    index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    rows = []
    for episode in inputs.episodes:
        row = []
        for model_id in MODEL_IDS:
            row.append(float(index[(episode.episode_id, model_id)].score))
        for model_id in MODEL_IDS:
            row.append(math.log(_outcome_cost(index[(episode.episode_id, model_id)], policy)))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _oof_predictions(X: np.ndarray, y: np.ndarray, *, folds: int, params: Mapping[str, Any]) -> np.ndarray:
    predictions = np.empty_like(y)
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    for train_idx, valid_idx in kf.split(X):
        for col in range(y.shape[1]):
            model = GradientBoostingRegressor(random_state=0, **params)
            model.fit(X[train_idx], y[train_idx, col])
            predictions[valid_idx, col] = model.predict(X[valid_idx])
    return predictions


def _select_hyperparams(X: np.ndarray, y: np.ndarray, folds: int) -> Tuple[Mapping[str, Any], Mapping[str, float]]:
    grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n in (50, 100)
        for d in (2, 3)
        for lr in (0.05, 0.1)
    ]
    diagnostics: Dict[str, float] = {}
    best = None
    for params in grid:
        predictions = _oof_predictions(X, y, folds=folds, params=params)
        score_mse = float(np.mean((predictions[:, :3] - y[:, :3]) ** 2))
        cost_mse = float(np.mean((predictions[:, 3:] - y[:, 3:]) ** 2))
        objective = score_mse + 0.05 * cost_mse
        key = json.dumps(params, sort_keys=True)
        diagnostics[key] = objective
        if best is None or objective < best[0]:
            best = (objective, params)
    assert best is not None
    return best[1], diagnostics


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
    return {
        "init_value": init_value,
        "learning_rate": float(model.learning_rate),
        "trees": trees,
    }


def _fit_final(X: np.ndarray, y: np.ndarray, params: Mapping[str, Any]) -> List[GradientBoostingRegressor]:
    models = []
    for col in range(y.shape[1]):
        model = GradientBoostingRegressor(random_state=0, **params)
        model.fit(X, y[:, col])
        models.append(model)
    return models


def main() -> int:
    policy = load_bundled_policy()
    root = Path(__file__).resolve().parents[1]
    train_input_path = root / "data/materialized/train/inputs.json"
    dev_input_path = root / "data/materialized/dev/inputs.json"
    cache_dir = root / "build/embedding-cache"

    train_inputs = load_input(train_input_path)
    train_outcomes = load_outcomes(root / "data/train/outcomes.json")
    dev_inputs = load_input(dev_input_path)
    dev_outcomes = load_outcomes(root / "data/dev/outcomes.json")

    train_embeddings = _load_cache(train_input_path, er.DEFAULT_EMBEDDING_MODEL, cache_dir)
    dev_embeddings = _load_cache(dev_input_path, er.DEFAULT_EMBEDDING_MODEL, cache_dir)
    print("embeddings loaded:", train_embeddings.shape, dev_embeddings.shape)

    X_train = er.raw_feature_matrix(
        train_inputs.episodes, feature_mode="hybrid", embedding_cache=train_embeddings
    )
    X_dev = er.raw_feature_matrix(
        dev_inputs.episodes, feature_mode="hybrid", embedding_cache=dev_embeddings
    )
    print("feature matrices:", X_train.shape, X_dev.shape)

    y_train = _targets(train_inputs, train_outcomes, policy)

    params, diagnostics = _select_hyperparams(X_train, y_train, folds=5)
    print("selected GBM hyperparams:", params)

    models = _fit_final(X_train, y_train, params)

    artifact = {
        "artifact_type": "ossp-embedding-gbm-v1",
        "schema_version": 1,
        "feature_version": 1,
        "embedding_model": er.DEFAULT_EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "feature_mode": "hybrid",
        "dense_feature_len": DENSE_LEN,
        "feature_dim": FEATURE_DIM,
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "gbm_params": params,
        "score_heads": {
            model_id: _export_head(models[index]) for index, model_id in enumerate(MODEL_IDS)
        },
        "log_cost_heads": {
            model_id: _export_head(models[3 + index]) for index, model_id in enumerate(MODEL_IDS)
        },
        "training_summary": {
            "num_train_episodes": len(train_inputs.episodes),
            "folds": 5,
            "alpha_objectives": diagnostics,
            "optimizer": "sklearn-gbm-oof-grid-v1",
        },
    }
    out_path = root / "artifacts/embedding-gbm.v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_path)

    np.save(root / "build/embedding-cache/dev_features_hybrid.npy", X_dev)
    print("wrote dev feature cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
