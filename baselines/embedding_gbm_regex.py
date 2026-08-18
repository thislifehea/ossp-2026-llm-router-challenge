# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Route with hybrid (regex-dense + multilingual embedding) features and GBM trees.

Pairs with baselines/train_embedding_gbm.py's artifact
(artifacts/embedding-gbm.v1.json). Tree traversal is pure Python (no numpy
needed at inference), matching gbm_regex.py's convention -- but building the
feature vector still needs an embedding for the input text, which (outside
this experiment's cached-embedding shortcut) requires torch +
sentence-transformers at runtime. See embedding_regex.py's runtime-dependency
tradeoff note; the same tradeoff applies here.
"""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hash_regex
import embedding_regex as er
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    InputBatch,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
    parse_submission,
    submission_to_dict,
)
from ossp_router.heuristic import write_submission_atomic


def load_artifact(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_value(tree: Mapping[str, Any], features: Sequence[float]) -> float:
    node = 0
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]
    value = tree["value"]
    while children_left[node] != -1:
        if features[feature[node]] <= threshold[node]:
            node = children_left[node]
        else:
            node = children_right[node]
    return value[node]


def _head_predict(head: Mapping[str, Any], features: Sequence[float]) -> float:
    return head["init_value"] + head["learning_rate"] * math.fsum(
        _tree_value(tree, features) for tree in head["trees"]
    )


def predict_all(
    episodes: Sequence[Episode], artifact: Mapping[str, Any], embedding_cache
) -> Tuple[Sequence[Mapping[str, float]], Sequence[Mapping[str, float]]]:
    raw = er.raw_feature_matrix(episodes, feature_mode="hybrid", embedding_cache=embedding_cache)
    scores_out = []
    costs_out = []
    for row in raw:
        scores = {
            model_id: min(1.0, max(0.0, _head_predict(artifact["score_heads"][model_id], row)))
            for model_id in MODEL_IDS
        }
        costs = {
            model_id: math.exp(
                min(50.0, max(-50.0, _head_predict(artifact["log_cost_heads"][model_id], row)))
            )
            for model_id in MODEL_IDS
        }
        light = costs[MODEL_IDS[0]]
        costs[MODEL_IDS[1]] = max(costs[MODEL_IDS[1]], light * (1.0 + 1e-12))
        costs[MODEL_IDS[2]] = max(costs[MODEL_IDS[2]], costs[MODEL_IDS[1]] * (1.0 + 1e-12))
        scores_out.append(scores)
        costs_out.append(costs)
    return scores_out, costs_out


def make_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: Mapping[str, Any],
    tier: str,
    safety_ratio: float,
    embedding_cache,
) -> Tuple[Submission, float]:
    scores, costs = predict_all(inputs.episodes, artifact, embedding_cache)
    selected, ratio = hash_regex.select_models(
        scores,
        costs,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=safety_ratio,
    )
    if tier == "premium":
        selected, ratio = hash_regex.fill_ax31_upgrades(
            selected,
            scores,
            costs,
            budget_multiplier=float(policy.tiers[tier].budget_multiplier),
            safety_ratio=hash_regex.PREMIUM_AX31_FILL_SAFETY_RATIO,
        )
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model_id)
            for episode, model_id in zip(inputs.episodes, selected)
        ),
    )
    return parse_submission(submission_to_dict(submission)), ratio
