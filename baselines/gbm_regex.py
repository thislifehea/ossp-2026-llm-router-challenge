# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Route with regex/hash features and a trained Gradient Boosted Trees model.

Experiment variant of ``baselines/hash_regex.py``: the exact same feature
extraction (``hash_regex.raw_feature_vector``) and the exact same budget
optimizer (``hash_regex.select_models`` / ``hash_regex.fill_ax31_upgrades``)
are reused unchanged. Only the score/cost prediction heads differ: instead of
six ridge-regression linear heads, this module replays six independently
trained ``sklearn.ensemble.GradientBoostingRegressor`` models (score/log-cost
x ax31-light/ax31/axk1-think) that were exported as plain JSON tree
structures by ``baselines/train_gbm.py``.

This module intentionally does NOT import NumPy or scikit-learn: at
inference (runtime submission) time, trees are walked with plain Python
using only the ``math`` standard-library module, matching the
lightweight-dependency convention already used by ``hash_regex.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from hash_regex import (
    DENSE_FEATURE_NAMES,
    FEATURE_VERSION,
    MAX_HASH_BINS,
    MIN_HASH_BINS,
    PREMIUM_AX31_FILL_SAFETY_RATIO,
    fill_ax31_upgrades,
    raw_feature_vector,
    select_models,
)
from ossp_router.heuristic import write_submission_atomic
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_json,
    load_policy,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)


ARTIFACT_TYPE = "ossp-gbm-hash-regex-v1"


@dataclass(frozen=True)
class GbmTree:
    children_left: Tuple[int, ...]
    children_right: Tuple[int, ...]
    feature: Tuple[int, ...]
    threshold: Tuple[float, ...]
    value: Tuple[float, ...]


@dataclass(frozen=True)
class GbmHead:
    init_value: float
    learning_rate: float
    trees: Tuple[GbmTree, ...]


@dataclass(frozen=True)
class GbmArtifact:
    hash_bins: int
    score_heads: Mapping[str, GbmHead]
    log_cost_heads: Mapping[str, GbmHead]
    tier_safety_ratios: Mapping[str, float]
    policy_id: str
    policy_digest: str
    training_summary: Mapping[str, Any]


@dataclass(frozen=True)
class GbmPlan:
    submission: Submission
    predicted_budget_ratio: float
    safety_ratio: float
    ax31_fill_safety_ratio: Optional[float]


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label}은(는) JSON 객체여야 합니다.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        raise ProtocolError(f"{label} 필드 오류: 누락={missing}, 초과={extra}")


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ProtocolError(f"{label} 값이 허용 범위를 벗어났습니다.")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ProtocolError(f"{label}은(는) 유한한 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{label}은(는) 유한한 숫자여야 합니다.")
    return result


def _int_array(value: Any, label: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{label}은(는) 비어 있지 않은 배열이어야 합니다.")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ProtocolError(f"{label}[{index}]은(는) 정수여야 합니다.")
        result.append(item)
    return tuple(result)


def _float_array(value: Any, label: str, length: int) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ProtocolError(f"{label}은(는) 길이 {length}의 배열이어야 합니다.")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _parse_tree(value: Any, label: str, feature_dim: int) -> GbmTree:
    raw = _object(value, label)
    _exact_keys(
        raw,
        ("children_left", "children_right", "feature", "threshold", "value"),
        label,
    )
    children_left = _int_array(raw["children_left"], f"{label}.children_left")
    node_count = len(children_left)
    children_right = _int_array(raw["children_right"], f"{label}.children_right")
    feature = _int_array(raw["feature"], f"{label}.feature")
    threshold = _float_array(raw["threshold"], f"{label}.threshold", node_count)
    value_array = _float_array(raw["value"], f"{label}.value", node_count)
    if not (len(children_right) == len(feature) == node_count):
        raise ProtocolError(f"{label}의 노드 배열 길이가 서로 다릅니다.")
    for index in range(node_count):
        left = children_left[index]
        right = children_right[index]
        is_leaf = left == -1
        if is_leaf != (right == -1):
            raise ProtocolError(f"{label}.children[{index}]의 leaf 여부가 일관되지 않습니다.")
        if not is_leaf:
            if not (0 <= left < node_count) or not (0 <= right < node_count):
                raise ProtocolError(f"{label}.children[{index}]의 자식 인덱스가 범위를 벗어났습니다.")
            if not 0 <= feature[index] < feature_dim:
                raise ProtocolError(f"{label}.feature[{index}]가 특징 차원 범위를 벗어났습니다.")
    return GbmTree(
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        value=value_array,
    )


def _parse_head(value: Any, label: str, feature_dim: int) -> GbmHead:
    raw = _object(value, label)
    _exact_keys(raw, ("init_value", "learning_rate", "trees"), label)
    init_value = _number(raw["init_value"], f"{label}.init_value")
    learning_rate = _number(raw["learning_rate"], f"{label}.learning_rate")
    if learning_rate <= 0:
        raise ProtocolError(f"{label}.learning_rate은(는) 0보다 커야 합니다.")
    trees_raw = raw["trees"]
    if not isinstance(trees_raw, list) or not trees_raw:
        raise ProtocolError(f"{label}.trees은(는) 비어 있지 않은 배열이어야 합니다.")
    trees = tuple(
        _parse_tree(item, f"{label}.trees[{index}]", feature_dim)
        for index, item in enumerate(trees_raw)
    )
    return GbmHead(init_value=init_value, learning_rate=learning_rate, trees=trees)


def parse_artifact(value: Any) -> GbmArtifact:
    root = _object(value, "artifact")
    expected = (
        "artifact_type",
        "schema_version",
        "feature_version",
        "hash_algorithm",
        "hash_bins",
        "dense_feature_names",
        "model_ids",
        "policy_id",
        "policy_sha256",
        "gbm_params",
        "score_heads",
        "log_cost_heads",
        "tier_safety_ratios",
        "training_summary",
    )
    _exact_keys(root, expected, "artifact")
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise ProtocolError("지원하지 않는 gbm artifact_type입니다.")
    if (
        _integer(root["schema_version"], "artifact.schema_version", 1, 1) != 1
        or _integer(
            root["feature_version"], "artifact.feature_version", FEATURE_VERSION, FEATURE_VERSION
        )
        != FEATURE_VERSION
    ):
        raise ProtocolError("지원하지 않는 gbm artifact 버전입니다.")
    if root["hash_algorithm"] != "fnv1a64-signed-word-1-2":
        raise ProtocolError("지원하지 않는 feature hash 방식입니다.")
    hash_bins = _integer(root["hash_bins"], "artifact.hash_bins", MIN_HASH_BINS, MAX_HASH_BINS)
    if hash_bins & (hash_bins - 1):
        raise ProtocolError("artifact.hash_bins는 2의 거듭제곱이어야 합니다.")
    if root["dense_feature_names"] != list(DENSE_FEATURE_NAMES):
        raise ProtocolError("dense feature 정의가 현재 런타임과 다릅니다.")
    if root["model_ids"] != list(MODEL_IDS):
        raise ProtocolError("artifact.model_ids가 공개 정책 모델과 다릅니다.")
    feature_dim = len(DENSE_FEATURE_NAMES) + hash_bins
    gbm_params = _object(root["gbm_params"], "artifact.gbm_params")
    _exact_keys(gbm_params, ("n_estimators", "max_depth", "learning_rate"), "artifact.gbm_params")
    score_raw = _object(root["score_heads"], "artifact.score_heads")
    cost_raw = _object(root["log_cost_heads"], "artifact.log_cost_heads")
    if set(score_raw) != set(MODEL_IDS) or set(cost_raw) != set(MODEL_IDS):
        raise ProtocolError("artifact GBM head의 모델 집합이 올바르지 않습니다.")
    safety_raw = _object(root["tier_safety_ratios"], "artifact.tier_safety_ratios")
    if set(safety_raw) != set(TIERS):
        raise ProtocolError("artifact 등급별 안전계수가 완전하지 않습니다.")
    safety = {
        tier: _number(safety_raw[tier], f"artifact.tier_safety_ratios.{tier}") for tier in TIERS
    }
    if any(not 0 < value <= 1 for value in safety.values()):
        raise ProtocolError("artifact 안전계수는 0보다 크고 1 이하여야 합니다.")
    policy_id = root["policy_id"]
    policy_digest = root["policy_sha256"]
    if not isinstance(policy_id, str) or not policy_id:
        raise ProtocolError("artifact.policy_id가 올바르지 않습니다.")
    if not isinstance(policy_digest, str) or re.fullmatch(r"[0-9a-f]{64}", policy_digest) is None:
        raise ProtocolError("artifact.policy_sha256가 올바르지 않습니다.")
    training_summary = _object(root["training_summary"], "artifact.training_summary")
    return GbmArtifact(
        hash_bins=hash_bins,
        score_heads={
            model_id: _parse_head(score_raw[model_id], f"score_heads.{model_id}", feature_dim)
            for model_id in MODEL_IDS
        },
        log_cost_heads={
            model_id: _parse_head(cost_raw[model_id], f"log_cost_heads.{model_id}", feature_dim)
            for model_id in MODEL_IDS
        },
        tier_safety_ratios=safety,
        policy_id=policy_id,
        policy_digest=policy_digest,
        training_summary=dict(training_summary),
    )


def load_artifact(path: Path) -> GbmArtifact:
    return parse_artifact(load_json(path))


def _tree_value(tree: GbmTree, features: Sequence[float]) -> float:
    node = 0
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value
    while children_left[node] != -1:
        if features[feature[node]] <= threshold[node]:
            node = children_left[node]
        else:
            node = children_right[node]
    return value[node]


def _gbm_predict(head: GbmHead, features: Sequence[float]) -> float:
    total = head.init_value + head.learning_rate * math.fsum(
        _tree_value(tree, features) for tree in head.trees
    )
    return total


def predict_episode(
    episode: Episode, artifact: GbmArtifact
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    raw = raw_feature_vector(episode, artifact.hash_bins)
    scores = {
        model_id: min(1.0, max(0.0, _gbm_predict(artifact.score_heads[model_id], raw)))
        for model_id in MODEL_IDS
    }
    costs = {
        model_id: math.exp(
            min(50.0, max(-50.0, _gbm_predict(artifact.log_cost_heads[model_id], raw)))
        )
        for model_id in MODEL_IDS
    }
    light = costs[MODEL_IDS[0]]
    costs[MODEL_IDS[1]] = max(costs[MODEL_IDS[1]], light * (1.0 + 1e-12))
    costs[MODEL_IDS[2]] = max(costs[MODEL_IDS[2]], costs[MODEL_IDS[1]] * (1.0 + 1e-12))
    return scores, costs


def make_gbm_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: GbmArtifact,
    tier: str,
) -> GbmPlan:
    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    if artifact.policy_id != policy.policy_id:
        raise ProtocolError("artifact와 정책의 policy_id가 다릅니다.")
    if artifact.policy_digest != policy_sha256(policy):
        raise ProtocolError("artifact와 현재 정책의 SHA-256이 다릅니다.")
    predictions = [predict_episode(episode, artifact) for episode in inputs.episodes]
    scores = [item[0] for item in predictions]
    costs = [item[1] for item in predictions]
    safety = artifact.tier_safety_ratios[tier]
    selected, ratio = select_models(
        scores,
        costs,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=safety,
    )
    fill_safety = None
    if tier == "premium":
        fill_safety = PREMIUM_AX31_FILL_SAFETY_RATIO
        selected, ratio = fill_ax31_upgrades(
            selected,
            scores,
            costs,
            budget_multiplier=float(policy.tiers[tier].budget_multiplier),
            safety_ratio=fill_safety,
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
    return GbmPlan(
        submission=parse_submission(submission_to_dict(submission)),
        predicted_budget_ratio=ratio,
        safety_ratio=safety,
        ax31_fill_safety_ratio=fill_safety,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="학습된 hash-regex 특징 + GBM 트리 baseline 라우터"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
        artifact = load_artifact(args.artifact)
        plan = make_gbm_submission(inputs, policy, artifact, args.tier)
        write_submission_atomic(args.output, plan.submission)
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    safety_message = f"기본 안전계수 {plan.safety_ratio:.4f}"
    if plan.ax31_fill_safety_ratio is not None:
        safety_message += f", AX31 fill 안전계수 {plan.ax31_fill_safety_ratio:.4f}"
    print(
        "OK: "
        f"{args.tier} 제출 파일을 생성했습니다 "
        f"(예측 비용 비율 {plan.predicted_budget_ratio:.6f}, {safety_message})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
