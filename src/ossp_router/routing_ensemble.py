# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Adopted routing policy: ridge (hash+regex) and GBM (hash+regex) mean ensemble.

This is the runtime counterpart of the offline experiments recorded as
EXPERIMENTS.md rows H (ridge), I (GBM) and K (this ensemble). The two trained
artifacts are bundled read-only in ``ossp_router.resources`` and loaded once
per process; per-episode routing calls neither model nor any network service.

Feature extraction and the two per-model prediction heads are ports of
``baselines/hash_regex.py`` and ``baselines/gbm_regex.py`` (kept dependency-free
of NumPy/scikit-learn on purpose, matching those modules' runtime contract) —
``baselines/`` is not shipped in the submission container, so this module does
not import from it. Any future change to the hashing/regex feature scheme must
be mirrored by hand in both places.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .heuristic import episode_text, extract_features, write_submission_atomic
from .protocol import (
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
    load_policy,
    loads_json,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)

FEATURE_VERSION = 1
MIN_HASH_BINS = 16
MAX_HASH_BINS = 16_384
AX31_FILL_SAFETY_RATIO = 0.65

# Calibrated on the public Dev split with a *robustness-first* search:
# instead of picking whatever passes the official self-check on the full
# 880-episode Dev batch, each candidate safety_ratio was additionally
# required to keep the tier's budget under cap on 10 independently-
# reshuffled ~440-episode halves of Dev before its Dev-880 tier_score was
# even considered (실험Q, 2026-08-19; 실험N's looser ratios scored higher on
# Dev-880 itself but failed budget on resampled halves).
#
# 실험DD(2026-08-20)에서 fast/balanced의 score 앙상블이 릿지+GBM에서
# 릿지+이항GLM으로 바뀌면서, Q 시절 값(fast=0.985/balanced=0.9125)을 그대로
# 쓰면 10/10 리샘플 검증을 통과하지 못한다는 게 확인됨(9/10, 9/10) -- 새
# score 모델에 맞춰 fast/balanced를 재탐색, premium은 score 앙상블이 안
# 바뀌었으므로 그대로 유지. Not part of any trained artifact because they
# are properties of the *routing policy*, not of a single model.
_TIER_SAFETY_RATIOS: Mapping[str, float] = {
    "fast": 0.98,
    "balanced": 0.90,
    "premium": 0.85,
}

# Which trained quantile level (see resources/quantile-gbm-cost.v1.json) each
# tier's cost head uses -- tighter budgets calibrated to a higher (more
# conservative) percentile of the cost distribution.
_TIER_COST_QUANTILE: Mapping[str, str] = {
    "fast": "0.75",
    "balanced": "0.65",
    "premium": "0.65",
}

_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_UINT64_MASK = (1 << 64) - 1
_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_FORMAL_REASONING = re.compile(
    r"\b(?:prove|derive|theorem|lemma|counterexample|induction|"
    r"증명|유도|정리|보조정리|반례|귀납)\b",
    re.IGNORECASE,
)
_PROGRAM_ANALYSIS = re.compile(
    r"```|\b(?:traceback|exception|complexity|big[- ]?o|"
    r"시간\s*복잡도|공간\s*복잡도|예외|스택\s*추적)\b",
    re.IGNORECASE,
)
_MULTI_CONSTRAINT = re.compile(
    r"\b(?:exactly|at least|at most|must|only|without|"
    r"정확히|이상|이하|반드시|오직|제외하고)\b",
    re.IGNORECASE,
)
_SIMPLE_TRANSFORM = re.compile(
    r"\b(?:summari[sz]e|rewrite|translate|list|extract|"
    r"요약|바꾸|번역|나열|추출)\b",
    re.IGNORECASE,
)

DENSE_FEATURE_NAMES = (
    "log_character_count",
    "log_word_count",
    "log_sentence_count",
    "log_message_count",
    "hangul_ratio",
    "log_code_marker_count",
    "log_math_marker_count",
    "numeric_density",
    "long_context",
    "log_reasoning_marker_count",
    "formal_reasoning",
    "program_analysis",
    "log_multi_constraint_count",
    "simple_transform",
)


def _stable_hash(value: str) -> int:
    digest = _FNV_OFFSET
    for byte in value.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _UINT64_MASK
    return digest


def _normalized_tokens(text: str) -> Tuple[str, ...]:
    result = []
    for token in _TOKEN.findall(text):
        normalized = token.casefold()
        if normalized.isdecimal():
            normalized = "<number>"
        result.append(normalized)
    return tuple(result)


def raw_feature_vector(episode: Episode, hash_bins: int) -> Tuple[float, ...]:
    """Build dense regex features plus signed word unigram/bigram hash bins."""

    if (
        isinstance(hash_bins, bool)
        or not isinstance(hash_bins, int)
        or not MIN_HASH_BINS <= hash_bins <= MAX_HASH_BINS
        or hash_bins & (hash_bins - 1)
    ):
        raise ValueError("hash_bins는 허용 범위의 2의 거듭제곱이어야 합니다.")
    features = extract_features(episode)
    text = episode_text(episode)
    dense = (
        math.log1p(features.character_count),
        math.log1p(features.word_count),
        math.log1p(features.sentence_count),
        math.log1p(features.message_count),
        features.hangul_ratio,
        math.log1p(features.code_marker_count),
        math.log1p(features.math_marker_count),
        features.numeric_density,
        float(features.long_context),
        math.log1p(features.reasoning_marker_count),
        float(bool(_FORMAL_REASONING.search(text))),
        float(bool(_PROGRAM_ANALYSIS.search(text))),
        math.log1p(len(_MULTI_CONSTRAINT.findall(text))),
        float(bool(_SIMPLE_TRANSFORM.search(text))),
    )
    bins = [0.0] * hash_bins
    tokens = _normalized_tokens(text)
    hashed_features = [f"w1:{token}" for token in tokens]
    hashed_features.extend(
        f"w2:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:])
    )
    for value in hashed_features:
        digest = _stable_hash(value)
        index = digest & (hash_bins - 1)
        bins[index] += -1.0 if digest & (1 << 63) else 1.0
    norm = math.sqrt(sum(value * value for value in bins))
    if norm:
        bins = [value / norm for value in bins]
    return dense + tuple(bins)


# === shared artifact-parsing helpers ===


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


def _vector(value: Any, length: int, label: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ProtocolError(f"{label}은(는) 길이 {length}의 배열이어야 합니다.")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _int_array(value: Any, label: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{label}은(는) 비어 있지 않은 배열이어야 합니다.")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ProtocolError(f"{label}[{index}]은(는) 정수여야 합니다.")
        result.append(item)
    return tuple(result)


def _check_policy_binding(policy_id: str, policy_digest: str, policy: RoutingPolicy, label: str) -> None:
    if policy_id != policy.policy_id:
        raise ProtocolError(f"{label}와 정책의 policy_id가 다릅니다.")
    if policy_digest != policy_sha256(policy):
        raise ProtocolError(f"{label}와 현재 정책의 SHA-256이 다릅니다.")


# === ridge (hash_regex) artifact ===


@dataclass(frozen=True)
class LinearHead:
    intercept: float
    coefficients: Tuple[float, ...]


@dataclass(frozen=True)
class RidgeArtifact:
    hash_bins: int
    feature_mean: Tuple[float, ...]
    feature_scale: Tuple[float, ...]
    score_heads: Mapping[str, LinearHead]
    log_cost_heads: Mapping[str, LinearHead]
    policy_id: str
    policy_digest: str


def _parse_linear_head(value: Any, length: int, label: str) -> LinearHead:
    raw = _object(value, label)
    _exact_keys(raw, ("intercept", "coefficients"), label)
    return LinearHead(
        intercept=_number(raw["intercept"], f"{label}.intercept"),
        coefficients=_vector(raw["coefficients"], length, f"{label}.coefficients"),
    )


def _parse_ridge_artifact(value: Any) -> RidgeArtifact:
    root = _object(value, "ridge artifact")
    if root["artifact_type"] != "ossp-hash-regex-linear-v1":
        raise ProtocolError("지원하지 않는 ridge artifact_type입니다.")
    hash_bins = _integer(root["hash_bins"], "artifact.hash_bins", MIN_HASH_BINS, MAX_HASH_BINS)
    if hash_bins & (hash_bins - 1):
        raise ProtocolError("artifact.hash_bins는 2의 거듭제곱이어야 합니다.")
    if tuple(root["dense_feature_names"]) != DENSE_FEATURE_NAMES:
        raise ProtocolError("dense feature 정의가 현재 런타임과 다릅니다.")
    if tuple(root["model_ids"]) != MODEL_IDS:
        raise ProtocolError("artifact.model_ids가 공개 정책 모델과 다릅니다.")
    length = len(DENSE_FEATURE_NAMES) + hash_bins
    mean = _vector(root["feature_mean"], length, "artifact.feature_mean")
    scale = _vector(root["feature_scale"], length, "artifact.feature_scale")
    if any(item <= 0 for item in scale):
        raise ProtocolError("artifact.feature_scale은 모두 0보다 커야 합니다.")
    score_raw = _object(root["score_heads"], "artifact.score_heads")
    cost_raw = _object(root["log_cost_heads"], "artifact.log_cost_heads")
    return RidgeArtifact(
        hash_bins=hash_bins,
        feature_mean=mean,
        feature_scale=scale,
        score_heads={
            model_id: _parse_linear_head(score_raw[model_id], length, f"score_heads.{model_id}")
            for model_id in MODEL_IDS
        },
        log_cost_heads={
            model_id: _parse_linear_head(cost_raw[model_id], length, f"log_cost_heads.{model_id}")
            for model_id in MODEL_IDS
        },
        policy_id=root["policy_id"],
        policy_digest=root["policy_sha256"],
    )


def _linear(head: LinearHead, values: Sequence[float]) -> float:
    return head.intercept + math.fsum(
        coefficient * value for coefficient, value in zip(head.coefficients, values)
    )


def _predict_ridge(
    episode: Episode, artifact: RidgeArtifact
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    raw = raw_feature_vector(episode, artifact.hash_bins)
    standardized = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(raw, artifact.feature_mean, artifact.feature_scale)
    )
    scores = {
        model_id: min(1.0, max(0.0, _linear(artifact.score_heads[model_id], standardized)))
        for model_id in MODEL_IDS
    }
    costs = {
        model_id: math.exp(
            min(50.0, max(-50.0, _linear(artifact.log_cost_heads[model_id], standardized)))
        )
        for model_id in MODEL_IDS
    }
    return scores, costs


# === GBM artifact ===


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
    policy_id: str
    policy_digest: str


def _float_array(value: Any, label: str, length: int) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ProtocolError(f"{label}은(는) 길이 {length}의 배열이어야 합니다.")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _parse_gbm_tree(value: Any, label: str, feature_dim: int) -> GbmTree:
    raw = _object(value, label)
    _exact_keys(raw, ("children_left", "children_right", "feature", "threshold", "value"), label)
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


def _parse_gbm_head(value: Any, label: str, feature_dim: int) -> GbmHead:
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
        _parse_gbm_tree(item, f"{label}.trees[{index}]", feature_dim)
        for index, item in enumerate(trees_raw)
    )
    return GbmHead(init_value=init_value, learning_rate=learning_rate, trees=trees)


def _parse_gbm_artifact(value: Any) -> GbmArtifact:
    root = _object(value, "gbm artifact")
    if root["artifact_type"] != "ossp-gbm-hash-regex-v1":
        raise ProtocolError("지원하지 않는 gbm artifact_type입니다.")
    hash_bins = _integer(root["hash_bins"], "artifact.hash_bins", MIN_HASH_BINS, MAX_HASH_BINS)
    if hash_bins & (hash_bins - 1):
        raise ProtocolError("artifact.hash_bins는 2의 거듭제곱이어야 합니다.")
    if tuple(root["dense_feature_names"]) != DENSE_FEATURE_NAMES:
        raise ProtocolError("dense feature 정의가 현재 런타임과 다릅니다.")
    if tuple(root["model_ids"]) != MODEL_IDS:
        raise ProtocolError("artifact.model_ids가 공개 정책 모델과 다릅니다.")
    feature_dim = len(DENSE_FEATURE_NAMES) + hash_bins
    score_raw = _object(root["score_heads"], "artifact.score_heads")
    cost_raw = _object(root["log_cost_heads"], "artifact.log_cost_heads")
    return GbmArtifact(
        hash_bins=hash_bins,
        score_heads={
            model_id: _parse_gbm_head(score_raw[model_id], f"score_heads.{model_id}", feature_dim)
            for model_id in MODEL_IDS
        },
        log_cost_heads={
            model_id: _parse_gbm_head(cost_raw[model_id], f"log_cost_heads.{model_id}", feature_dim)
            for model_id in MODEL_IDS
        },
        policy_id=root["policy_id"],
        policy_digest=root["policy_sha256"],
    )


def _tree_value(tree: GbmTree, features: Sequence[float]) -> float:
    node = 0
    while tree.children_left[node] != -1:
        if features[tree.feature[node]] <= tree.threshold[node]:
            node = tree.children_left[node]
        else:
            node = tree.children_right[node]
    return tree.value[node]


def _gbm_predict(head: GbmHead, features: Sequence[float]) -> float:
    return head.init_value + head.learning_rate * math.fsum(
        _tree_value(tree, features) for tree in head.trees
    )


def _predict_gbm(
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
    return scores, costs


# === binomial GLM score artifact (실험DD: score is k/n successes over
# num_generations trials, not an arbitrary continuous value -- a logistic-link
# head trained on the expanded per-trial Bernoulli outcomes gives n=4
# observations proportionally more weight than n=2 ones, unlike a plain MSE
# regression that treats every score value as equally reliable) ===


@dataclass(frozen=True)
class BinomialGlmArtifact:
    feature_mean: Tuple[float, ...]
    feature_scale: Tuple[float, ...]
    score_heads: Mapping[str, LinearHead]
    policy_id: str
    policy_digest: str


def _parse_binomial_glm_artifact(value: Any) -> BinomialGlmArtifact:
    root = _object(value, "binomial glm artifact")
    if root["artifact_type"] != "ossp-binomial-glm-score-v1":
        raise ProtocolError("지원하지 않는 binomial glm artifact_type입니다.")
    hash_bins = _integer(root["hash_bins"], "artifact.hash_bins", MIN_HASH_BINS, MAX_HASH_BINS)
    if hash_bins & (hash_bins - 1):
        raise ProtocolError("artifact.hash_bins는 2의 거듭제곱이어야 합니다.")
    if tuple(root["dense_feature_names"]) != DENSE_FEATURE_NAMES:
        raise ProtocolError("dense feature 정의가 현재 런타임과 다릅니다.")
    if tuple(root["model_ids"]) != MODEL_IDS:
        raise ProtocolError("artifact.model_ids가 공개 정책 모델과 다릅니다.")
    length = len(DENSE_FEATURE_NAMES) + hash_bins
    mean = _vector(root["feature_mean"], length, "artifact.feature_mean")
    scale = _vector(root["feature_scale"], length, "artifact.feature_scale")
    if any(item <= 0 for item in scale):
        raise ProtocolError("artifact.feature_scale은 모두 0보다 커야 합니다.")
    score_raw = _object(root["score_heads"], "artifact.score_heads")
    return BinomialGlmArtifact(
        feature_mean=mean,
        feature_scale=scale,
        score_heads={
            model_id: _parse_linear_head(score_raw[model_id], length, f"score_heads.{model_id}")
            for model_id in MODEL_IDS
        },
        policy_id=root["policy_id"],
        policy_digest=root["policy_sha256"],
    )


def _predict_binomial_glm_score(
    episode: Episode, artifact: BinomialGlmArtifact
) -> Mapping[str, float]:
    raw = raw_feature_vector(episode, len(artifact.feature_mean) - len(DENSE_FEATURE_NAMES))
    standardized = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(raw, artifact.feature_mean, artifact.feature_scale)
    )
    scores = {}
    for model_id in MODEL_IDS:
        z = _linear(artifact.score_heads[model_id], standardized)
        scores[model_id] = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z))))
    return scores


# === ensemble prediction ===


def _predict_ensemble_score(
    episode: Episode,
    ridge_artifact: RidgeArtifact,
    gbm_artifact: GbmArtifact,
    binomial_glm_artifact: BinomialGlmArtifact,
    tier: str,
) -> Mapping[str, float]:
    """실험DD(2026-08-20): 로버스트니스 우선 검증으로 fast/balanced는
    릿지+이항GLM 블렌드가, premium은 기존 릿지+GBM 블렌드(실험K)가 각각
    더 낫다는 게 확인됨 -- score(성공비율)는 num_generations번 시행의
    이항분포 관측치라, MSE로 학습한 릿지/GBM과 로지스틱 링크로 학습한
    이항GLM은 서로 다른 종류의 오차를 범한다. premium처럼 light/think
    격차가 극단적인 결정에서는 GBM 쪽이, fast/balanced처럼 애매한 경계
    판단이 많은 곳에서는 이항GLM 쪽이 우위를 보임."""

    r_scores, _r_costs = _predict_ridge(episode, ridge_artifact)
    if tier == "premium":
        g_scores, _g_costs = _predict_gbm(episode, gbm_artifact)
        return {m: (r_scores[m] + g_scores[m]) / 2.0 for m in MODEL_IDS}
    b_scores = _predict_binomial_glm_score(episode, binomial_glm_artifact)
    return {m: (r_scores[m] + b_scores[m]) / 2.0 for m in MODEL_IDS}


# === quantile cost artifact (실험N: per-tier high-percentile cost head) ===
#
# Trained by baselines/train_quantile_cost.py: quantiles are preserved under
# the monotonic exp() retransform, so a GBM fit with loss="quantile" on
# log(cost) and exponentiated gives the alpha-th percentile of actual cost --
# per-episode uncertainty (e.g. axk1-think's volatile output length) is baked
# into the prediction itself, replacing most of what a flat tier-wide
# safety_ratio used to compensate for.


@dataclass(frozen=True)
class QuantileCostArtifact:
    hash_bins: int
    alpha_heads: Mapping[str, Mapping[str, GbmHead]]
    policy_id: str
    policy_digest: str


def _parse_quantile_cost_artifact(value: Any) -> QuantileCostArtifact:
    root = _object(value, "quantile cost artifact")
    if root["artifact_type"] != "ossp-quantile-gbm-cost-v1":
        raise ProtocolError("지원하지 않는 quantile cost artifact_type입니다.")
    hash_bins = _integer(root["hash_bins"], "artifact.hash_bins", MIN_HASH_BINS, MAX_HASH_BINS)
    feature_dim = len(DENSE_FEATURE_NAMES) + hash_bins
    alpha_raw = _object(root["alpha_heads"], "artifact.alpha_heads")
    alpha_heads = {}
    for alpha_key, heads_value in alpha_raw.items():
        heads_raw = _object(heads_value, f"alpha_heads.{alpha_key}")
        if set(heads_raw) != set(MODEL_IDS):
            raise ProtocolError(f"alpha_heads.{alpha_key}의 모델 집합이 올바르지 않습니다.")
        alpha_heads[alpha_key] = {
            model_id: _parse_gbm_head(heads_raw[model_id], f"alpha_heads.{alpha_key}.{model_id}", feature_dim)
            for model_id in MODEL_IDS
        }
    return QuantileCostArtifact(
        hash_bins=hash_bins,
        alpha_heads=alpha_heads,
        policy_id=root["policy_id"],
        policy_digest=root["policy_sha256"],
    )


def _gbm_quantile_cost(
    episode: Episode, artifact: QuantileCostArtifact, alpha_key: str
) -> Mapping[str, float]:
    raw = raw_feature_vector(episode, artifact.hash_bins)
    heads = artifact.alpha_heads[alpha_key]
    return {
        model_id: math.exp(min(50.0, max(-50.0, _gbm_predict(heads[model_id], raw))))
        for model_id in MODEL_IDS
    }


# === linear quantile-regression cost artifact (실험P/Q: ensembled with the
# GBM quantile head the same way K ensembled ridge+GBM for score) ===


@dataclass(frozen=True)
class QuantileRidgeCostArtifact:
    feature_mean: Tuple[float, ...]
    feature_scale: Tuple[float, ...]
    alpha_heads: Mapping[str, Mapping[str, LinearHead]]
    policy_id: str
    policy_digest: str


def _parse_quantile_ridge_cost_artifact(value: Any) -> QuantileRidgeCostArtifact:
    root = _object(value, "quantile ridge cost artifact")
    if root["artifact_type"] != "ossp-quantile-ridge-cost-v1":
        raise ProtocolError("지원하지 않는 quantile ridge cost artifact_type입니다.")
    hash_bins = _integer(root["hash_bins"], "artifact.hash_bins", MIN_HASH_BINS, MAX_HASH_BINS)
    length = len(DENSE_FEATURE_NAMES) + hash_bins
    mean = _vector(root["feature_mean"], length, "artifact.feature_mean")
    scale = _vector(root["feature_scale"], length, "artifact.feature_scale")
    if any(item <= 0 for item in scale):
        raise ProtocolError("artifact.feature_scale은 모두 0보다 커야 합니다.")
    alpha_raw = _object(root["alpha_heads"], "artifact.alpha_heads")
    alpha_heads = {}
    for alpha_key, heads_value in alpha_raw.items():
        heads_raw = _object(heads_value, f"alpha_heads.{alpha_key}")
        if set(heads_raw) != set(MODEL_IDS):
            raise ProtocolError(f"alpha_heads.{alpha_key}의 모델 집합이 올바르지 않습니다.")
        alpha_heads[alpha_key] = {
            model_id: LinearHead(
                intercept=_number(
                    heads_raw[model_id]["intercept"], f"alpha_heads.{alpha_key}.{model_id}.intercept"
                ),
                coefficients=_vector(
                    heads_raw[model_id]["coefficients"], length, f"alpha_heads.{alpha_key}.{model_id}.coefficients"
                ),
            )
            for model_id in MODEL_IDS
        }
    return QuantileRidgeCostArtifact(
        feature_mean=mean,
        feature_scale=scale,
        alpha_heads=alpha_heads,
        policy_id=root["policy_id"],
        policy_digest=root["policy_sha256"],
    )


def _ridge_quantile_cost(
    episode: Episode, artifact: QuantileRidgeCostArtifact, alpha_key: str
) -> Mapping[str, float]:
    raw = raw_feature_vector(episode, len(artifact.feature_mean) - len(DENSE_FEATURE_NAMES))
    standardized = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(raw, artifact.feature_mean, artifact.feature_scale)
    )
    heads = artifact.alpha_heads[alpha_key]
    return {
        model_id: math.exp(min(50.0, max(-50.0, _linear(heads[model_id], standardized))))
        for model_id in MODEL_IDS
    }


def _predict_quantile_cost(
    episode: Episode,
    gbm_artifact: QuantileCostArtifact,
    ridge_artifact: QuantileRidgeCostArtifact,
    alpha_key: str,
) -> Mapping[str, float]:
    """Mean-ensemble of the GBM and linear quantile cost predictors (실험Q)."""

    g_costs = _gbm_quantile_cost(episode, gbm_artifact, alpha_key)
    r_costs = _ridge_quantile_cost(episode, ridge_artifact, alpha_key)
    costs = {m: (g_costs[m] + r_costs[m]) / 2.0 for m in MODEL_IDS}
    light = costs[MODEL_IDS[0]]
    costs[MODEL_IDS[1]] = max(costs[MODEL_IDS[1]], light * (1.0 + 1e-12))
    costs[MODEL_IDS[2]] = max(costs[MODEL_IDS[2]], costs[MODEL_IDS[1]] * (1.0 + 1e-12))
    return costs


# === batch budget optimizer (unchanged from baselines/hash_regex.py) ===


def select_models(
    predicted_scores: Sequence[Mapping[str, float]],
    predicted_costs: Sequence[Mapping[str, float]],
    *,
    budget_multiplier: float,
    safety_ratio: float,
) -> Tuple[Tuple[str, ...], float]:
    if len(predicted_scores) != len(predicted_costs) or not predicted_scores:
        raise ValueError("예측 score와 cost는 같은 길이의 비어 있지 않은 배열이어야 합니다.")
    light_total = math.fsum(row[MODEL_IDS[0]] for row in predicted_costs)
    effective_ratio = max(1.0, budget_multiplier * safety_ratio)
    cap = light_total * effective_ratio

    def choose(penalty: float) -> Tuple[Tuple[str, ...], float]:
        selected = []
        for scores, costs in zip(predicted_scores, predicted_costs):
            model_id = max(
                MODEL_IDS,
                key=lambda candidate: (
                    scores[candidate] - penalty * costs[candidate] / light_total,
                    -MODEL_IDS.index(candidate),
                ),
            )
            selected.append(model_id)
        total = math.fsum(costs[model_id] for costs, model_id in zip(predicted_costs, selected))
        return tuple(selected), total

    selected, total = choose(0.0)
    if total > cap:
        low = 0.0
        high = 1.0
        selected, total = choose(high)
        while total > cap and high < 2**60:
            low = high
            high *= 2.0
            selected, total = choose(high)
        for _iteration in range(80):
            middle = (low + high) / 2.0
            candidate, candidate_total = choose(middle)
            if candidate_total <= cap:
                high = middle
                selected, total = candidate, candidate_total
            else:
                low = middle
    if total > cap:
        selected = tuple(MODEL_IDS[0] for _row in predicted_scores)
        total = light_total
    return selected, total / light_total


def fill_ax31_upgrades(
    selected: Sequence[str],
    predicted_scores: Sequence[Mapping[str, float]],
    predicted_costs: Sequence[Mapping[str, float]],
    *,
    budget_multiplier: float,
    safety_ratio: float,
) -> Tuple[Tuple[str, ...], float]:
    if (
        len(selected) != len(predicted_scores)
        or len(selected) != len(predicted_costs)
        or not selected
    ):
        raise ValueError("선택과 예측 배열은 같은 길이의 비어 있지 않은 배열이어야 합니다.")
    if not 0 < safety_ratio <= 1:
        raise ValueError("AX31 fill 안전계수는 0보다 크고 1 이하여야 합니다.")

    light_id, ax31_id, _premium_id = MODEL_IDS
    light_total = math.fsum(row[light_id] for row in predicted_costs)
    current_total = math.fsum(costs[model_id] for costs, model_id in zip(predicted_costs, selected))
    cap = max(current_total, light_total * max(1.0, budget_multiplier * safety_ratio))

    def choose(penalty: float) -> Tuple[Tuple[str, ...], float]:
        filled = []
        for model_id, scores, costs in zip(selected, predicted_scores, predicted_costs):
            if model_id != light_id:
                filled.append(model_id)
                continue
            incremental_score = scores[ax31_id] - scores[light_id]
            incremental_cost = costs[ax31_id] - costs[light_id]
            if incremental_score - penalty * incremental_cost / light_total > 0:
                filled.append(ax31_id)
            else:
                filled.append(light_id)
        total = math.fsum(costs[model_id] for costs, model_id in zip(predicted_costs, filled))
        return tuple(filled), total

    filled, total = choose(0.0)
    if total > cap:
        low = 0.0
        high = 1.0
        filled, total = choose(high)
        while total > cap and high < 2**60:
            low = high
            high *= 2.0
            filled, total = choose(high)
        for _iteration in range(80):
            middle = (low + high) / 2.0
            candidate, candidate_total = choose(middle)
            if candidate_total <= cap:
                high = middle
                filled, total = candidate, candidate_total
            else:
                low = middle
    if total > cap:
        return tuple(selected), current_total / light_total
    return filled, total / light_total


# === bundled artifact loading (cached per process) ===

_ridge_artifact_cache: RidgeArtifact | None = None
_gbm_artifact_cache: GbmArtifact | None = None
_quantile_cost_artifact_cache: QuantileCostArtifact | None = None
_quantile_ridge_cost_artifact_cache: QuantileRidgeCostArtifact | None = None
_binomial_glm_artifact_cache: BinomialGlmArtifact | None = None


def _load_bundled_ridge_artifact() -> RidgeArtifact:
    global _ridge_artifact_cache
    if _ridge_artifact_cache is None:
        try:
            text = resources.read_text(
                "ossp_router.resources",
                "hash-regex-bins256-alpha1000.v1.json",
                encoding="utf-8",
            )
        except (OSError, UnicodeError) as exc:
            raise ProtocolError(f"내장 ridge artifact를 읽을 수 없습니다: {exc}") from exc
        _ridge_artifact_cache = _parse_ridge_artifact(loads_json(text))
    return _ridge_artifact_cache


def _load_bundled_gbm_artifact() -> GbmArtifact:
    global _gbm_artifact_cache
    if _gbm_artifact_cache is None:
        try:
            text = resources.read_text(
                "ossp_router.resources",
                "gbm.v1.json",
                encoding="utf-8",
            )
        except (OSError, UnicodeError) as exc:
            raise ProtocolError(f"내장 gbm artifact를 읽을 수 없습니다: {exc}") from exc
        _gbm_artifact_cache = _parse_gbm_artifact(loads_json(text))
    return _gbm_artifact_cache


def _load_bundled_quantile_cost_artifact() -> QuantileCostArtifact:
    global _quantile_cost_artifact_cache
    if _quantile_cost_artifact_cache is None:
        try:
            text = resources.read_text(
                "ossp_router.resources",
                "quantile-gbm-cost.v1.json",
                encoding="utf-8",
            )
        except (OSError, UnicodeError) as exc:
            raise ProtocolError(f"내장 quantile cost artifact를 읽을 수 없습니다: {exc}") from exc
        _quantile_cost_artifact_cache = _parse_quantile_cost_artifact(loads_json(text))
    return _quantile_cost_artifact_cache


def _load_bundled_quantile_ridge_cost_artifact() -> QuantileRidgeCostArtifact:
    global _quantile_ridge_cost_artifact_cache
    if _quantile_ridge_cost_artifact_cache is None:
        try:
            text = resources.read_text(
                "ossp_router.resources",
                "quantile-ridge-cost.v1.json",
                encoding="utf-8",
            )
        except (OSError, UnicodeError) as exc:
            raise ProtocolError(f"내장 quantile ridge cost artifact를 읽을 수 없습니다: {exc}") from exc
        _quantile_ridge_cost_artifact_cache = _parse_quantile_ridge_cost_artifact(loads_json(text))
    return _quantile_ridge_cost_artifact_cache


def _load_bundled_binomial_glm_artifact() -> BinomialGlmArtifact:
    global _binomial_glm_artifact_cache
    if _binomial_glm_artifact_cache is None:
        try:
            text = resources.read_text(
                "ossp_router.resources",
                "binomial-glm-score.v1.json",
                encoding="utf-8",
            )
        except (OSError, UnicodeError) as exc:
            raise ProtocolError(f"내장 binomial glm artifact를 읽을 수 없습니다: {exc}") from exc
        _binomial_glm_artifact_cache = _parse_binomial_glm_artifact(loads_json(text))
    return _binomial_glm_artifact_cache


def make_submission(inputs: InputBatch, policy: RoutingPolicy, tier: str) -> Submission:
    """Route one tier with the adopted policy (EXPERIMENTS.md 실험DD): score
    head is ridge+binomial-GLM for fast/balanced and ridge+GBM (실험K) for
    premium (tier-specific, robustness-validated); cost head is tier-specific
    (GBM quantile + linear quantile) mean-ensembled (실험Q), safety_ratio is
    robustness-validated per tier."""

    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")

    ridge_artifact = _load_bundled_ridge_artifact()
    gbm_artifact = _load_bundled_gbm_artifact()
    quantile_gbm_artifact = _load_bundled_quantile_cost_artifact()
    quantile_ridge_artifact = _load_bundled_quantile_ridge_cost_artifact()
    binomial_glm_artifact = _load_bundled_binomial_glm_artifact()
    _check_policy_binding(ridge_artifact.policy_id, ridge_artifact.policy_digest, policy, "ridge artifact")
    _check_policy_binding(gbm_artifact.policy_id, gbm_artifact.policy_digest, policy, "gbm artifact")
    _check_policy_binding(
        quantile_gbm_artifact.policy_id, quantile_gbm_artifact.policy_digest, policy, "quantile gbm cost artifact"
    )
    _check_policy_binding(
        quantile_ridge_artifact.policy_id,
        quantile_ridge_artifact.policy_digest,
        policy,
        "quantile ridge cost artifact",
    )
    _check_policy_binding(
        binomial_glm_artifact.policy_id,
        binomial_glm_artifact.policy_digest,
        policy,
        "binomial glm artifact",
    )

    alpha_key = _TIER_COST_QUANTILE[tier]
    scores = [
        _predict_ensemble_score(episode, ridge_artifact, gbm_artifact, binomial_glm_artifact, tier)
        for episode in inputs.episodes
    ]
    costs = [
        _predict_quantile_cost(episode, quantile_gbm_artifact, quantile_ridge_artifact, alpha_key)
        for episode in inputs.episodes
    ]
    safety = _TIER_SAFETY_RATIOS[tier]
    selected, _ratio = select_models(
        scores,
        costs,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=safety,
    )
    # 실험CC(2026-08-20): 라그랑주 이분탐색은 캡에 살짝 못 미치는 미세한
    # 여유를 남길 수 있다(이산적 argmax 결정이라 mu가 연속적으로 캡에
    # 수렴하지 못함) -- 이 fill 패스는 그 여유를 light->ax31 업그레이드로
    # 마저 채운다. 원래 premium에만 적용했지만, 이 여유는 tier 구조상
    # 어느 tier에나 생길 수 있어 전 tier로 확장(실측 여유는 tier당
    # 0.02~0.12%로 작지만, 순손실 없는 방어적 개선이라 유지 비용이 없음).
    selected, _ratio = fill_ax31_upgrades(
        selected,
        scores,
        costs,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=AX31_FILL_SAFETY_RATIO,
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
    return parse_submission(submission_to_dict(submission))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-run",
        description="채택된 ridge+GBM 앙상블 라우터를 한 등급에 대해 실행합니다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = (
            load_policy(args.policy) if args.policy is not None else load_bundled_policy()
        )
        submission = make_submission(inputs, policy, args.tier)
        write_submission_atomic(args.output, submission)
    except (OSError, ProtocolError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} 제출 파일을 생성했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
