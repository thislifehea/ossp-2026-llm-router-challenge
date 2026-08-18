# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Route with a pretrained multilingual sentence embedding + ridge regression.

Experimental alternative to ``hash_regex.py``: instead of (or in addition to)
signed feature-hashing of word n-grams, this baseline embeds ``episode_text``
with a pretrained sentence-transformers model and standardizes+ridge-regresses
on those (optionally concatenated with the 14 cheap regex dense features from
``hash_regex.DENSE_FEATURE_NAMES``).

Embedding model (third-party, used under its own license -- see
``baselines/requirements-embedding.txt`` and this module's docstring):
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  License: Apache-2.0. 384-dim, multilingual (supports Korean + English, both
  present in this challenge's prompts).

*** RUNTIME DEPENDENCY TRADEOFF ***
Every other policy in this repository (``heuristic.py``, ``prompt_heuristic.py``,
``feature_budget.py``, ``hash_regex.py``, ``gbm_regex.py``) performs inference
with either pure Python or NumPy only -- no ML framework needed at serving
time. This module is different: ``predict_episode``/``predict_batch`` run a
real transformer forward pass, so ``torch`` + ``sentence-transformers`` become
hard RUNTIME dependencies, not just training-time tooling. That means, if this
were adopted: (1) noticeably slower inference per batch (transformer forward
pass vs. regex/hash lookups), (2) the final submission container image
(``container/Dockerfile``, built for ``linux/arm64``) would need to bundle
torch + sentence-transformers + the embedding model weights, and (3) runtime
behavior would need to be re-validated with ``tools/check_runtime.py`` on that
image -- none of which has been done here. This module and its self-check
numbers only establish whether the *routing quality* is worth that cost; see
``EXPERIMENTS.md`` for the explicit tradeoff writeup before considering
adoption.

Run this module (and ``train_embedding.py``) with a Python environment that
has ``torch``, ``sentence-transformers``, and ``numpy`` installed -- the
system Python used for ``ossp_router.cli self-check`` does not need any of
these, since self-check only reads pre-generated submission JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without numpy
    np = None

import hash_regex
from ossp_router.heuristic import episode_text, write_submission_atomic
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


ARTIFACT_TYPE = "ossp-embedding-linear-v1"
FEATURE_VERSION = 1
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIM = 384
FEATURE_MODES = ("pure", "hybrid")
PREMIUM_AX31_FILL_SAFETY_RATIO = 0.65

_MODEL_CACHE: dict = {}


@dataclass(frozen=True)
class LinearHead:
    intercept: float
    coefficients: Tuple[float, ...]


@dataclass(frozen=True)
class EmbeddingArtifact:
    embedding_model: str
    embedding_dim: int
    feature_mode: str
    dense_feature_names: Tuple[str, ...]
    feature_mean: Tuple[float, ...]
    feature_scale: Tuple[float, ...]
    score_heads: Mapping[str, LinearHead]
    log_cost_heads: Mapping[str, LinearHead]
    tier_safety_ratios: Mapping[str, float]
    policy_id: str
    policy_digest: str
    training_summary: Mapping[str, Any]


@dataclass(frozen=True)
class EmbeddingPlan:
    submission: Submission
    predicted_budget_ratio: float
    safety_ratio: float
    ax31_fill_safety_ratio: Optional[float]


def _get_sentence_transformer(model_name: str):
    """Lazily load (and cache in-process) the sentence-transformers model."""

    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
):
    """Embed a batch of prompt/message texts with a pretrained sentence model."""

    if np is None:
        raise RuntimeError(
            "임베딩 계산에는 NumPy가 필요합니다. "
            "baselines/requirements-embedding.txt를 설치해 주세요."
        )
    model = _get_sentence_transformer(model_name)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float64)


def dense_feature_matrix(episodes: Sequence[Episode]):
    """Reuse hash_regex's validated 14 regex/length dense features (no hashing)."""

    if np is None:
        raise RuntimeError(
            "특징 계산에는 NumPy가 필요합니다. "
            "baselines/requirements-embedding.txt를 설치해 주세요."
        )
    dense_len = len(hash_regex.DENSE_FEATURE_NAMES)
    rows = [
        hash_regex.raw_feature_vector(episode, hash_regex.MIN_HASH_BINS)[:dense_len]
        for episode in episodes
    ]
    return np.asarray(rows, dtype=np.float64)


def raw_feature_matrix(
    episodes: Sequence[Episode],
    *,
    feature_mode: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 64,
    embedding_cache: Optional["np.ndarray"] = None,
):
    """Build the (n_episodes, n_features) matrix for pure or hybrid mode."""

    if np is None:
        raise RuntimeError(
            "특징 계산에는 NumPy가 필요합니다. "
            "baselines/requirements-embedding.txt를 설치해 주세요."
        )
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"알 수 없는 feature_mode: {feature_mode}")
    if embedding_cache is not None:
        embeddings = embedding_cache
    else:
        texts = [episode_text(episode) for episode in episodes]
        embeddings = embed_texts(
            texts, model_name=embedding_model, batch_size=embedding_batch_size
        )
    if feature_mode == "pure":
        return embeddings
    dense = dense_feature_matrix(episodes)
    return np.concatenate([dense, embeddings], axis=1)


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
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
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


def _head(value: Any, length: int, label: str) -> LinearHead:
    raw = _object(value, label)
    _exact_keys(raw, ("intercept", "coefficients"), label)
    return LinearHead(
        intercept=_number(raw["intercept"], f"{label}.intercept"),
        coefficients=_vector(
            raw["coefficients"], length, f"{label}.coefficients"
        ),
    )


def parse_artifact(value: Any) -> EmbeddingArtifact:
    root = _object(value, "artifact")
    expected = (
        "artifact_type",
        "schema_version",
        "feature_version",
        "embedding_model",
        "embedding_dim",
        "feature_mode",
        "dense_feature_names",
        "model_ids",
        "policy_id",
        "policy_sha256",
        "feature_mean",
        "feature_scale",
        "score_heads",
        "log_cost_heads",
        "tier_safety_ratios",
        "training_summary",
    )
    _exact_keys(root, expected, "artifact")
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise ProtocolError("지원하지 않는 embedding artifact_type입니다.")
    if (
        _integer(root["schema_version"], "artifact.schema_version", 1, 1) != 1
        or _integer(
            root["feature_version"],
            "artifact.feature_version",
            FEATURE_VERSION,
            FEATURE_VERSION,
        )
        != FEATURE_VERSION
    ):
        raise ProtocolError("지원하지 않는 embedding artifact 버전입니다.")
    if not isinstance(root["embedding_model"], str) or not root["embedding_model"]:
        raise ProtocolError("artifact.embedding_model이 올바르지 않습니다.")
    embedding_dim = _integer(root["embedding_dim"], "artifact.embedding_dim", 1, 4096)
    feature_mode = root["feature_mode"]
    if feature_mode not in FEATURE_MODES:
        raise ProtocolError("artifact.feature_mode는 pure 또는 hybrid여야 합니다.")
    dense_names = root["dense_feature_names"]
    if not isinstance(dense_names, list) or not all(
        isinstance(item, str) for item in dense_names
    ):
        raise ProtocolError("artifact.dense_feature_names가 올바르지 않습니다.")
    if feature_mode == "pure" and dense_names:
        raise ProtocolError("pure 모드는 dense_feature_names가 비어 있어야 합니다.")
    if feature_mode == "hybrid" and tuple(dense_names) != hash_regex.DENSE_FEATURE_NAMES:
        raise ProtocolError("hybrid 모드의 dense feature 정의가 현재 런타임과 다릅니다.")
    if root["model_ids"] != list(MODEL_IDS):
        raise ProtocolError("artifact.model_ids가 공개 정책 모델과 다릅니다.")
    length = len(dense_names) + embedding_dim
    mean = _vector(root["feature_mean"], length, "artifact.feature_mean")
    scale = _vector(root["feature_scale"], length, "artifact.feature_scale")
    if any(item <= 0 for item in scale):
        raise ProtocolError("artifact.feature_scale은 모두 0보다 커야 합니다.")
    score_raw = _object(root["score_heads"], "artifact.score_heads")
    cost_raw = _object(root["log_cost_heads"], "artifact.log_cost_heads")
    if set(score_raw) != set(MODEL_IDS) or set(cost_raw) != set(MODEL_IDS):
        raise ProtocolError("artifact 선형 head의 모델 집합이 올바르지 않습니다.")
    safety_raw = _object(
        root["tier_safety_ratios"], "artifact.tier_safety_ratios"
    )
    if set(safety_raw) != set(TIERS):
        raise ProtocolError("artifact 등급별 안전계수가 완전하지 않습니다.")
    safety = {
        tier: _number(safety_raw[tier], f"artifact.tier_safety_ratios.{tier}")
        for tier in TIERS
    }
    if any(not 0 < value <= 1 for value in safety.values()):
        raise ProtocolError("artifact 안전계수는 0보다 크고 1 이하여야 합니다.")
    policy_id = root["policy_id"]
    policy_digest = root["policy_sha256"]
    if not isinstance(policy_id, str) or not policy_id:
        raise ProtocolError("artifact.policy_id가 올바르지 않습니다.")
    import re as _re

    if (
        not isinstance(policy_digest, str)
        or _re.fullmatch(r"[0-9a-f]{64}", policy_digest) is None
    ):
        raise ProtocolError("artifact.policy_sha256가 올바르지 않습니다.")
    training_summary = _object(root["training_summary"], "artifact.training_summary")
    return EmbeddingArtifact(
        embedding_model=root["embedding_model"],
        embedding_dim=embedding_dim,
        feature_mode=feature_mode,
        dense_feature_names=tuple(dense_names),
        feature_mean=mean,
        feature_scale=scale,
        score_heads={
            model_id: _head(score_raw[model_id], length, f"score_heads.{model_id}")
            for model_id in MODEL_IDS
        },
        log_cost_heads={
            model_id: _head(cost_raw[model_id], length, f"log_cost_heads.{model_id}")
            for model_id in MODEL_IDS
        },
        tier_safety_ratios=safety,
        policy_id=policy_id,
        policy_digest=policy_digest,
        training_summary=dict(training_summary),
    )


def load_artifact(path: Path) -> EmbeddingArtifact:
    return parse_artifact(load_json(path))


def _prediction_rows(
    standardized,
    artifact: EmbeddingArtifact,
) -> Tuple[List[Mapping[str, float]], List[Mapping[str, float]]]:
    scores_out = []
    costs_out = []
    for row in standardized:
        score_row = {}
        cost_row = {}
        for model_id in MODEL_IDS:
            score_head = artifact.score_heads[model_id]
            score_value = score_head.intercept + float(
                np.dot(np.asarray(score_head.coefficients), row)
            )
            score_row[model_id] = min(1.0, max(0.0, score_value))
            cost_head = artifact.log_cost_heads[model_id]
            log_cost = cost_head.intercept + float(
                np.dot(np.asarray(cost_head.coefficients), row)
            )
            cost_row[model_id] = math.exp(min(50.0, max(-50.0, log_cost)))
        light = cost_row[MODEL_IDS[0]]
        cost_row[MODEL_IDS[1]] = max(cost_row[MODEL_IDS[1]], light * (1.0 + 1e-12))
        cost_row[MODEL_IDS[2]] = max(
            cost_row[MODEL_IDS[2]], cost_row[MODEL_IDS[1]] * (1.0 + 1e-12)
        )
        scores_out.append(score_row)
        costs_out.append(cost_row)
    return scores_out, costs_out


def predict_batch(
    episodes: Sequence[Episode],
    artifact: EmbeddingArtifact,
    *,
    embedding_cache: Optional["np.ndarray"] = None,
) -> Tuple[Sequence[Mapping[str, float]], Sequence[Mapping[str, float]]]:
    """Run the embedding forward pass + ridge heads for a whole batch at once."""

    if np is None:
        raise RuntimeError(
            "예측에는 NumPy가 필요합니다. "
            "baselines/requirements-embedding.txt를 설치해 주세요."
        )
    raw = raw_feature_matrix(
        episodes,
        feature_mode=artifact.feature_mode,
        embedding_model=artifact.embedding_model,
        embedding_cache=embedding_cache,
    )
    mean = np.asarray(artifact.feature_mean)
    scale = np.asarray(artifact.feature_scale)
    standardized = (raw - mean) / scale
    return _prediction_rows(standardized, artifact)


def predict_episode(
    episode: Episode, artifact: EmbeddingArtifact
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    """Single-episode convenience wrapper (inefficient -- prefer predict_batch)."""

    scores, costs = predict_batch([episode], artifact)
    return scores[0], costs[0]


def make_embedding_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: EmbeddingArtifact,
    tier: str,
    *,
    embedding_cache: Optional["np.ndarray"] = None,
) -> EmbeddingPlan:
    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    if artifact.policy_id != policy.policy_id:
        raise ProtocolError("artifact와 정책의 policy_id가 다릅니다.")
    if artifact.policy_digest != policy_sha256(policy):
        raise ProtocolError("artifact와 현재 정책의 SHA-256이 다릅니다.")
    scores, costs = predict_batch(
        inputs.episodes, artifact, embedding_cache=embedding_cache
    )
    safety = artifact.tier_safety_ratios[tier]
    selected, ratio = hash_regex.select_models(
        scores,
        costs,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=safety,
    )
    fill_safety = None
    if tier == "premium":
        fill_safety = PREMIUM_AX31_FILL_SAFETY_RATIO
        selected, ratio = hash_regex.fill_ax31_upgrades(
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
    return EmbeddingPlan(
        submission=parse_submission(submission_to_dict(submission)),
        predicted_budget_ratio=ratio,
        safety_ratio=safety,
        ax31_fill_safety_ratio=fill_safety,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "학습된 sentence-embedding + ridge regression baseline 라우터 "
            "(torch/sentence-transformers 환경에서 실행해야 함)"
        )
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
        policy = (
            load_policy(args.policy)
            if args.policy is not None
            else load_bundled_policy()
        )
        artifact = load_artifact(args.artifact)
        plan = make_embedding_submission(inputs, policy, artifact, args.tier)
        write_submission_atomic(args.output, plan.submission)
    except (OSError, ProtocolError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
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
