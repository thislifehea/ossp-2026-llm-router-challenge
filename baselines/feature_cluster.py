# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""UniRoute(Jitkrittum et al. 2025) 스타일 클러스터 기반 라우터.

hash_regex.py의 270차원(dense 14 + hash 256bin) 특징을 그대로 쓰되, 모델별로
회귀계수를 학습하는 대신 K-means로 프롬프트를 K개 클러스터로 묶고 "이 클러스터에서
각 모델의 평균 score/log-cost가 얼마였는가"만 조회한다. 유효 파라미터 수가
K*3*2개뿐이라(K=32면 192개) hash_regex(ridge, 256bins 기준 2166개)보다 훨씬 적어
Train 1,760개 규모에서 과적합 위험이 낮다는 가설을 검증하기 위한 실험용 baseline.

torch/sentence-transformers 기반 진짜 의미 임베딩은 이 환경에서 torch DLL 로드가
깨져서(Windows, WinError 127) 못 씀 - 이 스크립트는 그 대안으로, "클러스터링 자체가
과적합을 줄여주는가"라는 가설만 분리해서 테스트한다. 실제 배포용이 아니라 실험용.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import hash_regex
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    InputBatch,
    OutcomeBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_json,
    load_outcomes,
    load_policy,
    parse_submission,
    policy_sha256,
    submission_to_dict,
    write_json,
)
from ossp_router.scoring import score_submissions

ARTIFACT_TYPE = "ossp-feature-cluster-v1"
DEFAULT_HASH_BINS = 256
DEFAULT_K = 32
PREMIUM_AX31_FILL_SAFETY_RATIO = hash_regex.PREMIUM_AX31_FILL_SAFETY_RATIO
_RNG_SEED = 0


def _kmeans(matrix: np.ndarray, k: int, *, seed: int, iterations: int = 100) -> np.ndarray:
    """numpy만으로 짠 표준 Lloyd's algorithm K-means. 반환값: (k, dim) 중심점."""

    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    centroid_idx = rng.choice(n, size=k, replace=False)
    centroids = matrix[centroid_idx].copy()
    for _ in range(iterations):
        distances = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        new_centroids = centroids.copy()
        for cluster in range(k):
            members = matrix[assignments == cluster]
            if len(members) > 0:
                new_centroids[cluster] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def _assign(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def _cluster_stats(
    assignments: np.ndarray,
    scores: np.ndarray,
    log_costs: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """클러스터별 모델별 평균 score/log-cost. 빈 클러스터는 전역 평균으로 fallback."""

    global_score = scores.mean(axis=0)
    global_log_cost = log_costs.mean(axis=0)
    score_stats = np.tile(global_score, (k, 1))
    cost_stats = np.tile(global_log_cost, (k, 1))
    for cluster in range(k):
        mask = assignments == cluster
        if mask.any():
            score_stats[cluster] = scores[mask].mean(axis=0)
            cost_stats[cluster] = log_costs[mask].mean(axis=0)
    return score_stats, cost_stats


def predict_episode(
    episode: Episode,
    *,
    hash_bins: int,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    centroids: np.ndarray,
    cluster_score_stats: np.ndarray,
    cluster_log_cost_stats: np.ndarray,
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    raw = np.asarray(hash_regex.raw_feature_vector(episode, hash_bins), dtype=np.float64)
    standardized = (raw - feature_mean) / feature_scale
    distances = ((centroids - standardized[None, :]) ** 2).sum(axis=1)
    cluster = int(distances.argmin())
    score_row = cluster_score_stats[cluster]
    cost_row = cluster_log_cost_stats[cluster]
    scores = {
        model_id: float(min(1.0, max(0.0, score_row[index])))
        for index, model_id in enumerate(MODEL_IDS)
    }
    costs = {
        model_id: math.exp(min(50.0, max(-50.0, float(cost_row[index]))))
        for index, model_id in enumerate(MODEL_IDS)
    }
    light = costs[MODEL_IDS[0]]
    costs[MODEL_IDS[1]] = max(costs[MODEL_IDS[1]], light * (1.0 + 1e-12))
    costs[MODEL_IDS[2]] = max(costs[MODEL_IDS[2]], costs[MODEL_IDS[1]] * (1.0 + 1e-12))
    return scores, costs


def _outcome_cost(outcome, policy: RoutingPolicy) -> float:
    from decimal import Decimal

    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    cost = (
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )
    return float(cost)


def _training_matrix(
    inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy, hash_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcome_index = {
        (outcome.episode_id, outcome.model_id): outcome for outcome in outcomes.outcomes
    }
    matrix = np.asarray(
        [hash_regex.raw_feature_vector(episode, hash_bins) for episode in inputs.episodes],
        dtype=np.float64,
    )
    scores = []
    log_costs = []
    for episode in inputs.episodes:
        rows = [outcome_index[(episode.episode_id, model_id)] for model_id in MODEL_IDS]
        scores.append([float(row.score) for row in rows])
        log_costs.append([math.log(_outcome_cost(row, policy)) for row in rows])
    return matrix, np.asarray(scores, dtype=np.float64), np.asarray(log_costs, dtype=np.float64)


def _submission(inputs: InputBatch, policy: RoutingPolicy, tier: str, selected: Sequence[str]) -> Submission:
    return Submission(
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


def _score_one_tier(inputs, outcomes, policy, tier, selected):
    all_light = tuple(policy.light_model_id for _ in inputs.episodes)
    submissions = [
        _submission(inputs, policy, candidate, selected if candidate == tier else all_light)
        for candidate in TIERS
    ]
    return score_submissions(inputs, outcomes, submissions, policy)["tiers"][tier]


def _calibrate_safety_ratios(
    inputs, outcomes, policy, predicted_scores, predicted_costs, grid_size
):
    calibrated: Dict[str, float] = {}
    reports: Dict[str, Any] = {}
    for tier in TIERS:
        minimum = 1.0 / float(policy.tiers[tier].budget_multiplier)
        candidates = (
            [min(1.0, minimum)]
            if grid_size <= 1
            else [minimum + (1.0 - minimum) * i / (grid_size - 1) for i in range(grid_size)]
        )
        best = None
        for safety in candidates:
            selected, ratio = hash_regex.select_models(
                predicted_scores,
                predicted_costs,
                budget_multiplier=float(policy.tiers[tier].budget_multiplier),
                safety_ratio=safety,
            )
            report = _score_one_tier(inputs, outcomes, policy, tier, selected)
            if not report["budget_passed"]:
                continue
            from decimal import Decimal as D

            rank = (D(report["tier_score"]), -D(report["budget_ratio"]), -D(str(safety)))
            if best is None or rank > best[0]:
                best = (rank, safety, ratio, report)
        if best is None:
            raise RuntimeError(f"{tier} 예산을 통과하는 안전계수가 없습니다.")
        calibrated[tier] = best[1]
        reports[tier] = {
            "safety_ratio": best[1],
            "predicted_budget_ratio": best[2],
            "actual_budget_ratio": best[3]["budget_ratio"],
            "tier_score": best[3]["tier_score"],
            "budget_passed": best[3]["budget_passed"],
        }
    return calibrated, reports


def train(
    *,
    input_path: Path,
    outcomes_path: Path,
    validation_input_path: Path,
    validation_outcomes_path: Path,
    artifact_path: Path,
    report_path: Path,
    policy: RoutingPolicy,
    hash_bins: int,
    k: int,
    safety_grid_size: int,
) -> Mapping[str, Any]:
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    matrix, scores, log_costs = _training_matrix(inputs, outcomes, policy, hash_bins)

    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    standardized = (matrix - feature_mean) / feature_scale

    centroids = _kmeans(standardized, k, seed=_RNG_SEED)
    assignments = _assign(standardized, centroids)
    cluster_score_stats, cluster_log_cost_stats = _cluster_stats(assignments, scores, log_costs, k)

    cluster_sizes = [int((assignments == c).sum()) for c in range(k)]

    predicted_scores = []
    predicted_costs = []
    for episode in inputs.episodes:
        s, c = predict_episode(
            episode,
            hash_bins=hash_bins,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            centroids=centroids,
            cluster_score_stats=cluster_score_stats,
            cluster_log_cost_stats=cluster_log_cost_stats,
        )
        predicted_scores.append(s)
        predicted_costs.append(c)

    validation_inputs = load_input(validation_input_path)
    validation_outcomes = load_outcomes(validation_outcomes_path)
    val_predicted_scores = []
    val_predicted_costs = []
    for episode in validation_inputs.episodes:
        s, c = predict_episode(
            episode,
            hash_bins=hash_bins,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            centroids=centroids,
            cluster_score_stats=cluster_score_stats,
            cluster_log_cost_stats=cluster_log_cost_stats,
        )
        val_predicted_scores.append(s)
        val_predicted_costs.append(c)

    safety_ratios, validation_reports = _calibrate_safety_ratios(
        validation_inputs,
        validation_outcomes,
        policy,
        val_predicted_scores,
        val_predicted_costs,
        safety_grid_size,
    )

    artifact_value = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": 1,
        "hash_bins": hash_bins,
        "k": k,
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "centroids": centroids.tolist(),
        "cluster_score_stats": cluster_score_stats.tolist(),
        "cluster_log_cost_stats": cluster_log_cost_stats.tolist(),
        "tier_safety_ratios": {tier: float(safety_ratios[tier]) for tier in TIERS},
        "training_summary": {
            "num_episodes": len(inputs.episodes),
            "k": k,
            "hash_bins": hash_bins,
            "cluster_sizes": cluster_sizes,
            "min_cluster_size": min(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
        },
    }
    write_json(artifact_path, artifact_value)

    submissions = []
    for tier in TIERS:
        selected, _ratio = hash_regex.select_models(
            val_predicted_scores,
            val_predicted_costs,
            budget_multiplier=float(policy.tiers[tier].budget_multiplier),
            safety_ratio=safety_ratios[tier],
        )
        if tier == "premium":
            selected, _ratio = hash_regex.fill_ax31_upgrades(
                selected,
                val_predicted_scores,
                val_predicted_costs,
                budget_multiplier=float(policy.tiers[tier].budget_multiplier),
                safety_ratio=PREMIUM_AX31_FILL_SAFETY_RATIO,
            )
        submissions.append(_submission(validation_inputs, policy, tier, selected))
    final_report = score_submissions(validation_inputs, validation_outcomes, submissions, policy)

    report = {
        "report_type": "ossp-feature-cluster-training-v1",
        "training_summary": artifact_value["training_summary"],
        "validation_safety_calibration": validation_reports,
        "validation_self_check": final_report,
    }
    write_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="hash_regex 특징 + K-means 클러스터 기반 라우터 학습")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--validation-outcomes", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--hash-bins", type=int, default=DEFAULT_HASH_BINS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--safety-grid-size", type=int, default=61)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
    report = train(
        input_path=args.input,
        outcomes_path=args.outcomes,
        validation_input_path=args.validation_input,
        validation_outcomes_path=args.validation_outcomes,
        artifact_path=args.artifact,
        report_path=args.report,
        policy=policy,
        hash_bins=args.hash_bins,
        k=args.k,
        safety_grid_size=args.safety_grid_size,
    )
    print(f"OK: feature-cluster artifact 생성. final_score={report['validation_self_check']['final_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
