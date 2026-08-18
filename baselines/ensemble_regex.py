# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Average hash_regex(ridge) and gbm_regex predictions, route with the shared budget optimizer.

Exploratory script (not a submission-ready baseline): reuses hash_regex's
raw_feature_vector/select_models/fill_ax31_upgrades verbatim, only the
per-model score/cost prediction is an ensemble (simple mean) of the two
already-trained artifacts.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import gbm_regex
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
    load_outcomes,
    load_policy,
    parse_submission,
    submission_to_dict,
)
from ossp_router.heuristic import write_submission_atomic
from ossp_router.scoring import score_submissions


def predict_episode(
    episode: Episode,
    ridge_artifact: hash_regex.HashRegexArtifact,
    gbm_artifact: gbm_regex.GbmArtifact,
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    r_scores, r_costs = hash_regex.predict_episode(episode, ridge_artifact)
    g_scores, g_costs = gbm_regex.predict_episode(episode, gbm_artifact)
    scores = {m: (r_scores[m] + g_scores[m]) / 2.0 for m in MODEL_IDS}
    costs = {m: (r_costs[m] + g_costs[m]) / 2.0 for m in MODEL_IDS}
    return scores, costs


def predict_all(
    inputs: InputBatch,
    ridge_artifact: hash_regex.HashRegexArtifact,
    gbm_artifact: gbm_regex.GbmArtifact,
) -> Tuple[Sequence[Mapping[str, float]], Sequence[Mapping[str, float]]]:
    predictions = [
        predict_episode(episode, ridge_artifact, gbm_artifact)
        for episode in inputs.episodes
    ]
    scores = [item[0] for item in predictions]
    costs = [item[1] for item in predictions]
    return scores, costs


def select_from_predictions(
    inputs: InputBatch,
    policy: RoutingPolicy,
    scores: Sequence[Mapping[str, float]],
    costs: Sequence[Mapping[str, float]],
    tier: str,
    safety_ratio: float,
) -> Tuple[Submission, float]:
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


def make_ensemble_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    ridge_artifact: hash_regex.HashRegexArtifact,
    gbm_artifact: gbm_regex.GbmArtifact,
    tier: str,
    safety_ratio: float,
) -> Tuple[Submission, float]:
    scores, costs = predict_all(inputs, ridge_artifact, gbm_artifact)
    return select_from_predictions(inputs, policy, scores, costs, tier, safety_ratio)


def _safety_grid(minimum: float, size: int = 41) -> Sequence[float]:
    if minimum >= 1.0:
        return (1.0,)
    return tuple(minimum + (1.0 - minimum) * i / (size - 1) for i in range(size))


def _score_one_tier(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    tier: str,
    submission: Submission,
    all_light_submissions: Mapping[str, Submission],
) -> Mapping[str, object]:
    ordered = [
        submission if candidate == tier else all_light_submissions[candidate]
        for candidate in TIERS
    ]
    return score_submissions(inputs, outcomes, ordered, policy)["tiers"][tier]


def calibrate_safety_ratios(
    dev_inputs: InputBatch,
    dev_outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    ridge_artifact: hash_regex.HashRegexArtifact,
    gbm_artifact: gbm_regex.GbmArtifact,
) -> Mapping[str, Mapping[str, object]]:
    all_light = tuple(policy.light_model_id for _ in dev_inputs.episodes)
    all_light_submissions = {
        tier: parse_submission(
            submission_to_dict(
                Submission(
                    schema_version=dev_inputs.schema_version,
                    challenge_id=dev_inputs.challenge_id,
                    policy_id=policy.policy_id,
                    split=dev_inputs.split,
                    tier=tier,
                    decisions=tuple(
                        Decision(episode.episode_id, model_id)
                        for episode, model_id in zip(dev_inputs.episodes, all_light)
                    ),
                )
            )
        )
        for tier in TIERS
    }
    scores, costs = predict_all(dev_inputs, ridge_artifact, gbm_artifact)
    results: dict[str, Mapping[str, object]] = {}
    for tier in TIERS:
        minimum = 1.0 / float(policy.tiers[tier].budget_multiplier)
        best = None
        for safety in _safety_grid(minimum):
            submission, ratio = select_from_predictions(
                dev_inputs, policy, scores, costs, tier, safety
            )
            report = _score_one_tier(
                dev_inputs, dev_outcomes, policy, tier, submission, all_light_submissions
            )
            if not report["budget_passed"]:
                continue
            rank = (
                Decimal(str(report["tier_score"])),
                -Decimal(str(report["budget_ratio"])),
                -Decimal(str(safety)),
            )
            if best is None or rank > best[0]:
                best = (rank, safety, ratio, report)
        if best is None:
            raise RuntimeError(f"{tier}: Dev 예산을 통과하는 안전계수를 못 찾음")
        results[tier] = {
            "safety_ratio": best[1],
            "predicted_ratio": best[2],
            "tier_score": best[3]["tier_score"],
            "budget_ratio": best[3]["budget_ratio"],
            "budget_passed": best[3]["budget_passed"],
        }
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="릿지+GBM 앙상블 라우터")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--ridge-artifact", type=Path, required=True)
    parser.add_argument("--gbm-artifact", type=Path, required=True)
    parser.add_argument("--safety-ratio", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
        ridge_artifact = hash_regex.load_artifact(args.ridge_artifact)
        gbm_artifact = gbm_regex.load_artifact(args.gbm_artifact)
        submission, ratio = make_ensemble_submission(
            inputs, policy, ridge_artifact, gbm_artifact, args.tier, args.safety_ratio
        )
        write_submission_atomic(args.output, submission)
    except (OSError, ProtocolError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} 제출 파일 생성 (예측 비용 비율 {ratio:.6f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
