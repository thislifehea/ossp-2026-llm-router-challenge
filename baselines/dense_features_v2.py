# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""실험AA: hash_regex.py의 14개 dense 정규식 특징을 확장. 대회 공개 데이터
소스 구성(DATA_LICENSES.md: Belebele Korean/HRMCR/GSM8K/DeepMind Mathematics/
AIME/TruthfulQA/RuleTaker/CRUXEval/BABILong)을 참고해, 각 도메인을 더 세밀히
구분할 만한 저비용 정규식/휴리스틱 신호 8개를 추가.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hash_regex as hr
from ossp_router.protocol import Episode

_CHOICE_MARKER = re.compile(r"(?:^|\n|\s)[A-Da-d1-4가-라][\)\.]\s", re.MULTILINE)
_BINARY_ANSWER = re.compile(
    r"\b(?:true|false|yes|no)\b|참|거짓|맞다|틀리다", re.IGNORECASE
)
_EQUATION_SIGN = re.compile(r"=")
_CODE_SIGNATURE = re.compile(
    r"\bdef\s+\w+\s*\(|\bclass\s+\w+|\breturn\b|->", re.IGNORECASE
)
_RULE_IMPLICATION = re.compile(r"\bif\b.{0,40}\bthen\b|→|=>|만약.{0,20}(?:이면|라면)", re.IGNORECASE)
_COMPETITION_MATH = re.compile(
    r"\bfind the (?:number|value|sum|product)|\bhow many\b|\bcompute\b|\bdetermine\b|"
    r"몇\s*개|몇\s*가지|구하시오|구하라",
    re.IGNORECASE,
)
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}\b")
_QUESTION_MARK = re.compile(r"\?")


def extended_dense_features(episode: Episode) -> Tuple[float, ...]:
    features = hr.extract_features(episode)
    text = hr.episode_text(episode)
    base = (
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
        float(bool(hr._FORMAL_REASONING.search(text))),
        float(bool(hr._PROGRAM_ANALYSIS.search(text))),
        math.log1p(len(hr._MULTI_CONSTRAINT.findall(text))),
        float(bool(hr._SIMPLE_TRANSFORM.search(text))),
    )
    length = max(1, features.character_count)
    sentence_count = max(1, features.sentence_count)
    new = (
        math.log1p(len(_CHOICE_MARKER.findall(text))),
        float(bool(_BINARY_ANSWER.search(text))),
        len(_EQUATION_SIGN.findall(text)) / length,
        math.log1p(len(_CODE_SIGNATURE.findall(text))),
        math.log1p(len(_RULE_IMPLICATION.findall(text))),
        float(bool(_COMPETITION_MATH.search(text))),
        len(_PROPER_NOUN.findall(text)) / length,
        len(_QUESTION_MARK.findall(text)) / sentence_count,
    )
    return base + new


EXTENDED_DENSE_FEATURE_NAMES = (
    "log_character_count", "log_word_count", "log_sentence_count", "log_message_count",
    "hangul_ratio", "log_code_marker_count", "log_math_marker_count", "numeric_density",
    "long_context", "log_reasoning_marker_count", "formal_reasoning", "program_analysis",
    "log_multi_constraint_count", "simple_transform",
    "log_choice_marker_count", "binary_answer_marker", "equation_density",
    "log_code_signature_count", "log_rule_implication_count", "competition_math_marker",
    "proper_noun_density", "question_mark_density",
)


def raw_feature_vector_extended(episode: Episode, hash_bins: int) -> Tuple[float, ...]:
    dense = extended_dense_features(episode)
    text = hr.episode_text(episode)
    bins = [0.0] * hash_bins
    tokens = hr._normalized_tokens(text)
    hashed_features = [f"w1:{token}" for token in tokens]
    hashed_features.extend(f"w2:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:]))
    for value in hashed_features:
        digest = hr._stable_hash(value)
        index = digest & (hash_bins - 1)
        bins[index] += -1.0 if digest & (1 << 63) else 1.0
    norm = math.sqrt(sum(value * value for value in bins))
    if norm:
        bins = [value / norm for value in bins]
    return dense + tuple(bins)
