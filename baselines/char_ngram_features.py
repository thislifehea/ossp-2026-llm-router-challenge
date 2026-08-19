# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Character 3-5gram hashed features (실험T) — drop-in alternative to hash_regex's
word 1-2gram hashing, meant to be more robust to Korean particle/ending
variation (e.g. "이야기가" vs "이야기는" share the stem "이야기" as a
character n-gram even though they're different word tokens).

Same 14 dense regex features as hash_regex.py (reused, unchanged), only the
256-dim hashed block is built differently: instead of hashing whole-word
1-2gram tokens, each token is boundary-padded ("<token>") and its character
3/4/5-grams are hashed together with the same signed FNV1a64 scheme.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hash_regex as hr
from ossp_router.protocol import Episode

_CHAR_NGRAM_SIZES = (3, 4, 5)


def _char_ngrams(token: str) -> list:
    padded = f"<{token}>"
    grams = []
    for n in _CHAR_NGRAM_SIZES:
        if len(padded) < n:
            continue
        for i in range(len(padded) - n + 1):
            grams.append(f"c{n}:{padded[i:i+n]}")
    return grams


def raw_feature_vector_char_ngram(episode: Episode, hash_bins: int) -> Tuple[float, ...]:
    features = hr.extract_features(episode)
    text = hr.episode_text(episode)
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
        float(bool(hr._FORMAL_REASONING.search(text))),
        float(bool(hr._PROGRAM_ANALYSIS.search(text))),
        math.log1p(len(hr._MULTI_CONSTRAINT.findall(text))),
        float(bool(hr._SIMPLE_TRANSFORM.search(text))),
    )
    bins = [0.0] * hash_bins
    tokens = hr._normalized_tokens(text)
    hashed_features = []
    for token in tokens:
        hashed_features.extend(_char_ngrams(token))
    for value in hashed_features:
        digest = hr._stable_hash(value)
        index = digest & (hash_bins - 1)
        bins[index] += -1.0 if digest & (1 << 63) else 1.0
    norm = math.sqrt(sum(value * value for value in bins))
    if norm:
        bins = [value / norm for value in bins]
    return dense + tuple(bins)
