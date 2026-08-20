# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""BPE subword hashed features (실험W) -- drop-in alternative to hash_regex's
word 1-2gram hashing and char_ngram_features's char 3/5gram hashing. Uses a
pretrained public BPE tokenizer (klue/roberta-base, byte-level BPE trained on
Korean corpora) to split text into morpheme-aware subwords (e.g. "이야기가"
-> "이야기" + "##가"), then hashes those subword unigrams/bigrams with the
same signed FNV1a64 scheme as hash_regex.py.

Offline/training-only module (mirrors embedding_gbm_regex.py's pattern) --
NOT wired into the production container. Loading the tokenizer requires
network access on first use.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hash_regex as hr
from ossp_router.protocol import Episode

_TOKENIZER_NAME = "klue/roberta-base"
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer

        _tokenizer = Tokenizer.from_pretrained(_TOKENIZER_NAME)
    return _tokenizer


def _bpe_tokens(text: str) -> list:
    tok = _get_tokenizer()
    pieces = tok.encode(text).tokens
    return [p for p in pieces if p not in ("[CLS]", "[SEP]", "[PAD]", "[UNK]")]


def raw_feature_vector_bpe(episode: Episode, hash_bins: int) -> Tuple[float, ...]:
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
    pieces = _bpe_tokens(text)
    hashed_features = [f"b1:{p}" for p in pieces]
    hashed_features.extend(f"b2:{a}\x1f{b}" for a, b in zip(pieces, pieces[1:]))
    for value in hashed_features:
        digest = hr._stable_hash(value)
        index = digest & (hash_bins - 1)
        bins[index] += -1.0 if digest & (1 << 63) else 1.0
    norm = math.sqrt(sum(value * value for value in bins))
    if norm:
        bins = [value / norm for value in bins]
    return dense + tuple(bins)
