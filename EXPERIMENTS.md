# 정책 실험 로그

`.claude/skills/router-policy-experiment/SKILL.md` 절차로 `ossp_router.cli self-check`를
돌린 결과를 실험할 때마다 한 줄씩 추가한다. 공식 채점 하네스(`src/ossp_router/scoring.py`)를
그대로 쓰므로 여기 적힌 수치는 (materialize된 데이터 기준이라면) 실제 채점 방식과 동일하다 —
단 채점용 비공개 평가셋과 공개 Dev는 분포가 다를 수 있으니 "공개 Dev 대비 baseline과의
방향"으로 우선 읽을 것.

| 날짜 | 정책 | split | Fast (score/budget_ratio) | Balanced | Premium | final_score | 비고 |
|---|---|---|---|---|---|---|---|
| — | `baselines/always_light.py` | dev | 0.619318 / 1.000000 | 0.619318 / 1.000000 | 0.619318 / 1.000000 | 0.619318 | 공식 baseline (참고, `baselines/README.md`) |
| — | `baselines/prompt_heuristic.py` | dev | 0.625852 / 1.072334 | 0.658239 / 1.367866 | 0.691761 / 2.102044 | 0.655341 | 공식 baseline (참고) |
| — | `baselines/feature_budget.py` | dev | 0.621023 / 1.038210 | 0.623580 / 1.334059 | 0.691761 / 2.102044 | 0.643011 | 공식 baseline (참고) |
| — | `baselines/hash_regex.py` | dev | 0.663068 / 1.235989 | 0.693750 / 1.961506 | 0.740057 / 3.985205 | 0.695369 | 공식 baseline (참고, 현재 최고 — 목표 기준선) |
| 2026-08-12 | 실험A: hash_regex, hash_bins 256→1024 (재학습) | dev | 0.655682 / 1.249820 | 0.676420 / 1.973803 | 0.726136 / 3.965749 | 0.683040 | baseline보다 전 tier 악화(-0.0123 final). 차원(1038)이 Train 샘플 수(1760) 대비 과대해져 ridge(alpha=100 자동선택)로도 과적합/노이즈 못 잡은 것으로 추정. **채택 보류.** |
| 2026-08-12 | 실험B: hash_regex 공식 아티팩트, premium safety_ratio만 0.925→0.897(×0.97) | dev | 0.663068 / 1.235989 (fast/balanced 원본 그대로) | 0.693750 / 1.961506 | 0.737216 / 3.825778 | 0.694517 | final은 baseline 대비 -0.00085로 거의 동일, **premium 예산 여유는 3.985→3.826로 유의미하게 확보**(한도 4.0에 근접했던 리스크 완화). quality 손실 대비 안전마진 이득이 커 보임 — **채택 후보.** |
