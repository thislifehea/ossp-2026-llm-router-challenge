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
