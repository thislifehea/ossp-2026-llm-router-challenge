---
name: router-policy-experiment
description: Efficient LLM Routing Challenge 라우터에서 새 모델 선택 정책(select_model 로직)을 구현하거나 튜닝할 때 따르는 절차. "새 라우터 정책 실험해줘", "hash_regex 개선해줘", "임계값 튜닝해줘" 같은 요청에 사용.
---

# 라우터 정책 실험 워크플로우

공식 채점 하네스(`src/ossp_router/cli.py self-check` + `scoring.py`)는 고정된 채점기다.
`configs/routing-policy.v1.json`(tier 예산/가중치/모델 요율)도 고정값이다. 우리가 갈아끼우는
건 **모델 선택 로직**(`select_model` 계열 함수)뿐이고, 같은 하네스로 점수를 비교하는 게
기본 개발 루프다 (`CLAUDE.md` "핵심 경계" 참고).

## 절차

1. **정책 구현**: `src/ossp_router/heuristic.py`의 `select_model`을 바꾸거나, `baselines/`의
   기존 구현(`prompt_heuristic.py`, `feature_budget.py`, `hash_regex.py`)을 참고해 새 스크립트를
   만든다. 시그니처는 **"프롬프트/messages 내용 + tier → model_id"** 뿐이어야 한다.
   - `episode_id`, `split`, `challenge_id`, 입력 순서를 선택 근거로 쓰지 않는다(순서/ID를
     바꿔도 같은 출력이 나와야 함 — `tests/test_prompt_heuristic.py`의 결정성 검사 참고).
   - 모델 답변 본문, 정답, quality/outcome 정보는 런타임 입력에 없다 — 학습(오프라인) 단계에서
     `data/train/outcomes.json`으로 파라미터를 미리 학습해서 아티팩트로 저장하는 건 되지만
     (`hash_regex.py` + `train_hash_regex.py` 패턴), 추론 시점에 outcome을 읽으면 안 된다.
   - 반환값은 `schemas/submission.v1.schema.json`을 만족하는 `model_id`(`ax31-light`/`ax31`/
     `axk1-think`) 하나.

2. **테스트 실행**: 코드를 손대기 전/후로 항상 기존 테스트가 멀쩡한지 확인.
   ```console
   PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
   ```

3. **공식 self-check로 baseline과 비교**: 세 tier(`fast`/`balanced`/`premium`) 각각 제출
   JSON을 만든 뒤, 공식 도구로 채점한다.
   ```console
   for tier in fast balanced premium; do
     PYTHONPATH=src python3 <내_정책_스크립트>.py \
       --input data/materialized/dev/inputs.json \
       --tier "$tier" \
       --output "build/<정책이름>/$tier.json"
   done

   PYTHONPATH=src python3 -m ossp_router.cli self-check \
     --input data/materialized/dev/inputs.json \
     --outcomes data/dev/outcomes.json \
     --submissions build/<정책이름> \
     --report build/<정책이름>-report.json
   ```
   리포트의 `tiers.<tier>.tier_score`(예산 통과 시 quality, 초과 시 0), `budget_passed`,
   `budget_ratio`, 최상위 `final_score`를 확인한다. **`budget_passed`가 `false`인 tier는
   `tier_score`가 무조건 0**이라 quality가 아무리 좋아도 소용없다 — 예산 여유를 항상 남겨야
   함(공식 hash_regex baseline도 채점용 비공개 데이터에서 Premium 예산을 넘겨 그 tier가
   0점 처리된 전례 있음, `baselines/README.md` 참고).

4. **저예산 tier를 우선 판단 기준으로 삼는다**: tier 가중치는 fast **0.4** / balanced
   **0.3** / premium **0.3**(`configs/routing-policy.v1.json`) — Fast가 가장 크지만
   Balanced/Premium도 동일 비중이라 셋 다 무시할 수 없다. `final_score` 하나만 보지 말고
   tier별 `tier_score`와 `budget_ratio`를 baseline(`baselines/README.md`의 공개 Dev 표,
   현재 최고는 hash_regex 가중 최종 0.6954)과 나란히 비교.

5. **결과를 기록**: 저장소 루트 `EXPERIMENTS.md`(없으면 새로 만들 것 — 표 형식은 tier별
   score/budget_ratio/budget_passed/final_score, 정책명, 비고)에 한 줄 추가. 이건 우리가
   만든 개발 로그일 뿐 공식 스키마 파일이 아니므로 자유롭게 관리해도 된다.

## 하지 말아야 할 것

- `src/ossp_router/protocol.py`, `scoring.py`, `cli.py`, `schemas/*.json`,
  `configs/routing-policy.v1.json`을 정책 튜닝 목적으로 수정하지 않는다. 공식 스키마는
  모르는 필드를 거부하므로 여기 손대면 self-check/제출 검증 자체가 깨진다.
- `episode_id`/`split`/`challenge_id`/입력 순서를 모델 선택에 사용하지 않는다.
- 추론 시점에 outcome/정답/모델 답변 본문을 참조하지 않는다 (학습 단계 오프라인 사용은 허용).
- `data/train/`, `data/dev/`의 outcome 자체를 수정하지 않는다.
