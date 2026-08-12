---
name: parallel-policy-sweep
description: 서로 독립적인 후보 라우터 정책(select_model 로직)이 2개 이상 있을 때, 후보별로 격리된 서브에이전트를 병렬로 띄워 구현+테스트+self-check 채점을 동시에 돌리는 절차. "후보 정책 여러 개 한번에 비교해줘", "hash_regex 변형 여러 개 병렬로 실험해줘" 같은 요청에 사용. 후보가 하나뿐이면 이 스킬 대신 router-policy-experiment를 쓴다.
---

# 정책 후보 병렬 스윕

`router-policy-experiment` 절차([[router-policy-experiment]])는 정책 하나를 순차로
구현→검증하는 루프다. 이 스킬은 그 루프를 후보 개수만큼 동시에 돌리기 위한 것 —
같은 파일을 건드리지 않는 독립적인 아이디어(다른 스크립트, 다른 파라미터 세트)일 때만 쓴다.

## 언제 쓰나

- 후보가 2개 이상이고, 서로 다른 파일(예: `baselines/hash_regex_v2.py`,
  `baselines/embedding_knn.py`)로 구현 가능하며, 서로의 코드를 참조하지 않는다.
- 후보 중 하나라도 `src/ossp_router/heuristic.py`처럼 기존 공용 파일을 직접 고치는
  방식이면 쓰지 않는다 — worktree가 갈라져도 머지 시 의미 충돌이 나므로 순차 처리.

## 절차

1. **후보 목록 확정**: 후보마다 (a) 아이디어 한 줄, (b) 담을 파일 경로, (c) 학습이
   필요한지(`train_hash_regex.py` 같은 오프라인 학습 스텝 필요 여부) 정리.

2. **후보마다 Agent를 `isolation: "worktree"`로 병렬 호출** — 한 메시지에 여러 Agent
   호출을 동시에 담는다. 각 서브에이전트 프롬프트에 반드시 포함:
   - `router-policy-experiment` SKILL.md 절차를 그대로 따를 것 (구현 → unittest →
     `data/materialized/dev/inputs.json` + `data/dev/outcomes.json`으로 self-check →
     자기 worktree의 `EXPERIMENTS.md`에 기록)
   - 구현할 정책 아이디어와 목표 파일 경로
   - "하지 말아야 할 것": `src/ossp_router/protocol.py`/`scoring.py`/`cli.py`/
     `schemas/*.json`/`configs/routing-policy.v1.json` 수정 금지, episode_id/split/입력
     순서를 선택 근거로 사용 금지, 추론 시점에 outcome 참조 금지
   - 보고 형식: tier별 `tier_score`/`budget_ratio`/`budget_passed`, `final_score`,
     baseline(`baselines/README.md` 공개 Dev 표) 대비 방향, 한 일 한 줄 요약
   - 데이터 준비(`.venv-data` + `tools/materialize_public_data.py`)가 worktree마다
     따로 필요할 수 있음 — 서브에이전트가 `data/materialized/`가 없으면 먼저 생성하게
     안내(용량이 있으니 매번 새로 받는 대신 가능하면 공유 캐시 재사용을 시도해도 됨).

3. **결과가 모두 돌아오면 사람이 보는 대화에서 직접 표로 비교**한다. 서브에이전트
   tool 출력을 그대로 컨텍스트에 끌어오지 말고, 보고받은 숫자만 요약한다.
   **`budget_passed`가 `false`인 tier가 하나라도 있는 후보는 그 tier가 0점이라 1차
   탈락** — quality가 아무리 좋아도 예산 초과면 의미 없음.

4. **채택할 후보를 고르면 메인 세션에서 그 정책 파일을 다시 적용**한다. 서브에이전트는
   각자의 worktree 브랜치에 커밋을 남기므로, 채택된 것만 diff를 가져오거나 내용을
   그대로 옮겨 적는다 (여러 worktree를 머지하지 않는다 — 탈락한 후보의 브랜치는 그냥
   버려둔다).

5. **채택된 정책의 실험 결과만 메인 브랜치 `EXPERIMENTS.md`에 옮겨 적는다.** 기각된
   후보는 기록을 옮기지 않아도 되지만, 왜 버렸는지(대개 예산 초과 또는 baseline보다
   낮은 tier_score) 한 줄은 대화 요약에 남긴다.

## 하지 말아야 할 것

- 서로 같은 파일을 고쳐야 하는 후보를 병렬로 돌리지 않는다.
- 서브에이전트가 공식 하네스(`protocol.py`/`scoring.py`/`schemas/`) 또는
  `configs/routing-policy.v1.json`을 건드리게 하지 않는다 — 프롬프트에 명시적으로
  금지 사항을 포함시킨다.
- 탈락한 후보의 worktree/브랜치를 머지하지 않는다.
