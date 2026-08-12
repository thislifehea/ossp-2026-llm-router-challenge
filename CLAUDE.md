# CLAUDE.md

Efficient LLM Routing Challenge (SKT 2026 오픈소스 개발자대회) — 공식 대회 저장소
(`sktelecom/ossp-2026-llm-router-challenge`)를 fork한 실제 제출용 저장소.
`origin` = `thislifehea/ossp-2026-llm-router-challenge`, `upstream` = 공식 원본.

이전에 독립적으로 개발하던 `LLM_choice_Router`(GitHub: thislifehea/LLM_choice_Router)는
초기 설계/실험 기록 참고용으로 그대로 남아있고, **실제 구현·제출은 이 저장소에서** 한다 —
공식 규칙 확인 결과 원래 가정(캐스케이드/순차 호출)이 대회 규칙과 맞지 않아 이 저장소로
전환했다.

## 문제 정의 (필독 — 예전 캐스케이드 가정은 폐기)

라우터는 모델을 직접 호출하지 않는다. 문항(`episode`)마다, tier(`fast`/`balanced`/`premium`)
마다 프롬프트/messages 내용만 보고 세 모델(`ax31-light`/`ax31`/`axk1-think`) 중 하나를
**정확히 1회** 선택한다. 순차 호출, 재시도, 모델 교체, 여러 답변 비교는 전부 금지 —
history/remaining_budget 같은 런타임 상태 자체가 없다.

모델 선택에 쓸 수 있는 건 프롬프트/messages 내용에서 직접 계산한 정보(길이, 정규식,
n-gram, 해시, 임베딩)뿐. `episode_id`/`split`/`challenge_id`/입력 순서는 선택 근거로 쓰면
안 된다(순서·ID를 바꿔도 같은 출력이 나와야 검증을 통과함).

## 핵심 경계 — 공식 하네스, 함부로 안 건드림

- `src/ossp_router/protocol.py`, `scoring.py`, `runtime.py`, `cli.py`, `schemas/*.json` —
  공식 채점/검증 로직. 스키마는 모르는 필드가 있으면 거부하므로 여기 손대면 self-check나
  제출 검증 자체가 깨진다.
- `configs/routing-policy.v1.json` — 공식 tier 예산/가중치/모델별 토큰 요율. **값을 바꾸는
  게 아니라 이 값을 알고 정책을 최적화하는 게 우리 일.**
- `data/train/`, `data/dev/`(outcomes) — 공식 제공 라벨. `data/materialized/`는 gitignore된
  산출물(`tools/materialize_public_data.py`로 재생성, 최초 1회 필요).
- 이 저장소는 upstream에 기여(PR)를 받지 않는 정책(`CONTRIBUTING.md`) — 우리 fork 안에서만
  자유롭게 수정, upstream에 PR 시도하지 않는다.

## 우리가 실제로 만드는 것

- `src/ossp_router/heuristic.py`의 `select_model` 교체, 또는 `baselines/`에 새 구현 추가.
  시그니처는 **"프롬프트/messages 내용 + tier → model_id"** 뿐 — history/budget 파라미터
  없음(예전 `router/interfaces.py`의 `Policy.decide(...)` 시그니처는 여기서 안 맞음).
- 공식 baseline 4종(`always_light`, `prompt_heuristic`, `feature_budget`, `hash_regex`) 이미
  존재, 공개 Dev 880문항 실측 점수도 있음(`baselines/README.md`). 현재 최고는
  `hash_regex`(ridge regression, NumPy만 사용) — 가중 최종 **0.6954**. 우리 목표는 이걸
  능가하는 것.

## 로컬 개발 루프

1. 데이터 준비(최초 1회):
   ```console
   python3 -m venv .venv-data
   .venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
   .venv-data/bin/python tools/materialize_public_data.py
   ```
   → `data/materialized/{train,dev}/inputs.json` 생성.
2. 정책 구현 → tier(`fast`/`balanced`/`premium`)별 제출 JSON 생성.
3. 공식 채점 재현(우리의 예전 `eval/`에 해당 — 이제 직접 안 짬, 공식 도구 그대로 씀):
   ```console
   PYTHONPATH=src python3 -m ossp_router.cli self-check \
     --input <input.json> --outcomes <outcomes.json> \
     --submissions <tier별 제출 폴더> --report build/report.json
   ```
4. 테스트: `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'`
5. 필요시 런타임 검증: `docker build ... container/Dockerfile` + `tools/check_runtime.py`
   (`linux/arm64` 최종 이미지 기준).

## PLACEHOLDER는 대부분 해소됨

예전 저장소의 TIER_WEIGHTS/MODEL_METAS/TIER_BUDGETS 추정값은 이제
`configs/routing-policy.v1.json`에 확정값으로 있다:

- tier 가중치: fast **0.4** / balanced **0.3** / premium **0.3**
- tier 비용 한도(전부 `ax31-light`로 선택했을 때 비용 대비 배수): fast **1.25** /
  balanced **2.0** / premium **4.0** — 한도 초과 시 그 tier **통째로 0점**(부분감점 없음)
- 모델별 비용은 고정단가가 아니라 입력/출력 **토큰당 요율**
  (`fixed_cost + input_tokens*input_token_rate/1e6 + output_tokens*output_token_rate/1e6`)

## 라이선스·공개

- 우리 코드: Apache-2.0(저장소 기본 라이선스 그대로 유지).
- 새 의존성/자료 추가 시 `DEVELOPING.md`의 "변경 원칙" 확인 — 출처·고정 버전·라이선스·
  고지 조건 검토, 스키마에 없는 필드 추가 금지, 비공개 평가 자료·사내 경로 절대 포함 금지.
- 데이터 라이선스: 대부분 permissive, **Belebele Korean만 CC-BY-SA-4.0**(고지 필요) —
  `DATA_LICENSES.md`, `THIRD_PARTY_NOTICES.md` 참고.
- 새로 추가하는 파일의 SPDX 헤더 관례(참가자 저작물에 SKT 저작권 표기를 유지할지 여부)는
  `docs/SUBMISSION.md`를 최종 제출 전에 확인 — 아직 검증 안 함.

## 제출 절차 요약 (`docs/SUBMISSION.md`가 원본)

fork 상태로 개발 → 코드 커밋 공개 → 그 커밋에서 `linux/arm64` 이미지 빌드해 공개
레지스트리에 push → 저장소 루트에 `submission-ossp-skt.json` 추가·커밋 → 그 커밋의 고정
GitHub 스냅샷 URL을 결과보고서 "프로젝트 등록 URL"에 기재. 마감: **2026-08-27 18:00 KST**,
osscontest.kr. 수상 시 제출 저장소를 5년간 공개 유지해야 함.
