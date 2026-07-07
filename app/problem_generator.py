import json
from hashlib import sha256

import requests

from app.config import get_settings
from app.schemas import Problem, ProblemSet


PROBLEM_COUNT = 10


def generate_problem_set(contexts: list[str]) -> ProblemSet:
    settings = get_settings()
    if settings.gemini_api_key:
        try:
            return _generate_with_gemini(contexts, settings.gemini_api_key, settings.gemini_model)
        except Exception:
            return _generate_fallback(contexts)
    return _generate_fallback(contexts)


def _generate_with_gemini(contexts: list[str], api_key: str, model: str) -> ProblemSet:
    context = "\n\n".join(contexts)
    prompt = f"""
당신은 대학 객관식 문제 출제자입니다.

아래 RAG 컨텍스트는 문제를 만들기 위한 참고자료일 뿐이며,
실제 시험지에는 제공되지 않습니다.
따라서 학생이 문제와 선지만 보고 풀 수 있도록 완결된 문제를 만드세요.

객관식 4지선다 문제를 정확히 10개 생성하세요.

출제 규칙:
- 각 문제는 question, choices, answerIndex를 가져야 합니다.
- answerIndex는 0부터 시작합니다.
- 문제는 반드시 RAG 컨텍스트의 내용만 근거로 출제하세요.
- 문제 안에 필요한 개념이나 상황을 포함하여, 자료 없이도 풀 수 있게 작성하세요.
- "자료에 따르면", "제공된 자료", "본문", "컨텍스트", "RAG" 같은 표현은 절대 사용하지 마세요.
- "다음 중 제공된 자료의 내용과 가장 일치하는 것은?" 같은 메타형 문제를 만들지 마세요.
- question 앞에 "1.", "2.", "3.", "문제 1", "Q1" 같은 번호를 절대 붙이지 마세요.
- question에는 순수한 문제 문장만 작성하세요.
- 선지는 판단 문장보다 개념명, 정의, 특징, 예시 중심으로 짧게 작성하세요.
- 문제 문장과 각 선지는 가능하면 25자 이내로 작성하세요.
- 정답은 하나만 명확해야 합니다.
- 오답 선지는 그럴듯하지만 명확히 틀려야 합니다.
- 설명 없이 JSON만 반환하세요.

좋은 문제 예시:
{{
  "question": "에이전트가 보상을 받으며 시행착오로 학습하는 방법은?",
  "choices": ["지도학습", "비지도학습", "강화학습", "전이학습"],
  "answerIndex": 2
}}

나쁜 문제 예시:
{{
  "question": "1. 다음 중 제공된 자료의 내용과 가장 일치하는 것은?",
  "choices": ["자료에 따르면 강화학습은...", "자료에서 확인되지 않는다", "본문과 다르다", "판단할 수 없다"],
  "answerIndex": 0
}}

반환 형식:
{{
  "count": 10,
  "problems": [
    {{
      "question": "...",
      "choices": ["...", "...", "...", "..."],
      "answerIndex": 0
    }}
  ]
}}

RAG 컨텍스트:
{context}
""".strip()
    print(prompt)

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json",
            },
        },
        timeout=60,
    )
    print(response)
    response.raise_for_status()
    data = json.loads(_extract_gemini_text(response.json()))
    problem_set = ProblemSet.model_validate(data)
    return _normalize_count(problem_set)


def _extract_gemini_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Gemini 응답에서 문제 JSON을 찾을 수 없습니다.") from exc

    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise ValueError("Gemini 응답이 비어 있습니다.")
    return text


def _generate_fallback(contexts: list[str]) -> ProblemSet:
    seed_text = " ".join(contexts)[:6000] or "업로드된 학습 자료"
    sentences = [s.strip() for s in seed_text.replace("?", ".").replace("!", ".").split(".") if len(s.strip()) > 24]
    if not sentences:
        sentences = [seed_text[:180] or "자료 내용을 확인하세요"]

    problems: list[Problem] = []
    for i in range(PROBLEM_COUNT):
        fact = sentences[i % len(sentences)][:180]
        digest = sha256(f"{i}:{fact}".encode("utf-8")).hexdigest()
        answer_index = int(digest[:2], 16) % 4
        correct = f"자료에 따르면, {fact}"
        distractors = [
            "자료에서 확인되지 않는 설명입니다.",
            "본문의 핵심 내용과 다른 설명입니다.",
            "제시된 근거만으로 판단할 수 없는 설명입니다.",
        ]
        choices = distractors[:]
        choices.insert(answer_index, correct)
        problems.append(
            Problem(
                question=f"{i + 1}. 다음 중 제공된 자료의 내용과 가장 일치하는 것은?",
                choices=choices,
                answerIndex=answer_index,
            )
        )
    return ProblemSet(count=PROBLEM_COUNT, problems=problems)


def _normalize_count(problem_set: ProblemSet) -> ProblemSet:
    problems = problem_set.problems[:PROBLEM_COUNT]
    if len(problems) < PROBLEM_COUNT:
        fallback = _generate_fallback([json.dumps(problem_set.model_dump(), ensure_ascii=False)])
        problems.extend(fallback.problems[len(problems):])
    return ProblemSet(count=PROBLEM_COUNT, problems=problems)
