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
다음 RAG 컨텍스트만 근거로 객관식 4지선다 문제를 정확히 10개 생성하세요.
각 문제는 question, choices, answerIndex를 가져야 합니다.
answerIndex는 0부터 시작합니다.
반드시 설명 없이 JSON만 반환하세요.

반환 형식:
{{
  "count": 10,
  "problems": [
    {{"question": "...", "choices": ["...", "...", "...", "..."], "answerIndex": 0}}
  ]
}}

RAG 컨텍스트:
{context}
""".strip()

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
