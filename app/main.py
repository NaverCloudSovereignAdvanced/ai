import json
import shutil
from pathlib import Path
from uuid import uuid4

import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, init_db
from app.document_loader import extract_text
from app.models import Source, StudentScore
from app.naver_ocr import detect_answers_with_naver_ocr
from app.ocr_grader import detect_checkbox_answers
from app.problem_generator import generate_problem_set
from app.rag import build_index, retrieve
from app.schemas import GradeResponse, ProblemRequest, ProblemSet, SourceResponse

app = FastAPI(title="LLM RAG Problem Generator AI Server", version="1.0.0")

PROBLEM_CONTEXT_QUERIES = [
    "핵심 개념 정의 특징 절차 비교 원인 결과",
    "중요 용어 설명 예시 차이점 장점 단점",
    "학습 목표 요약 시험 문제로 만들 주요 내용",
    "문서에서 반드시 이해해야 하는 사실 관계",
]


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <!doctype html>
    <html lang="ko">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>LLM RAG OCR AI Server</title>
        <style>
          body {
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            font-family: Arial, sans-serif;
            background: #f6f7f9;
            color: #1f2937;
          }
          main {
            width: min(720px, calc(100% - 32px));
            padding: 32px;
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
          }
          h1 {
            margin: 0 0 12px;
            font-size: 28px;
          }
          p {
            line-height: 1.6;
          }
          a {
            display: inline-block;
            margin-top: 12px;
            padding: 10px 14px;
            border-radius: 6px;
            background: #2563eb;
            color: #fff;
            text-decoration: none;
          }
        </style>
      </head>
      <body>
        <main>
          <h1>LLM RAG OCR AI Server</h1>
          <p>서버가 정상 실행 중입니다. 이 프로젝트는 API 서버라서 기능 테스트는 API 문서 화면에서 진행하면 됩니다.</p>
          <a href="/docs">API 문서 열기</a>
        </main>
      </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/source", response_model=SourceResponse)
async def create_source(file: UploadFile = File(...), db: Session = Depends(get_db)) -> SourceResponse:
    settings = get_settings()
    source_id = str(uuid4())
    safe_name = Path(file.filename or "source.txt").name
    saved_path = settings.data_dir / "sources" / f"{source_id}_{safe_name}"
    rag_path = settings.data_dir / "rag" / f"{source_id}.pkl"

    with saved_path.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)

    try:
        text = extract_text(saved_path, file.content_type or "")
        chunk_count = build_index(text, rag_path)
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        rag_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.add(
        Source(
            id=source_id,
            filename=safe_name,
            content_type=file.content_type or "application/octet-stream",
            file_path=str(saved_path),
            rag_path=str(rag_path),
            chunk_count=chunk_count,
        )
    )
    db.commit()
    return SourceResponse(sourceId=source_id, filename=safe_name, chunkCount=chunk_count)


@app.post("/api/v1/problem", response_model=ProblemSet)
def create_problem_set(request: ProblemRequest, db: Session = Depends(get_db)) -> ProblemSet:
    source = db.get(Source, request.sourceId)
    if source is None:
        raise HTTPException(status_code=404, detail="sourceId를 찾을 수 없습니다.")

    contexts = _collect_problem_contexts(Path(source.rag_path), source.filename)
    return generate_problem_set(contexts)


def _collect_problem_contexts(rag_path: Path, filename: str, max_contexts: int = 16) -> list[str]:
    if not rag_path.exists():
        raise HTTPException(status_code=404, detail="해당 sourceId의 RAG 인덱스를 찾을 수 없습니다.")

    contexts: list[str] = []
    seen: set[str] = set()
    for query in [*PROBLEM_CONTEXT_QUERIES, filename]:
        for chunk in retrieve(rag_path, query, top_k=6):
            normalized = " ".join(chunk.split())
            if normalized in seen:
                continue
            seen.add(normalized)
            contexts.append(chunk)
            if len(contexts) >= max_contexts:
                return contexts

    if not contexts:
        raise HTTPException(status_code=422, detail="문제 생성을 위한 문서 컨텍스트를 찾을 수 없습니다.")
    return contexts


@app.post("/api/v1/grade", response_model=GradeResponse)
async def grade_answer_sheet(
    studentId: str = Form(...),
    answerSetJson: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> GradeResponse:
    settings = get_settings()
    image_id = str(uuid4())
    image_path = settings.data_dir / "sources" / f"{image_id}_{Path(image.filename or 'answer.png').name}"
    with image_path.open("wb") as fp:
        shutil.copyfileobj(image.file, fp)

    try:
        answer_set = ProblemSet.model_validate_json(answerSetJson)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="answerSetJson은 ProblemSet JSON이어야 합니다.") from exc

    correct_answers = [problem.answerIndex for problem in answer_set.problems]
    try:
        detected_answers = detect_answers_with_naver_ocr(image_path, question_count=len(correct_answers))
    except requests.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"네이버 OCR 호출에 실패했습니다: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if any(answer == -1 for answer in detected_answers):
        checkbox_answers = detect_checkbox_answers(image_path, question_count=len(correct_answers), choice_count=4)
        detected_answers = [
            detected if detected != -1 else checkbox
            for detected, checkbox in zip(detected_answers, checkbox_answers)
        ]
    score = sum(1 for detected, correct in zip(detected_answers, correct_answers) if detected == correct)

    db.add(
        StudentScore(
            student_id=studentId,
            score=score,
            total=len(correct_answers),
            answers_json=json.dumps(detected_answers, ensure_ascii=False),
        )
    )
    db.commit()

    return GradeResponse(
        studentId=studentId,
        score=score,
        total=len(correct_answers),
        detectedAnswers=detected_answers,
        correctAnswers=correct_answers,
    )
