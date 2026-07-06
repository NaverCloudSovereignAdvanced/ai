# LLM & RAG 문제 생성 + OCR 자동 채점 AI 서버

FastAPI 기반 AI 서버 예시입니다.

- `/`: 서버 실행 확인용 첫 화면
- `/docs`: API 문서 화면
- `/health`: 상태 확인 API
- `/api/v1/source`: PDF, TXT, DOCX 문서를 업로드하고 RAG 인덱스를 생성한 뒤 `sourceId`를 반환합니다.
- `/api/v1/problem`: `sourceId`를 받아 4지선다 객관식 10문항과 정답 인덱스를 JSON으로 반환합니다.
- `/api/v1/grade`: 답안 이미지를 OCR로 분석하고 정답 세트와 비교해 채점합니다.

## 실행

```bash
docker compose up --build
```

실행 후 브라우저에서 아래 주소를 열면 됩니다.

```text
http://localhost:8000
```

API 문서는 아래 주소에서 확인할 수 있습니다.

```text
http://localhost:8000/docs
```

## 환경 변수

- `GEMINI_API_KEY`: 있으면 Gemini 기반 문제 생성에 사용합니다.
- `GEMINI_MODEL`: 기본값은 `gemini-2.0-flash`입니다.
- `NAVER_CLOVA_OCR_INVOKE_URL`: 네이버 CLOVA OCR Invoke URL입니다.
- `NAVER_CLOVA_OCR_SECRET_KEY`: 네이버 CLOVA OCR Secret Key입니다.
- `DATABASE_URL`: 기본값은 `sqlite:////app/data/app.db`입니다.
- `DATA_DIR`: 기본값은 `/app/data`입니다.

`GEMINI_API_KEY`가 없어도 서버는 RAG 검색 결과를 바탕으로 기본 문제를 생성합니다.

## 답안 이미지 형식 권장

OCR 텍스트 인식으로 안정적으로 정규화하기 위해 답안지에는 각 문항의 선택 답을 `1 A`, `2 C` 또는 `1번 B`처럼 텍스트로 함께 표시하는 형식을 권장합니다. 체크박스만 있는 경우에는 OCR 결과가 부족할 수 있어 서버가 로컬 이미지 분석을 보조로 사용합니다.
