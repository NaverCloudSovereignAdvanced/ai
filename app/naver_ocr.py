import base64
import re
import time
from pathlib import Path
from uuid import uuid4

import requests

from app.config import get_settings


CHOICE_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "①": 0,
    "②": 1,
    "③": 2,
    "④": 3,
}


def detect_answers_with_naver_ocr(image_path: Path, question_count: int) -> list[int]:
    settings = get_settings()
    if not settings.naver_clova_ocr_invoke_url or not settings.naver_clova_ocr_secret_key:
        raise ValueError("NAVER_CLOVA_OCR_INVOKE_URL과 NAVER_CLOVA_OCR_SECRET_KEY가 필요합니다.")

    payload = _build_payload(image_path)
    response = requests.post(
        settings.naver_clova_ocr_invoke_url,
        headers={
            "Content-Type": "application/json",
            "X-OCR-SECRET": settings.naver_clova_ocr_secret_key,
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    text = _extract_text(data)
    return normalize_answers_from_text(text, question_count)


def normalize_answers_from_text(text: str, question_count: int) -> list[int]:
    answers = [-1] * question_count
    normalized = text.replace("\n", " ")

    patterns = [
        r"(?P<num>\d{1,2})\s*(?:번|\.|\)|:|-)?\s*(?P<choice>[ABCDabcd①②③④1-4])",
        r"(?P<num>\d{1,2})\s*(?:번|\.|\)|:|-)?\s*(?:답|정답)\s*(?P<choice>[ABCDabcd①②③④1-4])",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            number = int(match.group("num"))
            if not 1 <= number <= question_count:
                continue
            choice = match.group("choice").upper()
            answers[number - 1] = CHOICE_MAP.get(choice, -1)

    return answers


def _build_payload(image_path: Path) -> dict:
    image_format = image_path.suffix.lower().lstrip(".") or "jpg"
    with image_path.open("rb") as fp:
        image_data = base64.b64encode(fp.read()).decode("utf-8")

    return {
        "version": "V2",
        "requestId": str(uuid4()),
        "timestamp": int(time.time() * 1000),
        "images": [
            {
                "format": image_format,
                "name": image_path.stem,
                "data": image_data,
            }
        ],
    }


def _extract_text(data: dict) -> str:
    fields = []
    for image in data.get("images", []):
        for field in image.get("fields", []):
            text = field.get("inferText")
            if text:
                fields.append(text)
    return " ".join(fields)
