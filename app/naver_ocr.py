import base64
import re
import time
from uuid import uuid4

import requests

from app.config import get_settings


def extract_grading_fields(image_bytes: bytes, filename: str, content_type: str | None = None) -> tuple[str, list[int]]:
    """CLOVA OCR 템플릿 결과에서 이름과 체크박스 문항별 선택 index를 추출한다."""
    settings = get_settings()
    if not settings.naver_clova_ocr_invoke_url or not settings.naver_clova_ocr_secret_key:
        raise ValueError("NAVER_CLOVA_OCR_INVOKE_URL과 NAVER_CLOVA_OCR_SECRET_KEY가 필요합니다.")

    payload = _build_payload(image_bytes, filename, content_type)
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
    return _parse_answer_sheet_fields(response.json())


def _build_payload(image_bytes: bytes, filename: str, content_type: str | None) -> dict:
    image_format = _detect_image_format(filename, content_type)
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "version": "V2",
        "requestId": str(uuid4()),
        "timestamp": int(time.time() * 1000),
        "lang": "ko",
        "images": [
            {
                "format": image_format,
                "name": "answer_sheet",
                "data": image_data,
            }
        ],
    }


_CONTENT_TYPE_FORMATS = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/tiff": "tif",
    "image/bmp": "bmp",
    "application/pdf": "pdf",
}

_SUPPORTED_FORMATS = {"jpg", "jpeg", "png", "pdf", "tif", "tiff", "bmp"}


def _detect_image_format(filename: str, content_type: str | None = None) -> str:
    if content_type:
        mapped = _CONTENT_TYPE_FORMATS.get(content_type.split(";")[0].strip().lower())
        if mapped:
            return mapped

    ext = (filename or "").rsplit(".", 1)[-1].lower()
    if ext in _SUPPORTED_FORMATS:
        return ext

    return "jpg"


def _is_checked(field: dict) -> bool:
    checked = field.get("checked")
    if isinstance(checked, bool):
        return checked

    infer_text = str(field.get("inferText", "")).strip().lower()
    return infer_text in {"true", "checked", "yes", "1", "v", "✓"}


def _parse_field_number(field_name: str) -> int | None:
    match = re.fullmatch(r"Field\s*0*(\d+)", field_name.strip())
    if not match:
        return None
    return int(match.group(1))


def _parse_answer_sheet_fields(data: dict) -> tuple[str, list[int]]:
    images = data.get("images", [])
    if not images:
        raise ValueError("OCR 응답에 images가 없습니다.")

    image = images[0]
    if image.get("inferResult") != "SUCCESS":
        raise ValueError(f"OCR 처리 실패: {image.get('message', 'unknown error')}")

    fields = image.get("fields", [])

    name = ""
    checkbox_by_number: dict[int, dict] = {}
    for field in fields:
        field_name = str(field.get("name", ""))
        if field_name == "name":
            name = str(field.get("inferText", "")).strip()
            continue
        field_no = _parse_field_number(field_name)
        if field_no is not None:
            checkbox_by_number[field_no] = field

    if not checkbox_by_number:
        return name, []

    question_count = max(checkbox_by_number) // 4
    selected_indexes: list[int] = []
    for question_no in range(1, question_count + 1):
        start = (question_no - 1) * 4 + 1
        checked_options = [
            option_index
            for option_index in range(4)
            if _is_checked(checkbox_by_number.get(start + option_index, {}))
        ]
        selected_indexes.append(checked_options[0] if len(checked_options) == 1 else -1)

    return name, selected_indexes
