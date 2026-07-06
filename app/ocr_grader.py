from pathlib import Path

import cv2
import numpy as np


def detect_checkbox_answers(image_path: Path, question_count: int = 10, choice_count: int = 4) -> list[int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("이미지를 읽을 수 없습니다.")

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h)
        if 12 <= w <= 80 and 12 <= h <= 80 and 0.7 <= aspect <= 1.3:
            roi = threshold[y : y + h, x : x + w]
            fill_ratio = float(np.count_nonzero(roi)) / float(w * h)
            boxes.append((x, y, w, h, fill_ratio))

    boxes.sort(key=lambda item: (item[1], item[0]))
    rows = _group_rows(boxes, question_count)
    answers: list[int] = []
    for row in rows[:question_count]:
        row = sorted(row, key=lambda item: item[0])[:choice_count]
        if len(row) < choice_count:
            answers.append(-1)
            continue
        selected = max(range(len(row)), key=lambda idx: row[idx][4])
        answers.append(selected)

    while len(answers) < question_count:
        answers.append(-1)
    return answers


def _group_rows(
    boxes: list[tuple[int, int, int, int, float]], question_count: int
) -> list[list[tuple[int, int, int, int, float]]]:
    rows: list[list[tuple[int, int, int, int, float]]] = []
    for box in boxes:
        _, y, _, h, _ = box
        placed = False
        for row in rows:
            row_y = sum(item[1] for item in row) / len(row)
            if abs(y - row_y) <= max(12, h * 0.7):
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    rows.sort(key=lambda row: sum(item[1] for item in row) / len(row))
    return [row for row in rows if len(row) >= 2][:question_count]
