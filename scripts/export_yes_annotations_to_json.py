#!/usr/bin/env python3
"""
Экспорт разметки из изображений *_yes.png в JSON.

Предположение: разметка нанесена прямоугольниками (box) поверх изображения:
- зелёный box = explicit (явный)
- синий box   = implicit (неявный)
- красный     = игнорируем (по задаче)

Скрипт детектирует пиксели цветов рамок, собирает связные компоненты и переводит их в bbox.

Запуск:
  python scripts/export_yes_annotations_to_json.py \
    --mast_dir data/MAST_GEMINI/test \
    --out data/MAST_GEMINI/test/yes_annotations.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def green_box_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Детекция зелёной рамки: высокий G, низкие R/B и явное доминирование G.
    Подогнано по 1_yes.png (часто около (91,192,57) + антиалиасинг).
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return (g >= 175) & (r <= 170) & (b <= 170) & ((g - r) >= 35) & ((g - b) >= 35)


def blue_box_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Детекция синей рамки: высокий B, низкие R/G и явное доминирование B.
    Подогнано по 10_yes.png (часто около (28,71,205) + вариации).
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    # Более строгий порог, чтобы не захватывать «синие» оттенки гематоксилина в ткани
    return (b >= 190) & (r <= 140) & (g <= 140) & ((b - r) >= 60) & ((b - g) >= 60)


def _connected_components_bboxes(mask: np.ndarray, min_pixels: int = 150) -> list[dict]:
    """
    Находит bbox связных компонент в бинарной маске.
    Возвращает список bbox {x,y,w,h,pixels}.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    bboxes: list[dict] = []

    # 4-связность достаточна для рамок
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(h):
        row = mask[y]
        if not row.any():
            continue
        for x in np.where(row)[0]:
            if visited[y, x]:
                continue
            if not mask[y, x]:
                continue

            q = deque([(y, x)])
            visited[y, x] = True
            min_y = max_y = y
            min_x = max_x = x
            pixels = 0

            while q:
                cy, cx = q.popleft()
                pixels += 1
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx

                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))

            if pixels < min_pixels:
                continue

            bboxes.append(
                {
                    "x": int(min_x),
                    "y": int(min_y),
                    "w": int(max_x - min_x + 1),
                    "h": int(max_y - min_y + 1),
                    "pixels": int(pixels),
                }
            )

    return bboxes


def export_image_annotations(image_path: Path) -> dict:
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    height, width = rgb.shape[0], rgb.shape[1]

    green_mask = green_box_mask(rgb)
    blue_mask = blue_box_mask(rgb)

    green_boxes = _connected_components_bboxes(green_mask)
    blue_boxes = _connected_components_bboxes(blue_mask)

    annotations = []
    ann_id = 1
    for bbox in green_boxes:
        annotations.append(
            {
                "id": f"{image_path.stem}_g{ann_id}",
                "class": "explicit",
                "color": "green",
                "bbox": {k: bbox[k] for k in ("x", "y", "w", "h")},
                "pixels": bbox["pixels"],
                "source": "color_box",
            }
        )
        ann_id += 1

    ann_id = 1
    for bbox in blue_boxes:
        annotations.append(
            {
                "id": f"{image_path.stem}_b{ann_id}",
                "class": "implicit",
                "color": "blue",
                "bbox": {k: bbox[k] for k in ("x", "y", "w", "h")},
                "pixels": bbox["pixels"],
                "source": "color_box",
            }
        )
        ann_id += 1

    return {
        "image": image_path.name,
        "image_path": str(image_path),
        "width": int(width),
        "height": int(height),
        "annotations": annotations,
        "ignored": {"red": True},
    }


def iter_yes_images(mast_dir: Path) -> Iterable[Path]:
    yield from sorted(mast_dir.glob("*_yes.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export *_yes.png box annotations to JSON")
    parser.add_argument("--mast_dir", type=Path, required=True, help="Directory with *_yes.png")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    if not args.mast_dir.is_dir():
        raise SystemExit(f"Directory not found: {args.mast_dir}")

    images = list(iter_yes_images(args.mast_dir))
    if not images:
        raise SystemExit(f"No *_yes.png found in {args.mast_dir}")

    items = []
    for p in images:
        item = export_image_annotations(p)
        logger.info("%s: explicit=%d implicit=%d", p.name,
                    sum(1 for a in item["annotations"] if a["class"] == "explicit"),
                    sum(1 for a in item["annotations"] if a["class"] == "implicit"))
        items.append(item)

    out = {
        "schema": {
            "version": "1.0",
            "annotation_type": "bbox",
            "classes": ["explicit", "implicit"],
            "ignored_colors": ["red"],
            "bbox_format": {"x": "px", "y": "px", "w": "px", "h": "px"},
        },
        "items": items,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info("Wrote: %s", args.out)


if __name__ == "__main__":
    main()

