#!/usr/bin/env python3
"""
Конвейер для наполнения KB из data/train.

Отличается от pipeline_annotations_to_kb.py тем, что:
- Изображения уже готовые патчи (не нужно искать _no и вырезать кропы)
- Bbox относительны к самим изображениям (можно использовать как есть или вырезать меньшие кропы)
- Изображения лежат в data/train/images/, JSON в data/train/annotations.json

Использование:
  python scripts/pipeline_train_to_kb.py \
    --annotations data/train/annotations.json \
    --images_dir data/train/images \
    --kb_path ./mast_cells_kb \
    --crop_bbox  # опционально: вырезать меньшие кропы по bbox
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _get_kb(kb_path: str, use_clip: bool = True):
    """Ленивый импорт KB."""
    from train_knowledge_base import MastCellsKnowledgeBase
    return MastCellsKnowledgeBase(db_path=kb_path, use_clip=use_clip)


def crop_bbox_from_image(
    image: Image.Image,
    bbox: dict,
    padding_px: int = 10,
    padding_frac: float = 0.1,
) -> Image.Image:
    """Вырезает прямоугольник с отступом; границы обрезаются по размеру изображения."""
    w_img, h_img = image.size
    x = bbox["x"]
    y = bbox["y"]
    w = bbox["w"]
    h = bbox["h"]
    pad = max(padding_px, int(min(w, h) * padding_frac))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)
    return image.crop((x0, y0, x1, y1))


def run_pipeline(
    annotations_path: Path,
    images_dir: Path,
    kb_path: Path,
    use_clip: bool = True,
    crop_bbox: bool = False,
    crops_dir: Path | None = None,
) -> dict:
    """
    Загружает train JSON, добавляет примеры в KB.
    Если crop_bbox=True, вырезает меньшие кропы по bbox и сохраняет в crops_dir.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    if not items:
        logger.warning("Нет записей в JSON")
        return {"kb_added": 0, "crops_created": 0, "errors": []}

    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise ValueError(f"Директория изображений не найдена: {images_dir}")

    kb = _get_kb(str(kb_path), use_clip)
    stats = {"kb_added": 0, "crops_created": 0, "errors": []}

    if crop_bbox and crops_dir:
        crops_dir = Path(crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        image_name = item["image"]
        image_path = images_dir / image_name
        if not image_path.exists():
            stats["errors"].append(f"Изображение не найдено: {image_path}")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            stats["errors"].append(f"Ошибка открытия {image_path}: {e}")
            continue

        for ann in item.get("annotations", []):
            ann_id = ann["id"]
            cell_type = ann.get("class", "explicit")
            bbox = ann.get("bbox", {})

            # Если crop_bbox=True, вырезаем меньший кроп по bbox
            if crop_bbox and crops_dir:
                try:
                    crop = crop_bbox_from_image(img, bbox)
                    crop_path = crops_dir / f"{ann_id}.png"
                    crop.save(crop_path)
                    stats["crops_created"] += 1
                    image_path_for_kb = str(crop_path)
                except Exception as e:
                    stats["errors"].append(f"Кроп {ann_id}: {e}")
                    continue
            else:
                # Используем всё изображение целиком
                image_path_for_kb = str(image_path)

            # Добавляем в KB
            difficulty = "easy" if cell_type == "explicit" else "medium"
            confidence = "high" if cell_type == "explicit" else "medium"
            cx = bbox.get("x", 0) + bbox.get("w", 0) // 2 if bbox else img.size[0] // 2
            cy = bbox.get("y", 0) + bbox.get("h", 0) // 2 if bbox else img.size[1] // 2

            try:
                kb.add_example(
                    example_id=ann_id,
                    image_path=image_path_for_kb,
                    morphological_features={
                        "nucleus": "central, round/ovoid" if cell_type == "explicit" else "central, ovoid, low contrast",
                        "cytoplasm": "granular, eosinophilic" if cell_type == "explicit" else "pale, blurred boundaries",
                        "shape": "round to ovoid",
                        "location": "stroma",
                    },
                    difficulty=difficulty,
                    coordinates={"x": cx, "y": cy},
                    cell_type=cell_type,
                    confidence=confidence,
                )
                stats["kb_added"] += 1
            except Exception as e:
                stats["errors"].append(f"KB add {ann_id}: {e}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline: train annotations → KB")
    parser.add_argument("--annotations", type=Path, required=True, help="Путь к annotations.json")
    parser.add_argument("--images_dir", type=Path, required=True, help="Директория с изображениями")
    parser.add_argument("--kb_path", type=Path, default=REPO_ROOT / "mast_cells_kb", help="Путь к KB")
    parser.add_argument("--no_clip", action="store_true", help="Не использовать CLIP")
    parser.add_argument("--crop_bbox", action="store_true", help="Вырезать меньшие кропы по bbox (иначе использует всё изображение)")
    parser.add_argument("--crops_dir", type=Path, help="Директория для кропов (если --crop_bbox)")
    args = parser.parse_args()

    if not args.annotations.exists():
        raise SystemExit(f"Файл не найден: {args.annotations}")

    stats = run_pipeline(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        kb_path=args.kb_path,
        use_clip=not args.no_clip,
        crop_bbox=args.crop_bbox,
        crops_dir=args.crops_dir,
    )
    logger.info("Готово: добавлено в KB=%d, кропов создано=%d", stats["kb_added"], stats["crops_created"])
    for err in stats["errors"]:
        logger.error("%s", err)


if __name__ == "__main__":
    main()
