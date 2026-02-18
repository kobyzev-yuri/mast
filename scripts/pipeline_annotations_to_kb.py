#!/usr/bin/env python3
"""
Конвейер: ваш JSON разметки + изображения _no → кропы → добавление в KB.

Читает JSON с аннотациями (формат: docs/ANNOTATION_JSON_FORMAT.md). Для каждого
элемента items[].annotations[] находит соответствующий *_no.png в той же папке,
что и JSON (имя: из items[].image получается base, ищется {base}_no.png),
вырезает патч по bbox (с отступом), сохраняет кропы в crops_dir и при необходимости
добавляет записи в MastCellsKnowledgeBase.

Использование:
  python scripts/pipeline_annotations_to_kb.py \
    --annotations /path/to/your/annotations.json \
    --crops_dir data/MAST_GEMINI/test/crops \
    --kb_path ./mast_cells_kb
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
    """Ленивый импорт KB, чтобы при --crops_only не требовать sentence_transformers."""
    from train_knowledge_base import MastCellsKnowledgeBase
    return MastCellsKnowledgeBase(db_path=kb_path, use_clip=use_clip)

# Отступ вокруг bbox при вырезке (пиксели или доля от min(w,h)); минимум 10 px
CROP_PADDING_PX = 20
CROP_PADDING_FRAC = 0.15


def crop_bbox_from_image(
    image: Image.Image,
    bbox: dict,
    padding_px: int = CROP_PADDING_PX,
    padding_frac: float = CROP_PADDING_FRAC,
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
    crops_dir: Path,
    kb_path: Path,
    use_clip: bool = True,
    skip_existing_crops: bool = True,
    crops_only: bool = False,
) -> dict:
    """
    Загружает JSON, создаёт кропы из _no, добавляет примеры в KB.
    Возвращает статистику: crops_created, kb_added, errors.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    schema = data.get("schema", {})
    items = data.get("items", [])
    if not items:
        logger.warning("Нет записей в JSON")
        return {"crops_created": 0, "kb_added": 0, "errors": []}

    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    kb = None if crops_only else _get_kb(str(kb_path), use_clip)
    stats = {"crops_created": 0, "kb_added": 0, "errors": []}

    for item in items:
        image_name = item["image"]
        image_path_str = item.get("image_path", "")
        if not image_path_str:
            image_path = annotations_path.parent / image_name
        else:
            image_path = Path(image_path_str)
            if not image_path.is_absolute():
                image_path = annotations_path.parent / image_path.name
        # Соответствующий _no
        base = image_path.stem.replace("_yes", "")
        no_name = f"{base}_no.png"
        no_path = image_path.parent / no_name
        if not no_path.exists():
            stats["errors"].append(f"Файл не найден: {no_path}")
            continue
        try:
            img_no = Image.open(no_path).convert("RGB")
        except Exception as e:
            stats["errors"].append(f"Ошибка открытия {no_path}: {e}")
            continue

        for ann in item.get("annotations", []):
            ann_id = ann["id"]
            cell_type = ann.get("class", "explicit")
            bbox = ann.get("bbox", {})
            if not bbox:
                continue
            crop_path = crops_dir / f"{ann_id}.png"
            if skip_existing_crops and crop_path.exists():
                logger.debug("Пропуск существующего кропа %s", crop_path.name)
            else:
                try:
                    crop = crop_bbox_from_image(img_no, bbox)
                    crop.save(crop_path)
                    stats["crops_created"] += 1
                except Exception as e:
                    stats["errors"].append(f"Кроп {ann_id}: {e}")
                    continue

            if not crops_only and kb:
                difficulty = "easy" if cell_type == "explicit" else "medium"
                confidence = "high" if cell_type == "explicit" else "medium"
                cx = bbox.get("x", 0) + bbox.get("w", 0) // 2
                cy = bbox.get("y", 0) + bbox.get("h", 0) // 2
                try:
                    kb.add_example(
                        example_id=ann_id,
                        image_path=str(crop_path),
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
    parser = argparse.ArgumentParser(description="Pipeline: annotations JSON → crops → KB")
    parser.add_argument("--annotations", type=Path, required=True, help="Путь к yes_annotations.json")
    parser.add_argument("--crops_dir", type=Path, required=True, help="Директория для кропов")
    parser.add_argument("--kb_path", type=Path, default=REPO_ROOT / "mast_cells_kb", help="Путь к базе знаний")
    parser.add_argument("--crops_only", action="store_true", help="Только создать кропы, не добавлять в KB (не требует sentence_transformers)")
    parser.add_argument("--no_clip", action="store_true", help="Не использовать CLIP в KB")
    parser.add_argument("--no_skip_existing", action="store_true", help="Перезаписывать существующие кропы")
    args = parser.parse_args()

    if not args.annotations.exists():
        raise SystemExit(f"Файл не найден: {args.annotations}")

    stats = run_pipeline(
        annotations_path=args.annotations,
        crops_dir=args.crops_dir,
        kb_path=args.kb_path,
        use_clip=not args.no_clip,
        skip_existing_crops=not args.no_skip_existing,
        crops_only=args.crops_only,
    )
    logger.info("Готово: кропов создано=%d, добавлено в KB=%d", stats["crops_created"], stats["kb_added"])
    for err in stats["errors"]:
        logger.error("%s", err)


if __name__ == "__main__":
    main()
