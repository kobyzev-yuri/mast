#!/usr/bin/env python3
"""
Тестирование на data/test с использованием KB и Gemini.

Анализирует изображения из data/test/images/, использует KB для контекста,
отправляет запросы к Gemini для детекции мастоцитов, выдает результаты
в формате JSON, аналогичном data/train/annotations.json.

Использование:
  python scripts/test_with_kb.py \
    --test_images_dir data/test/images \
    --kb_path ./mast_cells_kb \
    --out data/test/predictions.json \
    --n_examples 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Загрузка config для Gemini
try:
    from dotenv import load_dotenv
    local_config = REPO_ROOT / "config.env"
    kb_service_config = REPO_ROOT.parent / "brats" / "kb-service" / "config.env"
    brats_config = REPO_ROOT.parent / "brats" / "config.env"
    if local_config.exists():
        load_dotenv(dotenv_path=local_config, override=True)
    if kb_service_config.exists():
        load_dotenv(dotenv_path=kb_service_config, override=False)
    if brats_config.exists():
        load_dotenv(dotenv_path=brats_config, override=False)
except ImportError:
    pass

from analyze_mast_cells_coordinates_gemini import GeminiVisionService
from analyze_with_kb import EnhancedMastCellsAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_gemini_response(response_text: str, image_name: str) -> list[dict]:
    """
    Парсит ответ Gemini и извлекает аннотации в формате train JSON.
    Ожидает структурированный ответ с координатами и типами мастоцитов.
    """
    import re
    annotations = []
    ann_id_counter = 1
    base_name = Path(image_name).stem

    # Ищем блоки с описанием мастоцитов
    # Паттерны: "Мастоцит #N:", "Cell #N:", координаты (x, y) или x: число, y: число
    lines = response_text.split("\n")
    current_cell = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("==="):
            continue

        # Ищем начало описания мастоцита
        cell_match = re.search(r"(?:Мастоцит|Mast\s+cell|Cell)\s*#?\s*(\d+)", line, re.IGNORECASE)
        if cell_match:
            if current_cell:
                annotations.append(current_cell)
            current_cell = None

        # Координаты: (x, y) или x: число, y: число
        coords_match = re.search(r"\((\d+)\s*,\s*(\d+)\)", line)
        if coords_match:
            x, y = int(coords_match.group(1)), int(coords_match.group(2))
        else:
            x_match = re.search(r"x[:\s]+(\d+)", line, re.IGNORECASE)
            y_match = re.search(r"y[:\s]+(\d+)", line, re.IGNORECASE)
            if x_match and y_match:
                x, y = int(x_match.group(1)), int(y_match.group(1))
            else:
                continue

        # Тип: explicit/implicit или явный/неявный
        cell_type = "explicit"  # по умолчанию
        if re.search(r"неявн|implicit", line, re.IGNORECASE):
            cell_type = "implicit"
        elif re.search(r"явн|explicit", line, re.IGNORECASE):
            cell_type = "explicit"

        # Размеры bbox (если указаны)
        w_match = re.search(r"w[:\s]+(\d+)|width[:\s]+(\d+)", line, re.IGNORECASE)
        h_match = re.search(r"h[:\s]+(\d+)|height[:\s]+(\d+)", line, re.IGNORECASE)
        w = int(w_match.group(1) or w_match.group(2)) if w_match else 40
        h = int(h_match.group(1) or h_match.group(2)) if h_match else 40

        ann_id = f"{base_name}_cell{ann_id_counter}"
        current_cell = {
            "id": ann_id,
            "class": cell_type,
            "bbox": {"x": max(0, x - w // 2), "y": max(0, y - h // 2), "w": w, "h": h}
        }
        ann_id_counter += 1

    if current_cell:
        annotations.append(current_cell)

    return annotations


async def test_images(
    test_images_dir: Path,
    kb_path: Path,
    out_path: Path,
    n_examples: int = 5,
) -> dict:
    """Анализирует изображения из test и сохраняет результаты в JSON формата train."""
    test_images_dir = Path(test_images_dir)
    if not test_images_dir.exists():
        raise ValueError(f"Директория не найдена: {test_images_dir}")

    images = sorted(test_images_dir.glob("*.jpeg")) + sorted(test_images_dir.glob("*.jpg")) + sorted(test_images_dir.glob("*.png"))
    if not images:
        raise ValueError(f"Не найдено изображений в {test_images_dir}")

    analyzer = EnhancedMastCellsAnalyzer(kb_path=str(kb_path))
    items = []

    for img_path in images:
        logger.info("Обработка %s...", img_path.name)
        try:
            result_text = await analyzer.analyze_with_context(
                image_path=img_path,
                query="""Find all mast cells in this image. For each found mast cell, provide:
1. Coordinates (x, y) of the nucleus center in pixels
2. Type: "explicit" or "implicit"
3. Bounding box dimensions (width w, height h) if possible, or use default 40x40

Format your response as:
Мастоцит #1: координаты (x, y), тип: explicit/implicit, размеры w x h
Мастоцит #2: координаты (x, y), тип: explicit/implicit, размеры w x h
...""",
                n_examples=n_examples,
            )

            annotations = parse_gemini_response(result_text, img_path.name)
            if annotations:
                items.append({
                    "image": img_path.name,
                    "annotations": annotations
                })
                logger.info("  Найдено мастоцитов: %d", len(annotations))
            else:
                logger.warning("  Мастоциты не найдены")
                items.append({
                    "image": img_path.name,
                    "annotations": []
                })

        except Exception as e:
            logger.error("Ошибка обработки %s: %s", img_path.name, e)
            items.append({
                "image": img_path.name,
                "annotations": []
            })

    await analyzer.close()

    # Сохраняем в формате train
    output = {"items": items}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("Результаты сохранены: %s", out_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Тестирование на data/test с KB и Gemini")
    parser.add_argument("--test_images_dir", type=Path, required=True, help="Директория с тестовыми изображениями")
    parser.add_argument("--kb_path", type=Path, default=REPO_ROOT / "mast_cells_kb", help="Путь к KB")
    parser.add_argument("--out", type=Path, required=True, help="Путь к выходному JSON (формат train)")
    parser.add_argument("--n_examples", type=int, default=5, help="Количество примеров из KB для контекста")
    args = parser.parse_args()

    result = asyncio.run(test_images(
        test_images_dir=args.test_images_dir,
        kb_path=args.kb_path,
        out_path=args.out,
        n_examples=args.n_examples,
    ))
    logger.info("Обработано изображений: %d", len(result["items"]))


if __name__ == "__main__":
    main()
