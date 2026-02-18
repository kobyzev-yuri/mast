#!/usr/bin/env python3
"""
Аудит разметки через Gemini: проверка согласованности классов (явные/неявные)
и рекомендации по единому формату.

Загружает yes_annotations.json и отправляет Gemini несколько пар (_no, _yes)
или кропов с описанием текущей схемы (зелёный=явные, синий=неявные, красный игнор).
Запрашивает: подтверждение/корректировку классов и советы по улучшению разметки и KB.

Использование:
  python scripts/audit_annotations_with_gemini.py \
    --annotations data/MAST_GEMINI/test/yes_annotations.json \
    --max_pairs 3 \
    --out results/audit_annotations_gemini.txt
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

# Загрузка config.env для GEMINI_API_KEY (тот же порядок, что в analyze_mast_cells_coordinates_gemini)
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


AUDIT_SYSTEM = """Ты эксперт-патолог. Твоя задача — проверить согласованность разметки мастоцитов на гистологических срезах (Г/Э) и дать рекомендации по единому формату классов."""

AUDIT_PROMPT_TPL = """
На этих изображениях используется разметка:
- Зелёные рамки = явные мастоциты (explicit): чёткая зернистость, хороший контраст.
- Синие рамки = неявные мастоциты (implicit): размытые границы, низкий контраст.
- Красная разметка не используется (игнорируем).

Твои задачи:
1. Подтверди или скорректируй: все объекты в зелёных рамках ты бы классифицировал как явные, в синих — как неявные? Если нет — укажи, какие именно и почему.
2. Единый формат: достаточно ли двух классов (явный/неявный) или целесообразно ввести третий (например, «неопределённый / только с ИГХ»)? Дай краткую рекомендацию.
3. Рекомендации для KB: какие морфологические признаки и примеры важнее всего добавить в базу знаний, чтобы детекция явных и неявных мастоцитов была стабильнее?
4. Качество разметки: есть ли на этих кадрах явные ошибки или субъективная непоследовательность (один и тот же тип клетки размечен по-разному)?

Ответь структурированно: 1) Подтверждение/корректировка 2) Единый формат 3) Рекомендации для KB 4) Качество разметки.
"""


def collect_image_pairs(annotations_path: Path, max_pairs: int) -> list[tuple[Path, Path]]:
    """По JSON находит пары (_no, _yes) и возвращает до max_pairs путей."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    base_dir = annotations_path.parent
    pairs = []
    seen_bases = set()
    for item in items:
        if len(pairs) >= max_pairs:
            break
        name = item.get("image", "")
        if not name.endswith("_yes.png"):
            continue
        base = name.replace("_yes.png", "")
        if base in seen_bases:
            continue
        no_path = base_dir / f"{base}_no.png"
        yes_path = base_dir / name
        if no_path.exists() and yes_path.exists():
            pairs.append((no_path, yes_path))
            seen_bases.add(base)
    return pairs


async def run_audit(
    annotations_path: Path,
    out_path: Path,
    max_pairs: int = 3,
) -> str:
    pairs = collect_image_pairs(annotations_path, max_pairs)
    if not pairs:
        return "Нет подходящих пар _no/_yes для аудита."
    try:
        gemini = GeminiVisionService()
    except ValueError as e:
        return f"Ошибка инициализации Gemini: {e}"
    image_paths = []
    for no_p, yes_p in pairs:
        image_paths.append(no_p)
        image_paths.append(yes_p)
    labels = []
    for no_p, yes_p in pairs:
        labels.append(no_p.name)
        labels.append(yes_p.name)
    prompt = AUDIT_PROMPT_TPL
    if labels:
        prompt += "\n\nИзображения по порядку: " + ", ".join(labels)
    try:
        result = await gemini.analyze_images(
            image_paths=[Path(p) for p in image_paths],
            prompt=prompt,
            system_prompt=AUDIT_SYSTEM,
            preserve_resolution=False,
        )
    finally:
        await gemini.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Аудит разметки мастоцитов (Gemini)\n\n")
        f.write(f"Пар изображений: {len(pairs)}\n")
        f.write("Файлы: " + ", ".join([f"{a.name} / {b.name}" for a, b in pairs]) + "\n\n")
        f.write("---\n\n")
        f.write(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Аудит разметки через Gemini")
    parser.add_argument("--annotations", type=Path, required=True, help="Путь к yes_annotations.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "audit_annotations_gemini.txt", help="Куда сохранить ответ")
    parser.add_argument("--max_pairs", type=int, default=3, help="Сколько пар (_no,_yes) отправить (по умолчанию 3)")
    args = parser.parse_args()
    if not args.annotations.exists():
        raise SystemExit(f"Файл не найден: {args.annotations}")
    result = asyncio.run(run_audit(args.annotations, args.out, args.max_pairs))
    logger.info("Ответ сохранён: %s", args.out)
    print(result[:800] + ("..." if len(result) > 800 else ""))


if __name__ == "__main__":
    main()
