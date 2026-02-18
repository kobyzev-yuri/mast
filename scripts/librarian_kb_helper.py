#!/usr/bin/env python3
"""
Помощник «библиотекаря» для формирования KB: импорт разметки из JSON,
статистика, запуск конвейера (кропы → KB) и отчёт по покрытию KB.

Команды:
  stats     — статистика по yes_annotations.json (явные/неявные по файлам).
  pipeline  — запуск конвейера (JSON → кропы → KB).
  kb_report — отчёт по содержимому KB (по cell_type, difficulty).
  full      — stats + pipeline + kb_report (последовательно).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_stats(annotations_path: Path) -> dict:
    """Статистика по JSON разметки."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    total_explicit = total_implicit = 0
    per_file = []
    for it in items:
        e = sum(1 for a in it.get("annotations", []) if a.get("class") == "explicit")
        i = sum(1 for a in it.get("annotations", []) if a.get("class") == "implicit")
        total_explicit += e
        total_implicit += i
        per_file.append((it.get("image", ""), e, i))
    return {
        "total_explicit": total_explicit,
        "total_implicit": total_implicit,
        "total_images": len(items),
        "per_file": per_file,
    }


def cmd_pipeline(
    annotations_path: Path,
    crops_dir: Path,
    kb_path: Path,
    crops_only: bool = False,
    no_clip: bool = False,
) -> int:
    """Запуск конвейера через subprocess (чтобы не тянуть тяжёлые импорты при stats)."""
    args = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "pipeline_annotations_to_kb.py"),
        "--annotations", str(annotations_path),
        "--crops_dir", str(crops_dir),
        "--kb_path", str(kb_path),
    ]
    if crops_only:
        args.append("--crops_only")
    if no_clip:
        args.append("--no_clip")
    return subprocess.run(args, cwd=str(REPO_ROOT)).returncode


def cmd_kb_report(kb_path: Path) -> dict:
    """Отчёт по KB: количество примеров по cell_type и difficulty."""
    try:
        from train_knowledge_base import MastCellsKnowledgeBase
    except Exception as e:
        return {"error": str(e)}
    kb = MastCellsKnowledgeBase(db_path=str(kb_path), use_clip=False)
    examples = kb.get_all_examples()
    by_cell = {}
    by_difficulty = {}
    for ex in examples:
        meta = ex.get("metadata") or {}
        ct = meta.get("cell_type") or "unknown"
        by_cell[ct] = by_cell.get(ct, 0) + 1
        diff = meta.get("difficulty") or "unknown"
        by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
    return {
        "total": len(examples),
        "by_cell_type": by_cell,
        "by_difficulty": by_difficulty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Помощник библиотекаря для KB")
    parser.add_argument("command", choices=["stats", "pipeline", "kb_report", "full"], help="Команда")
    parser.add_argument("--annotations", type=Path, default=REPO_ROOT / "data" / "MAST_GEMINI" / "test" / "yes_annotations.json", help="Путь к yes_annotations.json")
    parser.add_argument("--crops_dir", type=Path, default=REPO_ROOT / "data" / "MAST_GEMINI" / "test" / "crops", help="Директория кропов")
    parser.add_argument("--kb_path", type=Path, default=REPO_ROOT / "mast_cells_kb", help="Путь к KB")
    parser.add_argument("--crops_only", action="store_true", help="Только кропы (для pipeline)")
    parser.add_argument("--no_clip", action="store_true", help="Без CLIP (для pipeline)")
    args = parser.parse_args()

    if args.command == "stats":
        if not args.annotations.exists():
            raise SystemExit(f"Файл не найден: {args.annotations}")
        s = cmd_stats(args.annotations)
        print("Статистика разметки (yes_annotations.json):")
        print(f"  Всего изображений: {s['total_images']}")
        print(f"  Явные (explicit): {s['total_explicit']}")
        print(f"  Неявные (implicit): {s['total_implicit']}")
        print("  По файлам:")
        for name, e, i in s["per_file"]:
            print(f"    {name}: explicit={e} implicit={i}")
        return

    if args.command == "pipeline":
        if not args.annotations.exists():
            raise SystemExit(f"Файл не найден: {args.annotations}")
        code = cmd_pipeline(args.annotations, args.crops_dir, args.kb_path, args.crops_only, args.no_clip)
        raise SystemExit(code)

    if args.command == "kb_report":
        r = cmd_kb_report(args.kb_path)
        if "error" in r:
            print("Ошибка KB:", r["error"])
            raise SystemExit(1)
        print("Содержимое KB:")
        print(f"  Всего примеров: {r['total']}")
        print("  По cell_type:", r["by_cell_type"])
        print("  По difficulty:", r["by_difficulty"])
        return

    if args.command == "full":
        if not args.annotations.exists():
            raise SystemExit(f"Файл не найден: {args.annotations}")
        s = cmd_stats(args.annotations)
        print("Статистика разметки:", s["total_explicit"], "явных,", s["total_implicit"], "неявных,", s["total_images"], "изображений.")
        code = cmd_pipeline(args.annotations, args.crops_dir, args.kb_path, args.crops_only, args.no_clip)
        if code != 0:
            raise SystemExit(code)
        r = cmd_kb_report(args.kb_path)
        if "error" in r:
            print("KB отчёт недоступен:", r["error"])
        else:
            print("KB после конвейера:", r["total"], "примеров, по cell_type:", r["by_cell_type"])
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()
