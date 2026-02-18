#!/usr/bin/env python3
"""
Кластеризация изображений с разметкой мастоцитов (_yes) по визуальному сходству (CLIP)
и опционально по «подписи» разметки (доли красных/зелёных/синих пикселей).

Цель: выявить группы визуально похожих снимков и кластеры, где разметка может
быть перемешана (явные/неявные/красные), чтобы привести к единому формату.

Использование:
  python scripts/cluster_mast_annotations.py --mast_dir data/MAST_GEMINI/test
  python scripts/cluster_mast_annotations.py --mast_dir data/MAST_GEMINI/test --color_stats --n_clusters 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# проект в корне репозитория
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Опциональные зависимости
CLIP_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    CLIP_AVAILABLE = True
except ImportError:
    pass

SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    pass


# Пороги для подсчёта пикселей разметки (RGB). Аннотации в _yes.png могут быть
# не чистыми (255,0,0)/(0,255,0)/(0,0,255) — подстройте под свои цвета или
# уменьшите *_OTHER_MAX / увеличьте пороги доминантного канала.
GREEN_OTHER_MAX = 120   # макс. R и B для «зелёного» пикселя
BLUE_OTHER_MAX = 120    # макс. R и G для «синего»
RED_OTHER_MAX = 120     # макс. G и B для «красного»
GREEN_MIN = 200         # мин. G для зелёного
BLUE_MIN = 200         # мин. B для синего
RED_MIN = 200          # мин. R для красного


def count_annotation_pixels(image_path: Path) -> dict[str, int]:
    """Считает пиксели, похожие на красную/зелёную/синюю разметку (аннотации)."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    green = np.sum((g >= GREEN_MIN) & (r <= GREEN_OTHER_MAX) & (b <= GREEN_OTHER_MAX))
    blue = np.sum((b >= BLUE_MIN) & (r <= BLUE_OTHER_MAX) & (g <= BLUE_OTHER_MAX))
    red = np.sum((r >= RED_MIN) & (g <= RED_OTHER_MAX) & (b <= RED_OTHER_MAX))

    return {"red": int(red), "green": int(green), "blue": int(blue)}


def load_yes_images(mast_dir: Path) -> list[Path]:
    """Возвращает отсортированный список путей к *_yes.png в mast_dir."""
    images = sorted(mast_dir.glob("*_yes.png"))
    return images


def embed_images_clip(image_paths: list[Path], model_name: str = "clip-ViT-B-32") -> np.ndarray:
    """Строит CLIP-эмбеддинги для списка изображений."""
    if not CLIP_AVAILABLE:
        raise RuntimeError("sentence_transformers не установлен. Установите: pip install sentence-transformers")
    model = SentenceTransformer(model_name)
    embeddings = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        emb = model.encode(img)
        embeddings.append(emb)
    return np.array(embeddings)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """KMeans по эмбеддингам. Возвращает метки кластеров и центроиды."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn не установлен. Установите: pip install scikit-learn")
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_


def main() -> None:
    parser = argparse.ArgumentParser(description="Кластеризация изображений _yes по CLIP и/или по цветам разметки")
    parser.add_argument("--mast_dir", type=Path, default=REPO_ROOT / "data" / "MAST_GEMINI" / "test", help="Директория с *_yes.png")
    parser.add_argument("--out", type=Path, default=None, help="Путь к JSON/текстовому отчёту (по умолчанию stdout + mast_dir/cluster_report.json)")
    parser.add_argument("--n_clusters", type=int, default=3, help="Число кластеров для KMeans по CLIP")
    parser.add_argument("--color_stats", action="store_true", help="Подсчитать красные/зелёные/синие пиксели разметки по каждому изображению")
    parser.add_argument("--no_clip", action="store_true", help="Не использовать CLIP (только color_stats)")
    args = parser.parse_args()

    mast_dir = args.mast_dir
    if not mast_dir.is_dir():
        logger.error("Директория не найдена: %s", mast_dir)
        sys.exit(1)

    images = load_yes_images(mast_dir)
    if not images:
        logger.error("В директории нет файлов *_yes.png: %s", mast_dir)
        sys.exit(1)
    logger.info("Найдено изображений _yes: %d", len(images))

    report = {
        "mast_dir": str(mast_dir),
        "images": [p.name for p in images],
        "clusters_by_clip": None,
        "color_stats": None,
    }

    # Цветовая статистика по разметке
    if args.color_stats:
        color_stats = []
        for path in images:
            counts = count_annotation_pixels(path)
            color_stats.append({"file": path.name, **counts})
        report["color_stats"] = color_stats
        logger.info("Цветовая статистика разметки (red/green/blue пиксели):")
        for row in color_stats:
            logger.info("  %s: red=%d green=%d blue=%d", row["file"], row["red"], row["green"], row["blue"])

    # Кластеризация по CLIP
    if not args.no_clip and CLIP_AVAILABLE and SKLEARN_AVAILABLE:
        logger.info("Строим CLIP-эмбеддинги...")
        embeddings = embed_images_clip(images)
        labels, _ = cluster_embeddings(embeddings, args.n_clusters)
        clusters = {}
        for idx, path in enumerate(images):
            c = int(labels[idx])
            clusters.setdefault(c, []).append(path.name)
        report["clusters_by_clip"] = {f"cluster_{k}": v for k, v in sorted(clusters.items())}
        logger.info("Кластеры по CLIP (n_clusters=%d):", args.n_clusters)
        for k, v in sorted(clusters.items()):
            logger.info("  Кластер %d: %s", k, ", ".join(v))
    elif not args.no_clip and not CLIP_AVAILABLE:
        logger.warning("CLIP недоступен (sentence_transformers). Запустите с --no_clip или установите sentence-transformers.")
    elif not args.no_clip and not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn недоступен. Кластеризация по CLIP пропущена.")

    out_path = args.out or mast_dir / "cluster_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Отчёт сохранён: %s", out_path)

    if report.get("clusters_by_clip"):
        print("\nРекомендация: для кластеров, где в одном кластере оказались снимки с разной разметкой (по color_stats), имеет смысл проверить единообразие и при необходимости привести к формату явные/неявные (и решить, как трактовать красный).")


if __name__ == "__main__":
    main()
