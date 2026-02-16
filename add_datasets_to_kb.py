#!/usr/bin/env python3
"""
Скрипт для добавления различных типов датасетов в Knowledge Base
согласно требованиям Gemini 3 Pro.

Поддерживает:
- Hard Positives (трудные позитивы)
- Hard Negatives (трудные негативы)
- Confusion Set (смешанный датасет)
- Ambiguous (неоднозначные случаи)
"""
import json
from pathlib import Path
from train_knowledge_base import MastCellsKnowledgeBase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_hard_positives_from_dir(
    kb: MastCellsKnowledgeBase,
    data_dir: Path,
    hard_positive_type: str
):
    """
    Добавляет Hard Positives из директории
    
    Ожидаемая структура:
    data_dir/
    ├── spindle_shaped/
    │   ├── 001_he.png
    │   ├── 001_ihc.png
    │   └── 001_metadata.json
    """
    type_dir = data_dir / hard_positive_type
    
    if not type_dir.exists():
        logger.warning(f"Directory not found: {type_dir}")
        return
    
    # Ищем все пары изображений
    he_images = sorted(type_dir.glob("*_he.png"))
    
    for he_image in he_images:
        base_name = he_image.stem.replace("_he", "")
        ihc_image = type_dir / f"{base_name}_ihc.png"
        metadata_file = type_dir / f"{base_name}_metadata.json"
        
        if not ihc_image.exists():
            logger.warning(f"IHC image not found for {he_image.name}, skipping")
            continue
        
        # Загружаем метаданные если есть
        morphological_features = {}
        coordinates = None
        tissue_location = None
        inflammation_level = None
        gemini_insights = None
        
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                morphological_features = metadata.get("morphological_features", {})
                coordinates = metadata.get("coordinates")
                tissue_location = metadata.get("tissue_location")
                inflammation_level = metadata.get("inflammation_level")
                gemini_insights = metadata.get("gemini_insights")
        
        kb.add_hard_positive(
            example_id=f"hard_pos_{hard_positive_type}_{base_name}",
            image_path=str(he_image),
            ihc_image_path=str(ihc_image),
            hard_positive_type=hard_positive_type,
            morphological_features=morphological_features,
            coordinates=coordinates,
            tissue_location=tissue_location,
            inflammation_level=inflammation_level,
            gemini_insights=gemini_insights
        )
    
    logger.info(f"Added {len(he_images)} Hard Positives of type {hard_positive_type}")


def add_hard_negatives_from_dir(
    kb: MastCellsKnowledgeBase,
    data_dir: Path,
    cell_type: str,
    hard_negative_type: str
):
    """
    Добавляет Hard Negatives из директории
    
    Ожидаемая структура:
    data_dir/
    ├── plasmocytes_central/
    │   ├── 001_he.png
    │   ├── 001_metadata.json
    │   └── 001_distinguishing.json
    """
    type_dir = data_dir / hard_negative_type
    
    if not type_dir.exists():
        logger.warning(f"Directory not found: {type_dir}")
        return
    
    # Ищем все изображения
    he_images = sorted(type_dir.glob("*_he.png"))
    
    for he_image in he_images:
        base_name = he_image.stem.replace("_he", "")
        metadata_file = type_dir / f"{base_name}_metadata.json"
        distinguishing_file = type_dir / f"{base_name}_distinguishing.json"
        
        # Загружаем метаданные
        morphological_features = {}
        distinguishing_features = {}
        coordinates = None
        gemini_insights = None
        
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                morphological_features = metadata.get("morphological_features", {})
                coordinates = metadata.get("coordinates")
                gemini_insights = metadata.get("gemini_insights")
        
        if distinguishing_file.exists():
            with open(distinguishing_file, "r", encoding="utf-8") as f:
                distinguishing_features = json.load(f)
        
        kb.add_hard_negative(
            example_id=f"hard_neg_{hard_negative_type}_{base_name}",
            image_path=str(he_image),
            cell_type=cell_type,
            hard_negative_type=hard_negative_type,
            morphological_features=morphological_features,
            distinguishing_features=distinguishing_features,
            coordinates=coordinates,
            gemini_insights=gemini_insights
        )
    
    logger.info(f"Added {len(he_images)} Hard Negatives of type {hard_negative_type}")


def add_confusion_set_from_dir(
    kb: MastCellsKnowledgeBase,
    data_dir: Path
):
    """
    Добавляет Confusion Set из директории
    
    Ожидаемая структура:
    data_dir/
    ├── confusion_set/
    │   ├── 001_he.png
    │   ├── 001_ihc.png  # только если мастоцит
    │   └── 001_metadata.json
    """
    confusion_dir = data_dir / "confusion_set"
    
    if not confusion_dir.exists():
        logger.warning(f"Directory not found: {confusion_dir}")
        return
    
    he_images = sorted(confusion_dir.glob("*_he.png"))
    
    for he_image in confusion_dir.glob("*_he.png"):
        base_name = he_image.stem.replace("_he", "")
        ihc_image = confusion_dir / f"{base_name}_ihc.png"
        metadata_file = confusion_dir / f"{base_name}_metadata.json"
        
        # Загружаем метаданные
        morphological_features = {}
        distinguishing_features = None
        cell_type = "unknown"
        coordinates = None
        
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                cell_type = metadata.get("cell_type", "unknown")
                morphological_features = metadata.get("morphological_features", {})
                distinguishing_features = metadata.get("distinguishing_features")
                coordinates = metadata.get("coordinates")
        
        kb.add_confusion_set_example(
            example_id=f"confusion_{base_name}",
            image_path=str(he_image),
            cell_type=cell_type,
            morphological_features=morphological_features,
            distinguishing_features=distinguishing_features,
            ihc_image_path=str(ihc_image) if ihc_image.exists() else None,
            coordinates=coordinates
        )
    
    logger.info(f"Added {len(he_images)} Confusion Set examples")


def main():
    """Основная функция для добавления всех типов датасетов"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add datasets to Knowledge Base")
    parser.add_argument(
        "--kb_path",
        default="./mast_cells_kb",
        help="Path to knowledge base directory"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to datasets directory"
    )
    parser.add_argument(
        "--dataset_type",
        choices=["hard_positives", "hard_negatives", "confusion_set", "all"],
        default="all",
        help="Type of dataset to add"
    )
    parser.add_argument(
        "--use_clip",
        action="store_true",
        default=True,
        help="Use CLIP for image embeddings"
    )
    
    args = parser.parse_args()
    
    # Инициализируем KB
    kb = MastCellsKnowledgeBase(db_path=args.kb_path, use_clip=args.use_clip)
    
    data_dir = Path(args.data_dir)
    
    if args.dataset_type == "all" or args.dataset_type == "hard_positives":
        logger.info("Adding Hard Positives...")
        
        # Типы Hard Positives согласно Gemini
        hard_positive_types = [
            "spindle_shaped",      # Веретеновидные
            "overlapping",          # С наложением
            "degranulated",         # Дегранулированные
            "edge_cases",          # На краях
            "clusters"             # Кластеры
        ]
        
        for hp_type in hard_positive_types:
            add_hard_positives_from_dir(kb, data_dir / "hard_positives", hp_type)
    
    if args.dataset_type == "all" or args.dataset_type == "hard_negatives":
        logger.info("Adding Hard Negatives...")
        
        # Типы Hard Negatives согласно Gemini
        hard_negative_configs = [
            ("plasmocyte", "plasmocytes_central"),
            ("fibroblast", "fibroblasts"),
            ("histiocyte", "histiocytes"),
            ("eosinophil", "eosinophils"),
            ("lymphocyte", "lymphocytes")
        ]
        
        for cell_type, neg_type in hard_negative_configs:
            add_hard_negatives_from_dir(
                kb,
                data_dir / "hard_negatives",
                cell_type,
                neg_type
            )
    
    if args.dataset_type == "all" or args.dataset_type == "confusion_set":
        logger.info("Adding Confusion Set...")
        add_confusion_set_from_dir(kb, data_dir)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

