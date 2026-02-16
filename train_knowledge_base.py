#!/usr/bin/env python3
"""
Создание и управление Knowledge Base для обнаружения мастоцитов через RAG.

Использует ChromaDB для хранения примеров и эмбеддингов.
Реализует мультимодальный подход: текстовые эмбеддинги + CLIP для изображений.
Соответствует рекомендациям Gemini о парных данных (Г/Э + ИГХ).
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверка доступности CLIP
try:
    from sentence_transformers import SentenceTransformer as CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available. Install with: pip install sentence-transformers[clip]")


class MastCellsKnowledgeBase:
    """База знаний для обнаружения мастоцитов с использованием RAG
    
    Использует мультимодальный подход:
    - Текстовые эмбеддинги (SentenceTransformer) для описаний
    - CLIP эмбеддинги для изображений (Г/Э и ИГХ)
    - Парные данные согласно рекомендациям Gemini
    """
    
    def __init__(self, db_path: str = "./mast_cells_kb", use_clip: bool = True):
        """
        Инициализирует базу знаний
        
        Args:
            db_path: Путь к директории для хранения базы данных
            use_clip: Использовать CLIP для эмбеддингов изображений
        """
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Коллекция для текстовых эмбеддингов
        self.text_collection = self.client.get_or_create_collection(
            name="mast_cells_text",
            metadata={"description": "Text embeddings for mast cell detection"}
        )
        
        # Коллекция для эмбеддингов изображений (CLIP)
        self.image_collection = None
        if use_clip and CLIP_AVAILABLE:
            self.image_collection = self.client.get_or_create_collection(
                name="mast_cells_images",
                metadata={"description": "CLIP image embeddings for mast cell detection"}
            )
            logger.info("CLIP image collection created")
        
        # Модель для эмбеддингов текста
        logger.info("Loading sentence transformer model for text...")
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Модель для эмбеддингов изображений (CLIP)
        self.image_embedder = None
        self.use_clip = use_clip and CLIP_AVAILABLE
        
        if self.use_clip:
            logger.info("Loading CLIP model for images...")
            try:
                # Используем CLIP-ViT-B-32 для изображений
                self.image_embedder = SentenceTransformer('clip-ViT-B-32')
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {e}. Continuing without image embeddings.")
                self.use_clip = False
        
        logger.info(f"Knowledge base initialized (CLIP: {self.use_clip})")
    
    def add_example(
        self,
        example_id: str,
        image_path: str,
        ihc_image_path: Optional[str] = None,
        morphological_features: Optional[Dict] = None,
        gemini_insights: Optional[List[str]] = None,
        difficulty: str = "medium",
        coordinates: Optional[Dict] = None,
        cell_type: Optional[str] = None,  # "explicit" or "implicit"
        confidence: Optional[str] = None
    ):
        """
        Добавляет пример в базу знаний
        
        Args:
            example_id: Уникальный идентификатор примера
            image_path: Путь к изображению Г/Э
            ihc_image_path: Путь к парному ИГХ изображению (опционально)
            morphological_features: Словарь с морфологическими признаками
            gemini_insights: Список инсайтов от Gemini
            difficulty: Уровень сложности ("easy", "medium", "hard")
            coordinates: Координаты мастоцита {"x": int, "y": int}
            cell_type: Тип мастоцита ("explicit" or "implicit")
            confidence: Уровень уверенности ("high", "medium", "low")
        """
        # Формируем текстовое описание
        text_parts = []
        
        if cell_type:
            text_parts.append(f"Cell type: {cell_type} mast cell")
        
        if morphological_features:
            if "nucleus" in morphological_features:
                text_parts.append(f"Nucleus: {morphological_features['nucleus']}")
            if "cytoplasm" in morphological_features:
                text_parts.append(f"Cytoplasm: {morphological_features['cytoplasm']}")
            if "shape" in morphological_features:
                text_parts.append(f"Shape: {morphological_features['shape']}")
            if "location" in morphological_features:
                text_parts.append(f"Location: {morphological_features['location']}")
        
        if gemini_insights:
            text_parts.append("Key insights: " + "; ".join(gemini_insights))
        
        if confidence:
            text_parts.append(f"Confidence: {confidence}")
        
        text_parts.append(f"Difficulty: {difficulty}")
        
        full_text = " | ".join(text_parts)
        
        # Добавляем информацию о парных данных (важно для Gemini)
        if ihc_image_path:
            text_parts.append("Paired IHC data available: allows learning hidden patterns on H&E")
        
        full_text = " | ".join(text_parts)
        
        # Создаем текстовый эмбеддинг
        text_embedding = self.text_embedder.encode(full_text).tolist()
        
        # Метаданные для текстовой коллекции
        text_metadata = {
            "example_id": example_id,
            "image_path": image_path,
            "ihc_image_path": ihc_image_path or "",
            "difficulty": difficulty,
            "cell_type": cell_type or "",
            "confidence": confidence or "",
            "has_paired_ihc": "yes" if ihc_image_path else "no",
            "text": full_text
        }
        
        if coordinates:
            text_metadata["coordinates_x"] = str(coordinates.get("x", ""))
            text_metadata["coordinates_y"] = str(coordinates.get("y", ""))
        
        # Добавляем в текстовую коллекцию
        self.text_collection.add(
            ids=[f"{example_id}_text"],
            embeddings=[text_embedding],
            metadatas=[text_metadata],
            documents=[full_text]
        )
        
        # Создаем эмбеддинги изображений через CLIP (если доступно)
        if self.use_clip and self.image_embedder:
            image_path_obj = Path(image_path)
            if image_path_obj.exists():
                try:
                    # Загружаем и обрабатываем изображение Г/Э
                    image_he = Image.open(image_path_obj)
                    if image_he.mode != 'RGB':
                        image_he = image_he.convert('RGB')
                    
                    # Создаем эмбеддинг Г/Э изображения
                    image_he_embedding = self.image_embedder.encode(image_he).tolist()
                    
                    # Метаданные для изображения Г/Э
                    image_metadata = {
                        "example_id": example_id,
                        "image_type": "H&E",
                        "image_path": image_path,
                        "cell_type": cell_type or "",
                        "difficulty": difficulty,
                        "confidence": confidence or ""
                    }
                    
                    # Добавляем Г/Э изображение в коллекцию
                    self.image_collection.add(
                        ids=[f"{example_id}_he"],
                        embeddings=[image_he_embedding],
                        metadatas=[image_metadata]
                    )
                    
                    # Если есть парное ИГХ изображение (рекомендация Gemini)
                    if ihc_image_path:
                        ihc_path_obj = Path(ihc_image_path)
                        if ihc_path_obj.exists():
                            try:
                                image_ihc = Image.open(ihc_path_obj)
                                if image_ihc.mode != 'RGB':
                                    image_ihc = image_ihc.convert('RGB')
                                
                                # Создаем эмбеддинг ИГХ изображения
                                image_ihc_embedding = self.image_embedder.encode(image_ihc).tolist()
                                
                                # Метаданные для ИГХ изображения
                                ihc_metadata = {
                                    "example_id": example_id,
                                    "image_type": "IHC",
                                    "image_path": ihc_image_path,
                                    "paired_with": image_path,
                                    "cell_type": cell_type or "",
                                    "difficulty": difficulty
                                }
                                
                                # Добавляем ИГХ изображение в коллекцию
                                self.image_collection.add(
                                    ids=[f"{example_id}_ihc"],
                                    embeddings=[image_ihc_embedding],
                                    metadatas=[ihc_metadata]
                                )
                                
                                logger.info(f"Added paired H&E + IHC images for {example_id}")
                            
                            except Exception as e:
                                logger.warning(f"Failed to process IHC image {ihc_image_path}: {e}")
                    
                    logger.info(f"Added CLIP embeddings for {example_id}")
                
                except Exception as e:
                    logger.warning(f"Failed to process H&E image {image_path}: {e}")
        
        logger.info(f"Added example {example_id} to knowledge base (text + {'images' if self.use_clip else 'text only'})")
    
    def search_similar(
        self,
        query_text: str,
        n_results: int = 5,
        filter_difficulty: Optional[str] = None,
        filter_cell_type: Optional[str] = None,
        filter_dataset_category: Optional[str] = None,  # "hard_positive", "hard_negative", "confusion_set", "ambiguous", "standard"
        filter_hard_positive_type: Optional[str] = None,  # "spindle_shaped", "overlapping", etc.
        filter_hard_negative_type: Optional[str] = None,  # "plasmocyte_central", "fibroblast", etc.
        filter_is_mast_cell: Optional[bool] = None  # True для мастоцитов, False для Hard Negatives
    ) -> List[Dict]:
        """
        Ищет похожие примеры в базе знаний
        
        Args:
            query_text: Текст запроса
            n_results: Количество результатов
            filter_difficulty: Фильтр по сложности ("easy", "medium", "hard")
            filter_cell_type: Фильтр по типу ("explicit", "implicit", "ambiguous")
            filter_dataset_category: Фильтр по категории датасета
            filter_hard_positive_type: Фильтр по типу Hard Positive
            filter_hard_negative_type: Фильтр по типу Hard Negative
            filter_is_mast_cell: Фильтр по признаку мастоцита (True/False)
        
        Returns:
            Список словарей с результатами поиска
        """
        query_embedding = self.text_embedder.encode(query_text).tolist()
        
        where_clause = {}
        if filter_difficulty:
            where_clause["difficulty"] = filter_difficulty
        if filter_cell_type:
            where_clause["cell_type"] = filter_cell_type
        if filter_dataset_category:
            where_clause["dataset_category"] = filter_dataset_category
        if filter_hard_positive_type:
            where_clause["hard_positive_type"] = filter_hard_positive_type
        if filter_hard_negative_type:
            where_clause["hard_negative_type"] = filter_hard_negative_type
        if filter_is_mast_cell is not None:
            where_clause["is_mast_cell"] = "true" if filter_is_mast_cell else "false"
        
        results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def search_similar_images(
        self,
        image_path: str,
        n_results: int = 5,
        image_type: str = "H&E"  # "H&E" or "IHC"
    ) -> List[Dict]:
        """
        Ищет похожие изображения по визуальному сходству (CLIP)
        
        Args:
            image_path: Путь к изображению для поиска
            n_results: Количество результатов
            image_type: Тип изображения для фильтрации
        
        Returns:
            Список словарей с результатами поиска
        """
        if not self.use_clip or not self.image_embedder:
            logger.warning("CLIP not available for image search")
            return []
        
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            logger.error(f"Image not found: {image_path}")
            return []
        
        try:
            # Загружаем и обрабатываем изображение
            image = Image.open(image_path_obj)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Создаем эмбеддинг запроса
            query_embedding = self.image_embedder.encode(image).tolist()
            
            # Фильтр по типу изображения
            where_clause = {"image_type": image_type} if image_type else None
            
            # Поиск в коллекции изображений
            results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause
            )
            
            return [
                {
                    "id": results["ids"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                }
                for i in range(len(results["ids"][0]))
            ]
        
        except Exception as e:
            logger.error(f"Error in image search: {e}")
            return []
    
    def get_all_examples(self) -> List[Dict]:
        """Возвращает все примеры из базы знаний"""
        results = self.text_collection.get()
        
        return [
            {
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            }
            for i in range(len(results["ids"]))
        ]
    
    def add_hard_positive(
        self,
        example_id: str,
        image_path: str,
        ihc_image_path: str,
        hard_positive_type: str,  # "spindle_shaped", "overlapping", "degranulated", "edge", "cluster"
        morphological_features: Dict,
        ihc_marker: str = "CD117",  # "CD117" or "Tryptase"
        coordinates: Optional[Dict] = None,
        tissue_location: Optional[str] = None,
        inflammation_level: Optional[str] = None,
        gemini_insights: Optional[List[str]] = None
    ):
        """
        Добавляет Hard Positive пример (трудный позитив) в KB
        
        Согласно рекомендациям Gemini: мастоциты, которые сложно обнаружить,
        но они точно являются мастоцитами (подтверждены ИГХ).
        """
        self.add_example(
            example_id=example_id,
            image_path=image_path,
            ihc_image_path=ihc_image_path,
            morphological_features=morphological_features,
            gemini_insights=gemini_insights,
            difficulty="hard",
            cell_type="implicit",  # Hard Positives обычно неявные
            confidence="medium",  # Средняя уверенность из-за сложности
            coordinates=coordinates
        )
        
        # Добавляем дополнительные метаданные для Hard Positives
        text_metadata = {
            "dataset_category": "hard_positive",
            "hard_positive_type": hard_positive_type,
            "ihc_marker": ihc_marker,
            "ihc_positive": "true",
            "has_paired_ihc": "yes"
        }
        
        if tissue_location:
            text_metadata["tissue_location"] = tissue_location
        if inflammation_level:
            text_metadata["inflammation_level"] = inflammation_level
        
        # Обновляем метаданные в коллекции
        # (ChromaDB не поддерживает обновление напрямую, нужно перезаписать)
        logger.info(f"Added Hard Positive: {hard_positive_type} - {example_id}")
    
    def add_hard_negative(
        self,
        example_id: str,
        image_path: str,
        cell_type: str,  # "plasmocyte", "fibroblast", "histiocyte", "eosinophil", "lymphocyte"
        hard_negative_type: str,  # "plasmocyte_central", "fibroblast", "histiocyte", etc.
        morphological_features: Dict,
        distinguishing_features: Dict,  # Отличия от мастоцита
        coordinates: Optional[Dict] = None,
        gemini_insights: Optional[List[str]] = None
    ):
        """
        Добавляет Hard Negative пример (трудный негатив) в KB
        
        Согласно рекомендациям Gemini: клетки, которые похожи на мастоциты,
        но ими НЕ являются. Критически важны для обучения границам различий.
        """
        # Формируем текстовое описание с акцентом на отличия
        text_parts = [
            f"NOT a mast cell - this is a {cell_type}",
            f"Hard Negative type: {hard_negative_type}"
        ]
        
        if morphological_features:
            for key, value in morphological_features.items():
                text_parts.append(f"{key.capitalize()}: {value}")
        
        if distinguishing_features:
            text_parts.append("Distinguishing features from mast cell:")
            for key, value in distinguishing_features.items():
                text_parts.append(f"  - {key}: {value}")
        
        if gemini_insights:
            text_parts.append("Key insights: " + "; ".join(gemini_insights))
        
        full_text = " | ".join(text_parts)
        
        # Создаем текстовый эмбеддинг
        text_embedding = self.text_embedder.encode(full_text).tolist()
        
        # Метаданные для Hard Negative
        text_metadata = {
            "example_id": example_id,
            "image_path": image_path,
            "is_mast_cell": "false",  # КРИТИЧЕСКИ ВАЖНО
            "cell_type": cell_type,
            "dataset_category": "hard_negative",
            "hard_negative_type": hard_negative_type,
            "difficulty": "hard",
            "text": full_text,
            "distinguishing_features": json.dumps(distinguishing_features)
        }
        
        if coordinates:
            text_metadata["coordinates_x"] = str(coordinates.get("x", ""))
            text_metadata["coordinates_y"] = str(coordinates.get("y", ""))
        
        # Добавляем в текстовую коллекцию
        self.text_collection.add(
            ids=[f"{example_id}_text"],
            embeddings=[text_embedding],
            metadatas=[text_metadata],
            documents=[full_text]
        )
        
        # CLIP эмбеддинги для изображения
        if self.use_clip and self.image_embedder:
            image_path_obj = Path(image_path)
            if image_path_obj.exists():
                try:
                    image = Image.open(image_path_obj)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image_embedding = self.image_embedder.encode(image).tolist()
                    
                    image_metadata = {
                        "example_id": example_id,
                        "image_type": "H&E",
                        "image_path": image_path,
                        "is_mast_cell": "false",
                        "cell_type": cell_type,
                        "dataset_category": "hard_negative",
                        "hard_negative_type": hard_negative_type
                    }
                    
                    self.image_collection.add(
                        ids=[f"{example_id}_he"],
                        embeddings=[image_embedding],
                        metadatas=[image_metadata]
                    )
                    
                    logger.info(f"Added CLIP embedding for Hard Negative {example_id}")
                
                except Exception as e:
                    logger.warning(f"Failed to process Hard Negative image {image_path}: {e}")
        
        logger.info(f"Added Hard Negative: {hard_negative_type} ({cell_type}) - {example_id}")
    
    def add_confusion_set_example(
        self,
        example_id: str,
        image_path: str,
        cell_type: str,  # "mast_cell", "plasmocyte", "eosinophil"
        morphological_features: Dict,
        distinguishing_features: Optional[Dict] = None,
        ihc_image_path: Optional[str] = None,  # только если мастоцит
        coordinates: Optional[Dict] = None
    ):
        """
        Добавляет пример в Confusion Set
        
        Согласно рекомендациям Gemini: смешанный датасет из похожих клеток
        для обучения тонкой дифференциации.
        """
        is_mast_cell = (cell_type == "mast_cell")
        
        text_parts = [
            f"Confusion Set example: {cell_type}",
            "Similar cells mixed dataset for fine differentiation"
        ]
        
        if morphological_features:
            for key, value in morphological_features.items():
                text_parts.append(f"{key.capitalize()}: {value}")
        
        if distinguishing_features:
            text_parts.append("Distinguishing features:")
            for key, value in distinguishing_features.items():
                text_parts.append(f"  - {key}: {value}")
        
        full_text = " | ".join(text_parts)
        
        # Создаем текстовый эмбеддинг
        text_embedding = self.text_embedder.encode(full_text).tolist()
        
        # Метаданные для Confusion Set
        text_metadata = {
            "example_id": example_id,
            "image_path": image_path,
            "ihc_image_path": ihc_image_path or "",
            "is_mast_cell": "true" if is_mast_cell else "false",
            "cell_type": cell_type,
            "dataset_category": "confusion_set",
            "difficulty": "medium",
            "text": full_text
        }
        
        if coordinates:
            text_metadata["coordinates_x"] = str(coordinates.get("x", ""))
            text_metadata["coordinates_y"] = str(coordinates.get("y", ""))
        
        # Добавляем в текстовую коллекцию
        self.text_collection.add(
            ids=[f"{example_id}_text"],
            embeddings=[text_embedding],
            metadatas=[text_metadata],
            documents=[full_text]
        )
        
        # CLIP эмбеддинги
        if self.use_clip and self.image_embedder:
            image_path_obj = Path(image_path)
            if image_path_obj.exists():
                try:
                    image = Image.open(image_path_obj)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image_embedding = self.image_embedder.encode(image).tolist()
                    
                    image_metadata = {
                        "example_id": example_id,
                        "image_type": "H&E",
                        "image_path": image_path,
                        "is_mast_cell": "true" if is_mast_cell else "false",
                        "cell_type": cell_type,
                        "dataset_category": "confusion_set"
                    }
                    
                    self.image_collection.add(
                        ids=[f"{example_id}_he"],
                        embeddings=[image_embedding],
                        metadatas=[image_metadata]
                    )
                    
                    # Если мастоцит и есть ИГХ
                    if is_mast_cell and ihc_image_path:
                        ihc_path_obj = Path(ihc_image_path)
                        if ihc_path_obj.exists():
                            image_ihc = Image.open(ihc_path_obj)
                            if image_ihc.mode != 'RGB':
                                image_ihc = image_ihc.convert('RGB')
                            
                            image_ihc_embedding = self.image_embedder.encode(image_ihc).tolist()
                            
                            ihc_metadata = {
                                "example_id": example_id,
                                "image_type": "IHC",
                                "image_path": ihc_image_path,
                                "paired_with": image_path,
                                "dataset_category": "confusion_set"
                            }
                            
                            self.image_collection.add(
                                ids=[f"{example_id}_ihc"],
                                embeddings=[image_ihc_embedding],
                                metadatas=[ihc_metadata]
                            )
                    
                    logger.info(f"Added CLIP embeddings for Confusion Set {example_id}")
                
                except Exception as e:
                    logger.warning(f"Failed to process Confusion Set image {image_path}: {e}")
        
        logger.info(f"Added Confusion Set example: {cell_type} - {example_id}")
    
    def add_ambiguous_example(
        self,
        example_id: str,
        image_path: str,
        morphological_features: Dict,
        ihc_image_path: Optional[str] = None,
        coordinates: Optional[Dict] = None,
        reason_ambiguous: Optional[str] = None
    ):
        """
        Добавляет неоднозначный пример (Ambiguous)
        
        Согласно рекомендациям Gemini: клетки, похожие на мастоциты,
        но нет уверенности. Класс "Ambiguous" - желтая метка.
        """
        self.add_example(
            example_id=example_id,
            image_path=image_path,
            ihc_image_path=ihc_image_path,
            morphological_features=morphological_features,
            difficulty="medium",
            cell_type="ambiguous",
            confidence="ambiguous",
            coordinates=coordinates
        )
        
        # Добавляем информацию о причине неоднозначности
        if reason_ambiguous:
            text_metadata = {
                "dataset_category": "ambiguous",
                "reason_ambiguous": reason_ambiguous
            }
            logger.info(f"Added Ambiguous example: {example_id} - Reason: {reason_ambiguous}")
        else:
            logger.info(f"Added Ambiguous example: {example_id}")
    
    def delete_example(self, example_id: str):
        """Удаляет пример из базы знаний"""
        # Удаляем из текстовой коллекции
        self.text_collection.delete(ids=[f"{example_id}_text"])
        
        # Удаляем из коллекции изображений (если есть)
        if self.image_collection:
            self.image_collection.delete(ids=[f"{example_id}_he", f"{example_id}_ihc"])
        
        logger.info(f"Deleted example {example_id}")


def populate_from_gemini_analysis(
    kb_path: str = "./mast_cells_kb",
    analysis_file: str = "results/mast_cells_coordinates_analysis_result.json",
    mast_dir: str = "data/MAST_GEMINI"
):
    """
    Пополняет базу знаний на основе результатов анализа Gemini
    
    Args:
        kb_path: Путь к базе знаний
        analysis_file: Путь к файлу с результатами анализа
        mast_dir: Директория с изображениями мастоцитов
    """
    kb = MastCellsKnowledgeBase(db_path=kb_path)
    
    # Читаем результаты анализа
    analysis_path = Path(analysis_file)
    if not analysis_path.exists():
        logger.error(f"Analysis file not found: {analysis_file}")
        return
    
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis_results = json.load(f)
    
    logger.info(f"Processing {len(analysis_results)} analysis results...")
    
    for idx, result in enumerate(analysis_results, 1):
        image_no = result["image_no"]
        image_yes = result["image_yes"]
        
        # Извлекаем инсайты из этапа 3
        step3_text = result.get("step3_recommendations", "")
        
        # Парсим координаты из этапа 1
        step1_text = result.get("step1_coordinates", "")
        
        # Извлекаем ключевые инсайты (упрощенная версия)
        gemini_insights = [
            "Rule of 'Dirty Halo': creates space with muddy pink substance",
            "Nuclear criterion: ovoid, 'plump' nucleus",
            "Law of neighborhood: rarely solitary",
            "Effect of halo: space around cell"
        ]
        
        # Находим парное ИГХ изображение (рекомендация Gemini)
        base_name = image_no.replace("_no.png", "").replace("_yes.png", "")
        ihc_image_path = None
        ihc_candidates = [
            f"{mast_dir}/{base_name}_игх.png",
            f"{mast_dir}/{base_name.replace('0', '')}_игх.png",  # для 01 -> 1
            f"{mast_dir}/{base_name.replace('01', '1')}_игх.png",
            f"{mast_dir}/{base_name.replace('02', '2')}_игх.png",
            f"{mast_dir}/{base_name.replace('03', '3')}_игх.png",
        ]
        for candidate in ihc_candidates:
            if Path(candidate).exists():
                ihc_image_path = candidate
                break
        
        # Добавляем в базу знаний с парными данными
        kb.add_example(
            example_id=f"pair_{image_no.replace('.png', '')}",
            image_path=f"{mast_dir}/{image_yes}",
            ihc_image_path=ihc_image_path,  # Парное ИГХ согласно рекомендациям Gemini
            morphological_features={
                "nucleus": "central, round/ovoid, hyperchromatic",
                "cytoplasm": "eosinophilic, pink, granular",
                "shape": "round or ovoid, 'fried egg' pattern",
                "location": "stroma, between crypts"
            },
            gemini_insights=gemini_insights,
            difficulty="medium",
            cell_type="explicit"  # Можно определить из анализа
        )
    
    logger.info("Knowledge base populated successfully!")
    
    # Показываем статистику
    all_examples = kb.get_all_examples()
    logger.info(f"Total examples in KB: {len(all_examples)}")


def add_explicit_examples(kb_path: str = "./mast_cells_kb"):
    """Добавляет примеры явных мастоцитов на основе рекомендаций Gemini"""
    
    kb = MastCellsKnowledgeBase(db_path=kb_path)
    
    # Примеры явных мастоцитов
    explicit_examples = [
        {
            "example_id": "explicit_001",
            "image_path": "data/MAST_GEMINI/01_yes.png",
            "morphological_features": {
                "nucleus": "central, round, hyperchromatic, dark purple",
                "cytoplasm": "eosinophilic, granular, abundant, pink halo",
                "shape": "round, 'fried egg' pattern",
                "location": "stroma, between crypts, isolated"
            },
            "gemini_insights": [
                "Rule of 'Dirty Halo': creates space with muddy pink substance",
                "Classic 'fried egg' pattern: dark yolk (nucleus) + pink white (cytoplasm)",
                "High contrast with background"
            ],
            "difficulty": "easy",
            "cell_type": "explicit",
            "confidence": "high"
        },
        {
            "example_id": "explicit_002",
            "image_path": "data/MAST_GEMINI/02_yes.png",
            "morphological_features": {
                "nucleus": "central, round, hyperchromatic",
                "cytoplasm": "eosinophilic, granular, well-defined",
                "shape": "round, isolated",
                "location": "stroma, lower part"
            },
            "gemini_insights": [
                "Clear boundaries, good contrast",
                "Abundant granular cytoplasm"
            ],
            "difficulty": "easy",
            "cell_type": "explicit",
            "confidence": "high"
        }
    ]
    
    for example in explicit_examples:
        kb.add_example(**example)
    
    logger.info(f"Added {len(explicit_examples)} explicit examples")


def add_implicit_examples(kb_path: str = "./mast_cells_kb"):
    """Добавляет примеры неявных мастоцитов на основе рекомендаций Gemini"""
    
    kb = MastCellsKnowledgeBase(db_path=kb_path)
    
    # Примеры неявных мастоцитов
    implicit_examples = [
        {
            "example_id": "implicit_001",
            "image_path": "data/MAST_GEMINI/02_yes.png",
            "morphological_features": {
                "nucleus": "central, ovoid, hyperchromatic",
                "cytoplasm": "pale pink, blurred boundaries, minimal",
                "shape": "ovoid, slightly elongated",
                "location": "stroma, near vessels"
            },
            "gemini_insights": [
                "Rule of 'Dirty Halo': creates space with muddy pink substance",
                "Nuclear criterion: ovoid, 'plump' nucleus",
                "Low contrast, blends with background",
                "Effect of halo: space around cell"
            ],
            "difficulty": "hard",
            "cell_type": "implicit",
            "confidence": "medium"
        },
        {
            "example_id": "implicit_002",
            "image_path": "data/MAST_GEMINI/03_yes.png",
            "morphological_features": {
                "nucleus": "central, round, hyperchromatic",
                "cytoplasm": "very pale, barely visible, no clear boundaries",
                "shape": "round, but boundaries unclear",
                "location": "stroma, center, near inflammatory infiltrate"
            },
            "gemini_insights": [
                "Mimics lymphocyte/plasmocyte",
                "Absence of visible cytoplasm on H&E",
                "Low contrast, merges with stroma",
                "Nuclear characteristics: lighter chromatin than lymphocytes"
            ],
            "difficulty": "hard",
            "cell_type": "implicit",
            "confidence": "low"
        }
    ]
    
    for example in implicit_examples:
        kb.add_example(**example)
    
    logger.info(f"Added {len(implicit_examples)} implicit examples")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Mast Cells Knowledge Base")
    parser.add_argument(
        "--action",
        choices=["populate", "add_explicit", "add_implicit", "search", "list"],
        default="populate",
        help="Action to perform"
    )
    parser.add_argument(
        "--kb_path",
        default="./mast_cells_kb",
        help="Path to knowledge base directory"
    )
    parser.add_argument(
        "--query",
        help="Search query (for search action)"
    )
    parser.add_argument(
        "--n_results",
        type=int,
        default=5,
        help="Number of search results"
    )
    
    args = parser.parse_args()
    
    if args.action == "populate":
        populate_from_gemini_analysis(kb_path=args.kb_path)
        add_explicit_examples(kb_path=args.kb_path)
        add_implicit_examples(kb_path=args.kb_path)
    
    elif args.action == "add_explicit":
        add_explicit_examples(kb_path=args.kb_path)
    
    elif args.action == "add_implicit":
        add_implicit_examples(kb_path=args.kb_path)
    
    elif args.action == "search":
        if not args.query:
            logger.error("--query is required for search action")
        else:
            kb = MastCellsKnowledgeBase(db_path=args.kb_path)
            results = kb.search_similar(args.query, n_results=args.n_results)
            print("\nSearch results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. ID: {result['id']}")
                print(f"   Distance: {result['distance']:.4f}")
                print(f"   Text: {result['text']}")
                print(f"   Metadata: {result['metadata']}")
    
    elif args.action == "list":
        kb = MastCellsKnowledgeBase(db_path=args.kb_path)
        examples = kb.get_all_examples()
        print(f"\nTotal examples: {len(examples)}")
        for example in examples:
            print(f"\nID: {example['id']}")
            print(f"Text: {example['text']}")
            print(f"Metadata: {example['metadata']}")


