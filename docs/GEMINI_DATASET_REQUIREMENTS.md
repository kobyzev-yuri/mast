# Требования Gemini 3 к датасетам для Knowledge Base

## Анализ рекомендаций Gemini

На основе детального анализа результатов Gemini 3 Pro выявлены следующие требования к данным для Knowledge Base.

---

## 1. ТИПЫ ДАТАСЕТОВ (по приоритету)

### 1.1. Трудные позитивы (Hard Positives) - ПРИОРИТЕТ ВЫСОКИЙ

**Что это:** Мастоциты, которые сложно обнаружить, но они точно являются мастоцитами (подтверждены ИГХ).

**Типы примеров:**

1. **Веретеновидные мастоциты**
   - Описание: Вытянутая форма, мимикрируют под фибробласты
   - Признаки: Овальное ядро по центру, розовая цитоплазма с размытыми границами
   - Сложность: "hard"
   - Почему важно: Без ИГХ их невозможно отличить от фибробластов

2. **Мастоциты с наложением на другие клетки**
   - Описание: Несколько клеток сливаются, воспринимаются как одна
   - Признаки: Множественные ядра внутри одного облака цитоплазмы
   - Сложность: "hard"
   - Почему важно: Нужно научиться разделять слипшиеся клетки

3. **Дегранулированные мастоциты**
   - Описание: Выбросили гранулы, цитоплазма бледная
   - Признаки: Очень бледная цитоплазма, сливается с отечной стромой
   - Сложность: "hard"
   - Почему важно: На текущем разрешении невидимы без ИГХ

4. **Мастоциты на краях изображения**
   - Описание: Частично срезанные краем кадра
   - Признаки: Нарушена целостность морфологии "глазуньи"
   - Сложность: "medium"
   - Почему важно: Глаз патолога часто игнорирует объекты на границах

5. **Мастоциты в кластерах**
   - Описание: Несколько мастоцитов рядом, цитоплазма сливается
   - Признаки: "Двойное ядро" или "гантелевидная" форма
   - Сложность: "hard"
   - Почему важно: Нужно разделять объекты внутри кластера

**Требования Gemini:**
- Минимум **50-100 примеров** сложных случаев
- Все должны быть подтверждены ИГХ (CD117/Tryptase)
- Парные данные: Г/Э + ИГХ обязательно

---

### 1.2. Трудные негативы (Hard Negatives) - КРИТИЧЕСКИ ВАЖНО

**Что это:** Клетки, которые похожи на мастоциты, но ими НЕ являются. Помечены как "ЭТО НЕ МАСТОЦИТ".

**Типы примеров:**

1. **Плазмоциты с центральным ядром**
   - Описание: Редкий случай, когда ядро по центру (обычно эксцентричное)
   - Отличия от мастоцита:
     - Хроматин "колесо телеги" (у мастоцита более монохромный)
     - Есть светлый "дворик" вокруг ядра
     - Цитоплазма гомогенная (у мастоцита зернистая)
   - Метка: `is_mast_cell: false`, `cell_type: "plasmocyte"`

2. **Активные фибробласты**
   - Описание: Похожи на веретеновидные мастоциты
   - Отличия от мастоцита:
     - Ядро вытянутое, веретеновидное (у мастоцита овальное)
     - Цитоплазма тянется нитями (у мастоцита округлая)
     - Интегрированы в коллагеновые волокна (мастоцит создает пространство)
   - Метка: `is_mast_cell: false`, `cell_type: "fibroblast"`

3. **Гистиоциты (макрофаги)**
   - Описание: Похожи на неявные мастоциты
   - Отличия от мастоцита:
     - Ядро более везикулярное (у мастоцита плотное)
     - Цитоплазма может быть вакуолизированной
     - Нет эффекта "гало"
   - Метка: `is_mast_cell: false`, `cell_type: "histiocyte"`

4. **Эозинофилы**
   - Описание: Ярко-красная цитоплазма, можно спутать
   - Отличия от мастоцита:
     - Цитоплазма ярко-красная и рефрактерная (у мастоцита тусклая розовая)
     - Ядро би-лобарное/сегментированное (у мастоцита округлое)
   - Метка: `is_mast_cell: false`, `cell_type: "eosinophil"`

5. **Активированные лимфоциты**
   - Описание: Могут иметь видимую цитоплазму
   - Отличия от мастоцита:
     - Ядро идеально круглое (у мастоцита овальное)
     - Цитоплазмы почти нет (у мастоцита есть ободок)
     - Хроматин очень плотный (у мастоцита светлее)
   - Метка: `is_mast_cell: false`, `cell_type: "lymphocyte"`

**Требования Gemini:**
- Критически важны для обучения границам различий
- Нужны специально размеченные "ложные цели"
- Метка: `is_mast_cell: false` обязательна

---

### 1.3. Confusion Set - ДЛЯ ТОНКОЙ ДИФФЕРЕНЦИАЦИИ

**Что это:** Смешанный датасет из похожих клеток для обучения различиям.

**Состав:**
- Плазмоциты
- Эозинофилы  
- Мастоциты (явные и неявные)
- Все вперемешку

**Цель:** Научить модель тонкой дифференциации между этими похожими клетками.

**Требования:**
- Все клетки должны быть размечены с правильным типом
- Парные данные (Г/Э + ИГХ) для мастоцитов
- Метаданные о различиях между типами

---

### 1.4. Парные данные (Г/Э + ИГХ) - ОБЯЗАТЕЛЬНО

**Что это:** Тот же участок ткани, окрашенный Г/Э и затем ИГХ (CD117/Tryptase).

**Требования Gemini:**
- **Минимум 50-100 парных патчей** для сложных случаев
- Идеальная регистрация изображений (тот же срез)
- Позволяет найти скрытые паттерны на Г/Э, коррелирующие с ИГХ-позитивностью

**Формат:**
- Г/Э изображение: `example_001_he.png`
- ИГХ изображение: `example_001_ihc.png`
- Метаданные: `paired: true`, `ihc_positive: true/false`

---

### 1.5. Класс "Ambiguous" (Неоднозначный)

**Что это:** Клетки, похожие на мастоциты, но нет уверенности.

**Требования Gemini:**
- Ввести класс "Ambiguous" - желтая метка
- Не заставлять выбирать только Да/Нет
- Позволяет отделить "чистые" данные от "шумных"

**Метка:** `confidence: "ambiguous"` или `cell_type: "ambiguous"`

---

## 2. МЕТАДАННЫЕ ДЛЯ КАЖДОГО ПРИМЕРА

### Обязательные метаданные:

```python
{
    # Идентификация
    "example_id": "hard_positive_001",
    "image_path": "path/to/he_image.png",
    "ihc_image_path": "path/to/ihc_image.png",  # если есть
    
    # Классификация
    "is_mast_cell": true,  # или false для Hard Negatives
    "cell_type": "mast_cell",  # или "plasmocyte", "fibroblast", etc.
    "mast_cell_subtype": "implicit",  # "explicit", "implicit", "ambiguous"
    "difficulty": "hard",  # "easy", "medium", "hard"
    "confidence": "high",  # "high", "medium", "low", "ambiguous"
    
    # Категория датасета
    "dataset_category": "hard_positive",  # "hard_positive", "hard_negative", "confusion_set", "standard"
    "hard_positive_type": "spindle_shaped",  # "spindle_shaped", "overlapping", "degranulated", "edge", "cluster"
    "hard_negative_type": null,  # "plasmocyte_central", "fibroblast", "histiocyte", "eosinophil", "lymphocyte"
    
    # Парные данные
    "has_paired_ihc": true,
    "ihc_positive": true,  # подтверждено ИГХ
    "ihc_marker": "CD117",  # или "Tryptase"
    
    # Морфологические признаки
    "morphological_features": {
        "nucleus": "central, ovoid, hyperchromatic",
        "cytoplasm": "pale pink, blurred boundaries",
        "shape": "spindle-shaped, elongated",
        "location": "stroma, near vessels"
    },
    
    # Контекст
    "tissue_location": "colon_mucosa",  # "stomach", "skin", "lung", etc.
    "inflammation_level": "moderate",  # "none", "mild", "moderate", "severe"
    "tissue_context": "perivascular",  # "perivascular", "subepithelial", "stromal"
    
    # Технические параметры
    "resolution": "40x",  # "20x", "40x", "100x"
    "image_format": "PNG",  # "PNG", "TIFF", "JPEG"
    "compression": "lossless",  # "lossless", "high_quality", "medium"
    
    # Координаты
    "coordinates": {"x": 195, "y": 190},
    
    # Инсайты от Gemini
    "gemini_insights": [
        "Rule of 'Dirty Halo'",
        "Nuclear criterion: ovoid nucleus"
    ],
    
    # Отличия (для Hard Negatives)
    "distinguishing_features": {
        "vs_fibroblasts": "nucleus ovoid not elongated",
        "vs_plasmocytes": "nucleus central not eccentric"
    }
}
```

---

## 3. СТРУКТУРА ДИРЕКТОРИЙ ДЛЯ ДАННЫХ

```
mast_cells_datasets/
├── hard_positives/
│   ├── spindle_shaped/
│   │   ├── 001_he.png
│   │   ├── 001_ihc.png
│   │   └── 001_metadata.json
│   ├── overlapping/
│   ├── degranulated/
│   ├── edge_cases/
│   └── clusters/
│
├── hard_negatives/
│   ├── plasmocytes_central/
│   │   ├── 001_he.png
│   │   ├── 001_metadata.json  # is_mast_cell: false
│   │   └── 001_distinguishing.json
│   ├── fibroblasts/
│   ├── histiocytes/
│   ├── eosinophils/
│   └── lymphocytes/
│
├── confusion_set/
│   ├── mixed_cells/
│   │   ├── 001_he.png
│   │   ├── 001_ihc.png  # если мастоцит
│   │   └── 001_metadata.json
│
├── paired_data/
│   ├── registered_pairs/
│   │   ├── 001_he.png
│   │   ├── 001_ihc.png
│   │   └── 001_registration.json  # параметры регистрации
│
└── ambiguous/
    ├── 001_he.png
    ├── 001_metadata.json  # confidence: "ambiguous"
```

---

## 4. ПРОЦЕСС ДОБАВЛЕНИЯ В KB

### 4.1. Hard Positives

```python
kb.add_hard_positive(
    example_id="spindle_001",
    image_path="hard_positives/spindle_shaped/001_he.png",
    ihc_image_path="hard_positives/spindle_shaped/001_ihc.png",
    hard_positive_type="spindle_shaped",
    morphological_features={
        "nucleus": "central, ovoid, hyperchromatic",
        "cytoplasm": "pale pink, blurred boundaries",
        "shape": "spindle-shaped, elongated",
        "location": "stroma, near vessels"
    },
    ihc_positive=True,
    ihc_marker="CD117",
    difficulty="hard",
    confidence="medium"
)
```

### 4.2. Hard Negatives

```python
kb.add_hard_negative(
    example_id="plasmocyte_central_001",
    image_path="hard_negatives/plasmocytes_central/001_he.png",
    cell_type="plasmocyte",
    hard_negative_type="plasmocyte_central",
    morphological_features={
        "nucleus": "central, round, 'cartwheel' chromatin",
        "cytoplasm": "homogeneous, light halo around nucleus",
        "shape": "round"
    },
    distinguishing_features={
        "vs_mast_cell": "chromatin 'cartwheel' pattern, not monochromatic",
        "key_difference": "eccentric nucleus usually, but here central"
    },
    is_mast_cell=False,
    difficulty="hard"
)
```

### 4.3. Confusion Set

```python
kb.add_confusion_set_example(
    example_id="confusion_001",
    image_path="confusion_set/mixed_cells/001_he.png",
    cell_type="plasmocyte",  # или "mast_cell", "eosinophil"
    ihc_image_path=None,  # только если мастоцит
    morphological_features={...},
    distinguishing_features={
        "similar_to_mast_cell": "central nucleus, visible cytoplasm",
        "different_from_mast_cell": "cartwheel chromatin pattern"
    }
)
```

---

## 5. ТРЕБОВАНИЯ К КАЧЕСТВУ ИЗОБРАЖЕНИЙ

### Технические требования:

1. **Формат:**
   - PNG (lossless) - предпочтительно
   - TIFF - альтернатива
   - JPEG - только high quality (не рекомендуется для неявных мастоцитов)

2. **Разрешение:**
   - Минимум: 40x увеличение
   - Оптимально: 100x с иммерсией
   - Должно позволять видеть хроматин в ядре

3. **Размер патчей:**
   - Минимум: 512x512 пикселей при 40x
   - Оптимально: 1024x1024 или больше

4. **Качество:**
   - Без артефактов сжатия
   - Должна быть видна текстура цитоплазмы (зернистость)
   - Хроматин в ядре различим

---

## 6. КОЛИЧЕСТВО ДАННЫХ

### Минимальные требования:

- **Hard Positives:** 50-100 примеров (сложные случаи)
- **Hard Negatives:** 50-100 примеров (каждый тип)
- **Confusion Set:** 100-200 примеров (смешанные)
- **Парные данные:** 50-100 пар (Г/Э + ИГХ)

### Оптимальные требования:

- **Hard Positives:** 500-1000 примеров
- **Hard Negatives:** 200-500 примеров каждого типа
- **Confusion Set:** 500-1000 примеров
- **Парные данные:** 200-500 пар

---

## 7. ИНТЕГРАЦИЯ В KB

### Структура коллекций в ChromaDB:

```
mast_cells_kb/
├── text_collection/
│   ├── hard_positives/     # Текстовые эмбеддинги
│   ├── hard_negatives/
│   ├── confusion_set/
│   └── standard/
│
└── image_collection/
    ├── hard_positives/     # CLIP эмбеддинги
    ├── hard_negatives/
    ├── confusion_set/
    └── standard/
```

### Фильтры для поиска:

```python
# Поиск Hard Positives определенного типа
kb.search_similar(
    query_text="spindle-shaped mast cell",
    filter_dataset_category="hard_positive",
    filter_hard_positive_type="spindle_shaped"
)

# Поиск Hard Negatives для обучения различиям
kb.search_similar(
    query_text="cell with central nucleus",
    filter_dataset_category="hard_negative",
    filter_hard_negative_type="plasmocyte_central"
)

# Поиск в Confusion Set
kb.search_similar(
    query_text="cell similar to mast cell",
    filter_dataset_category="confusion_set"
)
```

---

## 8. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ

### Приоритеты для сбора данных:

1. **Высокий приоритет:**
   - Hard Positives (веретеновидные, дегранулированные)
   - Hard Negatives (плазмоциты с центральным ядром, фибробласты)
   - Парные данные (Г/Э + ИГХ)

2. **Средний приоритет:**
   - Confusion Set
   - Мастоциты на краях
   - Кластеры

3. **Низкий приоритет:**
   - Стандартные явные мастоциты (уже есть достаточно)

### Процесс разметки:

1. Разметить все клетки с правильным типом
2. Для мастоцитов: указать explicit/implicit/ambiguous
3. Для Hard Negatives: обязательно пометить `is_mast_cell: false`
4. Добавить distinguishing_features для Hard Negatives
5. Убедиться в наличии парных ИГХ для мастоцитов

---

## 9. ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

После добавления всех типов данных в KB:

1. ✅ Улучшение обнаружения неявных мастоцитов
2. ✅ Снижение ложноположительных срабатываний
3. ✅ Улучшение дифференциации похожих клеток
4. ✅ Понимание скрытых паттернов через парные данные
5. ✅ Более точные рекомендации для Gemini

---

## 10. СЛЕДУЮЩИЕ ШАГИ

1. ⏳ Создать структуру директорий для данных
2. ⏳ Разработать скрипты для добавления каждого типа данных
3. ⏳ Начать сбор Hard Positives и Hard Negatives
4. ⏳ Организовать парные данные (Г/Э + ИГХ)
5. ⏳ Создать Confusion Set
6. ⏳ Интегрировать все в KB с правильными метаданными

