# Датасеты для Knowledge Base согласно требованиям Gemini 3

## Краткая сводка требований

Gemini 3 Pro четко указала на необходимость следующих типов данных для улучшения обнаружения неявных мастоцитов:

---

## 1. HARD POSITIVES (Трудные позитивы) - ПРИОРИТЕТ ВЫСОКИЙ

**Что это:** Мастоциты, которые сложно обнаружить, но они точно являются мастоцитами (подтверждены ИГХ).

### Типы Hard Positives:

1. **Веретеновидные мастоциты** (`spindle_shaped`)
   - Вытянутая форма, мимикрируют под фибробласты
   - Требуется: Г/Э + ИГХ обязательно

2. **Мастоциты с наложением** (`overlapping`)
   - Несколько клеток сливаются
   - Требуется: Г/Э + ИГХ обязательно

3. **Дегранулированные мастоциты** (`degranulated`)
   - Выбросили гранулы, цитоплазма бледная
   - Требуется: Г/Э + ИГХ обязательно

4. **Мастоциты на краях** (`edge_cases`)
   - Частично срезанные краем кадра
   - Требуется: Г/Э + ИГХ желательно

5. **Мастоциты в кластерах** (`clusters`)
   - Несколько мастоцитов рядом, цитоплазма сливается
   - Требуется: Г/Э + ИГХ обязательно

**Количество:** Минимум 50-100 примеров каждого типа (всего 250-500)

---

## 2. HARD NEGATIVES (Трудные негативы) - КРИТИЧЕСКИ ВАЖНО

**Что это:** Клетки, которые похожи на мастоциты, но ими НЕ являются. Помечены как `is_mast_cell: false`.

### Типы Hard Negatives:

1. **Плазмоциты с центральным ядром** (`plasmocytes_central`)
   - Редкий случай, когда ядро по центру
   - Отличия: хроматин "колесо телеги", светлый "дворик"

2. **Активные фибробласты** (`fibroblasts`)
   - Похожи на веретеновидные мастоциты
   - Отличия: ядро вытянутое, цитоплазма тянется нитями

3. **Гистиоциты** (`histiocytes`)
   - Похожи на неявные мастоциты
   - Отличия: ядро везикулярное, нет эффекта "гало"

4. **Эозинофилы** (`eosinophils`)
   - Ярко-красная цитоплазма
   - Отличия: ядро би-лобарное, цитоплазма рефрактерная

5. **Активированные лимфоциты** (`lymphocytes`)
   - Могут иметь видимую цитоплазму
   - Отличия: ядро идеально круглое, хроматин очень плотный

**Количество:** Минимум 50-100 примеров каждого типа (всего 250-500)

**Критически важно:** Все должны быть помечены `is_mast_cell: false` и иметь `distinguishing_features` (отличия от мастоцита).

---

## 3. CONFUSION SET (Смешанный датасет)

**Что это:** Смешанный датасет из похожих клеток для обучения тонкой дифференциации.

**Состав:**
- Плазмоциты
- Эозинофилы
- Мастоциты (явные и неявные)
- Все вперемешку

**Количество:** 100-200 примеров

**Цель:** Научить модель тонкой дифференциации между похожими клетками.

---

## 4. ПАРНЫЕ ДАННЫЕ (Г/Э + ИГХ) - ОБЯЗАТЕЛЬНО

**Что это:** Тот же участок ткани, окрашенный Г/Э и затем ИГХ (CD117/Tryptase).

**Требования:**
- Минимум 50-100 парных патчей для сложных случаев
- Идеальная регистрация изображений (тот же срез)
- Позволяет найти скрытые паттерны на Г/Э

**Формат:**
- Г/Э: `example_001_he.png`
- ИГХ: `example_001_ihc.png`
- Метаданные: `paired: true`, `ihc_positive: true/false`

---

## 5. AMBIGUOUS (Неоднозначные случаи)

**Что это:** Клетки, похожие на мастоциты, но нет уверенности.

**Метка:** `confidence: "ambiguous"` или `cell_type: "ambiguous"`

**Цель:** Отделить "чистые" данные от "шумных", не заставлять выбирать только Да/Нет.

---

## Структура директорий

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
│   │   ├── 001_metadata.json
│   │   └── 001_distinguishing.json
│   ├── fibroblasts/
│   ├── histiocytes/
│   ├── eosinophils/
│   └── lymphocytes/
│
├── confusion_set/
│   ├── 001_he.png
│   ├── 001_ihc.png  # только если мастоцит
│   └── 001_metadata.json
│
└── ambiguous/
    ├── 001_he.png
    └── 001_metadata.json
```

---

## Формат метаданных

### Hard Positive (`001_metadata.json`):

```json
{
    "morphological_features": {
        "nucleus": "central, ovoid, hyperchromatic",
        "cytoplasm": "pale pink, blurred boundaries",
        "shape": "spindle-shaped, elongated",
        "location": "stroma, near vessels"
    },
    "coordinates": {"x": 195, "y": 190},
    "tissue_location": "colon_mucosa",
    "inflammation_level": "moderate",
    "gemini_insights": [
        "Rule of 'Dirty Halo'",
        "Nuclear criterion: ovoid nucleus"
    ]
}
```

### Hard Negative (`001_metadata.json` + `001_distinguishing.json`):

**metadata.json:**
```json
{
    "morphological_features": {
        "nucleus": "central, round, 'cartwheel' chromatin",
        "cytoplasm": "homogeneous, light halo around nucleus",
        "shape": "round"
    },
    "coordinates": {"x": 300, "y": 250}
}
```

**distinguishing.json:**
```json
{
    "vs_mast_cell": {
        "chromatin": "cartwheel pattern, not monochromatic",
        "key_difference": "eccentric nucleus usually, but here central",
        "cytoplasm": "homogeneous, not granular"
    }
}
```

### Confusion Set (`001_metadata.json`):

```json
{
    "cell_type": "plasmocyte",  // или "mast_cell", "eosinophil"
    "morphological_features": {
        "nucleus": "...",
        "cytoplasm": "..."
    },
    "distinguishing_features": {
        "similar_to_mast_cell": "central nucleus, visible cytoplasm",
        "different_from_mast_cell": "cartwheel chromatin pattern"
    }
}
```

---

## Как добавлять в KB

### Вариант 1: Через скрипт (рекомендуется)

```bash
# Добавить все типы датасетов
python add_datasets_to_kb.py \
    --data_dir ./mast_cells_datasets \
    --kb_path ./mast_cells_kb \
    --dataset_type all \
    --use_clip

# Добавить только Hard Positives
python add_datasets_to_kb.py \
    --data_dir ./mast_cells_datasets \
    --dataset_type hard_positives

# Добавить только Hard Negatives
python add_datasets_to_kb.py \
    --data_dir ./mast_cells_datasets \
    --dataset_type hard_negatives
```

### Вариант 2: Программно

```python
from train_knowledge_base import MastCellsKnowledgeBase

kb = MastCellsKnowledgeBase(use_clip=True)

# Hard Positive
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
    ihc_marker="CD117"
)

# Hard Negative
kb.add_hard_negative(
    example_id="plasmocyte_001",
    image_path="hard_negatives/plasmocytes_central/001_he.png",
    cell_type="plasmocyte",
    hard_negative_type="plasmocytes_central",
    morphological_features={
        "nucleus": "central, round, 'cartwheel' chromatin",
        "cytoplasm": "homogeneous, light halo"
    },
    distinguishing_features={
        "vs_mast_cell": "chromatin 'cartwheel' pattern, not monochromatic",
        "key_difference": "eccentric nucleus usually, but here central"
    }
)

# Confusion Set
kb.add_confusion_set_example(
    example_id="confusion_001",
    image_path="confusion_set/001_he.png",
    cell_type="plasmocyte",
    morphological_features={...},
    distinguishing_features={...}
)
```

---

## Поиск в KB по типам датасетов

```python
# Поиск Hard Positives определенного типа
results = kb.search_similar(
    query_text="spindle-shaped mast cell",
    filter_dataset_category="hard_positive",
    filter_hard_positive_type="spindle_shaped"
)

# Поиск Hard Negatives для обучения различиям
results = kb.search_similar(
    query_text="cell with central nucleus",
    filter_dataset_category="hard_negative",
    filter_hard_negative_type="plasmocytes_central"
)

# Поиск только мастоцитов
results = kb.search_similar(
    query_text="implicit mast cell",
    filter_is_mast_cell=True
)

# Поиск только НЕ мастоцитов (Hard Negatives)
results = kb.search_similar(
    query_text="cell similar to mast cell",
    filter_is_mast_cell=False,
    filter_dataset_category="hard_negative"
)
```

---

## Требования к качеству изображений

1. **Формат:** PNG (lossless) или TIFF
2. **Разрешение:** Минимум 40x, оптимально 100x
3. **Размер:** Минимум 512x512 пикселей при 40x
4. **Качество:** Без артефактов сжатия, видна текстура цитоплазмы

---

## Минимальные требования

- **Hard Positives:** 50-100 примеров каждого типа (250-500 всего)
- **Hard Negatives:** 50-100 примеров каждого типа (250-500 всего)
- **Confusion Set:** 100-200 примеров
- **Парные данные:** 50-100 пар (Г/Э + ИГХ)

**Оптимально:** Увеличить в 2-5 раз для лучших результатов.

---

## Следующие шаги

1. ⏳ Создать структуру директорий для данных
2. ⏳ Начать сбор Hard Positives (приоритет: веретеновидные, дегранулированные)
3. ⏳ Начать сбор Hard Negatives (приоритет: плазмоциты с центральным ядром, фибробласты)
4. ⏳ Организовать парные данные (Г/Э + ИГХ)
5. ⏳ Создать Confusion Set
6. ⏳ Добавить все в KB используя `add_datasets_to_kb.py`

