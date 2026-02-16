# Краткое руководство: Knowledge Base для мастоцитов

## Что нужно подготовить

### 1. Данные (уже есть ✅)

```
MAST_GEMINI/
├── 01_no.png          # Изображение без разметки
├── 01_yes.png         # То же изображение с разметкой
├── 1_игх.png          # Парное ИГХ изображение (эталон)
├── 02_no.png
├── 02_yes.png
└── ...

mast_cells_coordinates_analysis_result.json  # Результаты анализа Gemini
```

### 2. Структура данных для KB

Каждый пример содержит:

```python
{
    "example_id": "explicit_001",
    "image_path": "MAST_GEMINI/01_yes.png",      # Путь к изображению
    "ihc_image_path": "MAST_GEMINI/1_игх.png",   # Парное ИГХ (опционально)
    
    # Морфологические признаки
    "morphological_features": {
        "nucleus": "central, round, hyperchromatic",
        "cytoplasm": "eosinophilic, granular, abundant",
        "shape": "round, 'fried egg' pattern",
        "location": "stroma, between crypts"
    },
    
    # Координаты (если известны)
    "coordinates": {"x": 195, "y": 190},
    
    # Классификация
    "cell_type": "explicit",      # или "implicit"
    "confidence": "high",         # "high", "medium", "low"
    "difficulty": "easy",         # "easy", "medium", "hard"
    
    # Инсайты от Gemini
    "gemini_insights": [
        "Rule of 'Dirty Halo'",
        "Nuclear criterion: ovoid nucleus"
    ]
}
```

## Блоки системы

### Блок 1: Текстовое описание → Эмбеддинг

**Вход:** Структурированные данные (см. выше)

**Процесс:**
```
Морфологические признаки + Инсайты + Метаданные
    ↓
Текстовое описание:
"Cell type: explicit mast cell | Nucleus: central, round... | ..."
    ↓
SentenceTransformer('all-MiniLM-L6-v2')
    ↓
Вектор [0.123, -0.456, ..., 0.234] (384 числа)
```

**Модель:** `all-MiniLM-L6-v2` (384 размерности)

### Блок 2: Векторное хранилище (ChromaDB)

**Структура записи:**

```
ID: "explicit_001"
├── Embedding: [384 числа]          # Векторное представление
├── Document: "текстовое описание"  # Для чтения человеком
└── Metadata: {
        "image_path": "...",         # Путь к изображению
        "cell_type": "explicit",     # Для фильтрации
        "difficulty": "easy",        # Для фильтрации
        "coordinates_x": "195",       # Координаты
        "coordinates_y": "190"
    }
```

**Физическое хранилище:**
```
./mast_cells_kb/
├── chroma.sqlite3    # База данных
└── index/            # Индексы для быстрого поиска
```

### Блок 3: Поиск похожих примеров

**Процесс:**

```
Запрос: "Find implicit mast cells"
    ↓
Эмбеддинг запроса: [0.234, -0.123, ...]
    ↓
Поиск в векторном пространстве (cosine similarity)
    ↓
Топ-5 наиболее похожих примеров:
[
    {id: "implicit_001", distance: 0.15, ...},
    {id: "implicit_002", distance: 0.23, ...},
    ...
]
```

**Метрика:** Cosine similarity (чем меньше distance, тем похожее)

### Блок 4: Формирование контекста для Gemini

**Процесс:**

```
Найденные примеры из KB
    ↓
Структурированный контекст:
"=== KEY PATTERNS ===
 EXPLICIT MAST CELLS: ...
 IMPLICIT MAST CELLS: ...
 === EXAMPLES ===
 Example 1: ...
 Example 2: ..."
    ↓
Расширенный промпт для Gemini
```

### Блок 5: Изображения

**Текущий подход:**
- Изображения НЕ хранятся в KB
- Хранятся только пути в метаданных
- При поиске возвращаются пути, не сами изображения

**Будущее расширение (опционально):**
- Эмбеддинги изображений через CLIP
- Мультимодальный поиск (текст + изображение)

## Что создается

### 1. Векторные эмбеддинги (текст)

**Формат:** Массив из 384 чисел (float32)

**Пример:**
```python
[0.123, -0.456, 0.789, ..., 0.234]  # 384 числа
```

**Где хранится:** В ChromaDB как `embedding`

### 2. Текстовые описания

**Формат:** Строка с разделителями `|`

**Пример:**
```
"Cell type: explicit mast cell | Nucleus: central, round, hyperchromatic | Cytoplasm: eosinophilic, granular | Shape: round, 'fried egg' pattern | Location: stroma, between crypts | Key insights: Rule of 'Dirty Halo' | Confidence: high | Difficulty: easy"
```

**Где хранится:** В ChromaDB как `document`

### 3. Метаданные

**Формат:** JSON объект

**Пример:**
```json
{
    "example_id": "explicit_001",
    "image_path": "MAST_GEMINI/01_yes.png",
    "ihc_image_path": "MAST_GEMINI/1_игх.png",
    "difficulty": "easy",
    "cell_type": "explicit",
    "confidence": "high",
    "coordinates_x": "195",
    "coordinates_y": "190"
}
```

**Где хранится:** В ChromaDB как `metadata`

### 4. Изображения

**Формат:** PNG/JPEG файлы

**Где хранятся:** В файловой системе (`MAST_GEMINI/`)

**В KB:** Только пути в метаданных

## Процесс работы

### Шаг 1: Создание KB

```bash
python train_knowledge_base.py --action populate
```

**Что происходит:**
1. Читаются результаты анализа Gemini
2. Для каждого примера создается текстовое описание
3. Создается эмбеддинг (384 числа)
4. Сохраняется в ChromaDB с метаданными

### Шаг 2: Поиск

```python
results = kb.search_similar(
    query_text="implicit mast cell",
    n_results=5
)
```

**Что происходит:**
1. Запрос кодируется в эмбеддинг
2. Поиск в векторном пространстве
3. Возврат топ-5 результатов

### Шаг 3: Анализ с KB

```python
analyzer = EnhancedMastCellsAnalyzer()
result = await analyzer.analyze_with_context(image_path)
```

**Что происходит:**
1. Поиск похожих примеров в KB
2. Формирование контекста
3. Отправка к Gemini с контекстом
4. Получение улучшенного результата

## Схема данных

```
┌─────────────────────────────────────────┐
│         ИСХОДНЫЕ ДАННЫЕ                  │
├─────────────────────────────────────────┤
│  • Изображения (PNG/JPEG)               │
│  • Результаты анализа Gemini (JSON)     │
│  • Морфологические признаки             │
│  • Координаты мастоцитов                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      ТЕКСТОВОЕ ОПИСАНИЕ                 │
│  "Cell type: explicit | Nucleus: ..."  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      ЭМБЕДДИНГ (384 числа)              │
│  [0.123, -0.456, ..., 0.234]           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      CHROMADB ХРАНИЛИЩЕ                 │
│  • Embedding: [384 числа]               │
│  • Document: "текст"                    │
│  • Metadata: {пути, координаты, ...}    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      ПОИСК ПОХОЖИХ                      │
│  Query → Embedding → Cosine Similarity  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      КОНТЕКСТ ДЛЯ GEMINI                │
│  Примеры + Паттерны + Эвристики         │
└─────────────────────────────────────────┘
```

## Ключевые моменты

1. **Эмбеддинги только текстовые** (изображения хранятся как пути)
2. **Размерность: 384** (модель all-MiniLM-L6-v2)
3. **Метрика поиска: Cosine similarity**
4. **Изображения остаются в файловой системе**
5. **KB хранит только векторы и метаданные**

## Быстрый старт

```bash
# 1. Создать KB
python train_knowledge_base.py --action populate

# 2. Проверить содержимое
python train_knowledge_base.py --action list

# 3. Поиск
python train_knowledge_base.py \
    --action search \
    --query "implicit mast cell" \
    --n_results 5

# 4. Анализ с KB
python analyze_with_kb.py --image MAST_GEMINI/01_no.png
```

