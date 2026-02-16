# Архитектура Knowledge Base для обнаружения мастоцитов

## Обзор системы

Knowledge Base (KB) использует подход RAG (Retrieval-Augmented Generation) для улучшения результатов Gemini при обнаружении мастоцитов. 

**Ключевые особенности согласно рекомендациям Gemini:**
- ✅ **Мультимодальный подход**: Текстовые эмбеддинги + CLIP для изображений
- ✅ **Парные данные**: Г/Э + ИГХ изображения для обучения на скрытых паттернах
- ✅ **Визуальный поиск**: Поиск похожих изображений через CLIP эмбеддинги
- ✅ **Hard Positives**: Трудные позитивы (веретеновидные, дегранулированные и т.д.)
- ✅ **Hard Negatives**: Трудные негативы (плазмоциты, фибробласты и т.д.)
- ✅ **Confusion Set**: Смешанный датасет для тонкой дифференциации
- ✅ **Ambiguous**: Неоднозначные случаи

Система хранит примеры успешных анализов и использует их для предоставления контекста при новых запросах.

**Подробнее о типах датасетов:** см. `docs/GEMINI_DATASET_REQUIREMENTS.md` и `docs/DATASETS_FOR_KB.md`

## Архитектура системы

```
┌─────────────────────────────────────────────────────────────────┐
│                    ИСТОЧНИКИ ДАННЫХ                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Результаты анализа Gemini                                  │
│     - mast_cells_coordinates_analysis_result.json              │
│     - mast_cells_analysis_result.txt                           │
│                                                                 │
│  2. Изображения                                                 │
│     - MAST_GEMINI/*_yes.png (с разметкой)                      │
│     - MAST_GEMINI/*_no.png (без разметки)                      │
│     - MAST_GEMINI/*_игх.png (ИГХ эталон)                       │
│                                                                 │
│  3. Ручные аннотации                                           │
│     - Координаты мастоцитов                                    │
│     - Типы (explicit/implicit)                                 │
│     - Морфологические признаки                                 │
│                                                                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              ПОДГОТОВКА И СТРУКТУРИРОВАНИЕ ДАННЫХ               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Каждый пример содержит:                                       │
│                                                                 │
│  {                                                              │
│    "example_id": "explicit_001",                               │
│    "image_path": "MAST_GEMINI/01_yes.png",                     │
│    "ihc_image_path": "MAST_GEMINI/1_игх.png",                  │
│    "morphological_features": {                                 │
│      "nucleus": "central, round, hyperchromatic",              │
│      "cytoplasm": "eosinophilic, granular, abundant",         │
│      "shape": "round, 'fried egg' pattern",                    │
│      "location": "stroma, between crypts"                      │
│    },                                                           │
│    "coordinates": {"x": 195, "y": 190},                       │
│    "cell_type": "explicit",                                     │
│    "confidence": "high",                                       │
│    "difficulty": "easy",                                       │
│    "gemini_insights": [                                        │
│      "Rule of 'Dirty Halo'",                                   │
│      "Nuclear criterion: ovoid nucleus"                        │
│    ]                                                            │
│  }                                                              │
│                                                                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    СОЗДАНИЕ ЭМБЕДДИНГОВ                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ТЕКСТОВЫЕ ЭМБЕДДИНГИ (Sentence Transformers)                  │
│  ────────────────────────────────────────────────────────────   │
│                                                                 │
│  Модель: all-MiniLM-L6-v2                                      │
│  Размерность: 384                                              │
│                                                                 │
│  Текст для эмбеддинга формируется из:                          │
│  "Cell type: explicit mast cell |                              │
│   Nucleus: central, round, hyperchromatic |                    │
│   Cytoplasm: eosinophilic, granular, abundant |               │
│   Shape: round, 'fried egg' pattern |                         │
│   Location: stroma, between crypts |                          │
│   Key insights: Rule of 'Dirty Halo'; ... |                   │
│   Confidence: high |                                            │
│   Difficulty: easy"                                             │
│                                                                 │
│  → [0.123, -0.456, 0.789, ..., 0.234] (384 числа)            │
│                                                                 │
│                                                                 │
│  ИЗОБРАЖЕНИЯ (опционально, для будущего расширения)            │
│  ────────────────────────────────────────────────────────────   │
│                                                                 │
│  Модель: CLIP (ViT-B/32) или ResNet                            │
│  Размерность: 512                                              │
│                                                                 │
│  Изображение → [0.234, -0.567, 0.890, ..., 0.123] (512 чисел) │
│                                                                 │
│  ПРИМЕЧАНИЕ: В текущей реализации используем только            │
│              текстовые эмбеддинги. Изображения хранятся       │
│              как пути в метаданных.                           │
│                                                                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              ВЕКТОРНОЕ ХРАНИЛИЩЕ (ChromaDB)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Коллекция: "mast_cells"                                        │
│                                                                 │
│  Структура записи:                                              │
│  ────────────────────────────────────────────────────────────   │
│                                                                 │
│  ID: "explicit_001"                                            │
│                                                                 │
│  Embedding: [0.123, -0.456, ..., 0.234] (384 числа)          │
│                                                                 │
│  Document (текст):                                              │
│  "Cell type: explicit mast cell | Nucleus: ... | ..."          │
│                                                                 │
│  Metadata:                                                      │
│  {                                                              │
│    "example_id": "explicit_001",                               │
│    "image_path": "MAST_GEMINI/01_yes.png",                     │
│    "ihc_image_path": "MAST_GEMINI/1_игх.png",                  │
│    "difficulty": "easy",                                       │
│    "cell_type": "explicit",                                    │
│    "confidence": "high",                                       │
│    "coordinates_x": "195",                                     │
│    "coordinates_y": "190",                                     │
│    "text": "полный текст описания"                            │
│  }                                                              │
│                                                                 │
│  Физическое хранилище:                                          │
│  ./mast_cells_kb/                                               │
│    ├── chroma.sqlite3 (SQLite база)                            │
│    ├── index/ (индексы для быстрого поиска)                    │
│    └── ...                                                      │
│                                                                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ПОИСК ПОХОЖИХ ПРИМЕРОВ                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Запрос: "Find implicit mast cells with blurred boundaries"    │
│                                                                 │
│  1. Создаем эмбеддинг запроса:                                  │
│     query_embedding = embedder.encode(query)                  │
│     → [0.234, -0.123, ..., 0.456]                             │
│                                                                 │
│  2. Поиск в векторном пространстве:                            │
│     - Cosine similarity между query и всеми примерами          │
│     - Возвращаем топ-N наиболее похожих                        │
│                                                                 │
│  3. Фильтрация (опционально):                                   │
│     - По difficulty: "hard"                                    │
│     - По cell_type: "implicit"                                │
│                                                                 │
│  4. Результаты:                                                 │
│     [                                                            │
│       {                                                          │
│         "id": "implicit_001",                                  │
│         "text": "...",                                          │
│         "metadata": {...},                                      │
│         "distance": 0.15  // чем меньше, тем похожее           │
│       },                                                         │
│       ...                                                        │
│     ]                                                            │
│                                                                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              ФОРМИРОВАНИЕ КОНТЕКСТА ДЛЯ GEMINI                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Из найденных примеров формируем расширенный промпт:           │
│                                                                 │
│  "Based on previous successful analyses...                     │
│                                                                 │
│   === KEY MORPHOLOGICAL PATTERNS ===                           │
│   ...                                                           │
│                                                                 │
│   === SUCCESSFUL DETECTION EXAMPLES ===                          │
│   Example 1 (similarity: 85%):                                 │
│     Cell type: implicit mast cell |                            │
│     Nucleus: central, ovoid | ...                              │
│     Type: implicit                                             │
│     Confidence: medium                                         │
│                                                                 │
│   Example 2 (similarity: 78%):                                │
│     ...                                                         │
│                                                                 │
│   === ANALYSIS TASK ===                                         │
│   Find all mast cells in this image..."                        │
│                                                                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ЗАПРОС К GEMINI API                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Gemini получает:                                               │
│  1. Расширенный промпт с контекстом                            │
│  2. Изображение для анализа                                     │
│                                                                 │
│  Gemini использует контекст для:                                │
│  - Понимания паттернов                                          │
│  - Применения эвристик                                          │
│  - Более точного обнаружения                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Структура данных

### 1. Формат входных данных

Каждый пример в KB должен содержать:

```python
{
    # Идентификатор
    "example_id": "explicit_001",  # уникальный ID
    
    # Изображения
    "image_path": "MAST_GEMINI/01_yes.png",  # Г/Э с разметкой
    "ihc_image_path": "MAST_GEMINI/1_игх.png",  # парное ИГХ (опционально)
    
    # Морфологические признаки
    "morphological_features": {
        "nucleus": "central, round, hyperchromatic, dark purple",
        "cytoplasm": "eosinophilic, granular, abundant, pink halo",
        "shape": "round, 'fried egg' pattern",
        "location": "stroma, between crypts, isolated"
    },
    
    # Координаты (если известны)
    "coordinates": {
        "x": 195,
        "y": 190
    },
    
    # Классификация
    "cell_type": "explicit",  # или "implicit"
    "confidence": "high",  # "high", "medium", "low"
    "difficulty": "easy",  # "easy", "medium", "hard"
    
    # Инсайты от Gemini
    "gemini_insights": [
        "Rule of 'Dirty Halo': creates space with muddy pink substance",
        "Nuclear criterion: ovoid, 'plump' nucleus",
        "Law of neighborhood: rarely solitary"
    ]
}
```

### 2. Формат хранения в ChromaDB

```python
# Структура записи в ChromaDB
{
    "id": "explicit_001",
    
    # Векторное представление (embedding)
    "embedding": [0.123, -0.456, 0.789, ..., 0.234],  # 384 числа
    
    # Текстовое описание (для поиска и отображения)
    "document": "Cell type: explicit mast cell | Nucleus: central, round...",
    
    # Метаданные (для фильтрации и доступа к файлам)
    "metadata": {
        "example_id": "explicit_001",
        "image_path": "MAST_GEMINI/01_yes.png",
        "ihc_image_path": "MAST_GEMINI/1_игх.png",
        "difficulty": "easy",
        "cell_type": "explicit",
        "confidence": "high",
        "coordinates_x": "195",
        "coordinates_y": "190",
        "text": "полный текст описания"
    }
}
```

## Блоки системы

### Блок 1: Подготовка данных

**Файл:** `train_knowledge_base.py`

**Функции:**
- `populate_from_gemini_analysis()` - извлечение данных из результатов Gemini
- `add_explicit_examples()` - добавление примеров явных мастоцитов
- `add_implicit_examples()` - добавление примеров неявных мастоцитов

**Входные данные:**
- JSON файлы с результатами анализа Gemini
- Изображения из `MAST_GEMINI/`
- Ручные аннотации (опционально)

**Выходные данные:**
- Структурированные записи для KB

### Блок 2: Создание эмбеддингов

**Модель:** `SentenceTransformer('all-MiniLM-L6-v2')`

**Параметры:**
- Размерность: 384
- Тип: текстовые эмбеддинги
- Метрика: cosine similarity

**Процесс:**
1. Формирование текстового описания из всех признаков
2. Кодирование текста в вектор
3. Сохранение вектора в ChromaDB

**Пример текста для эмбеддинга:**
```
Cell type: explicit mast cell | 
Nucleus: central, round, hyperchromatic, dark purple | 
Cytoplasm: eosinophilic, granular, abundant, pink halo | 
Shape: round, 'fried egg' pattern | 
Location: stroma, between crypts, isolated | 
Key insights: Rule of 'Dirty Halo'; Nuclear criterion: ovoid nucleus | 
Confidence: high | 
Difficulty: easy
```

### Блок 3: Векторное хранилище

**Технология:** ChromaDB (PersistentClient)

**Структура:**
```
./mast_cells_kb/
├── chroma.sqlite3          # SQLite база данных
├── index/                  # Индексы для быстрого поиска
│   ├── index_metadata.pkl
│   └── ...
└── ...
```

**Коллекция:** `mast_cells`

**Операции:**
- `add()` - добавление примера
- `query()` - поиск похожих примеров
- `get()` - получение по ID
- `delete()` - удаление примера

### Блок 4: Поиск и извлечение

**Функция:** `search_similar()`

**Алгоритм:**
1. Кодирование запроса в эмбеддинг
2. Поиск в векторном пространстве (cosine similarity)
3. Фильтрация по метаданным (опционально)
4. Возврат топ-N результатов

**Параметры поиска:**
- `query_text` - текст запроса
- `n_results` - количество результатов (по умолчанию 5)
- `filter_difficulty` - фильтр по сложности
- `filter_cell_type` - фильтр по типу клетки

### Блок 5: Формирование контекста

**Файл:** `analyze_with_kb.py`

**Процесс:**
1. Поиск похожих примеров в KB
2. Формирование структурированного контекста
3. Добавление паттернов и эвристик
4. Создание расширенного промпта для Gemini

**Структура контекста:**
```
=== KEY MORPHOLOGICAL PATTERNS ===
EXPLICIT MAST CELLS: ...
IMPLICIT MAST CELLS: ...
=== DISTINGUISHING FEATURES ===
vs LYMPHOCYTES: ...
vs PLASMOCYTES: ...
=== SUCCESSFUL DETECTION EXAMPLES ===
Example 1: ...
Example 2: ...
```

### Блок 6: Интеграция с Gemini

**Файл:** `analyze_with_kb.py` → `EnhancedMastCellsAnalyzer`

**Процесс:**
1. Получение контекста из KB
2. Формирование расширенного промпта
3. Отправка запроса к Gemini API
4. Получение улучшенного результата

## Данные для подготовки

### Минимальный набор данных

**Для старта нужно:**
1. ✅ Результаты анализа Gemini (уже есть)
   - `mast_cells_coordinates_analysis_result.json`
   - `mast_cells_analysis_result.txt`

2. ✅ Изображения (уже есть)
   - `MAST_GEMINI/*_yes.png` (с разметкой)
   - `MAST_GEMINI/*_no.png` (без разметки)
   - `MAST_GEMINI/*_игх.png` (ИГХ эталоны)

3. ⏳ Дополнительные примеры (для расширения)
   - 50-100 парных патчей Г/Э + ИГХ
   - Разметка координат мастоцитов
   - Классификация на explicit/implicit

### Расширенный набор данных

**Для улучшения качества:**

1. **Больше примеров неявных мастоцитов**
   - Дегранулированные мастоциты
   - Веретеновидные формы
   - Мастоциты с наложением на другие клетки

2. **Hard Negatives**
   - Фибробласты, похожие на мастоциты
   - Плазмоциты с центральным ядром
   - Гистиоциты

3. **Парные данные**
   - Тот же участок ткани: Г/Э + ИГХ
   - Идеальная регистрация изображений

## Формат хранения изображений

### Текущий подход

**Изображения НЕ хранятся в KB:**
- Хранятся только пути к файлам в метаданных
- Изображения остаются в файловой системе
- При поиске возвращаются пути, а не сами изображения

**Преимущества:**
- Меньший размер базы данных
- Быстрее операции поиска
- Легче обновлять изображения

**Недостатки:**
- Нужен доступ к файловой системе
- Нельзя искать по визуальному сходству изображений

### Будущее расширение (опционально)

**Мультимодальные эмбеддинги:**

```python
# Использование CLIP для эмбеддингов изображений
from sentence_transformers import SentenceTransformer

# Текстовые эмбеддинги
text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
text_embedding = text_embedder.encode(text_description)

# Эмбеддинги изображений (CLIP)
image_embedder = SentenceTransformer('clip-ViT-B-32')
image_embedding = image_embedder.encode(image)

# Комбинированный поиск
# Можно искать по тексту ИЛИ по изображению
```

## Процесс работы системы

### 1. Инициализация KB

```python
kb = MastCellsKnowledgeBase(db_path="./mast_cells_kb")
```

**Что происходит:**
- Создается/загружается ChromaDB коллекция
- Загружается модель для эмбеддингов
- Готовность к добавлению примеров

### 2. Пополнение KB

```python
# Из результатов Gemini
populate_from_gemini_analysis()

# Ручное добавление
kb.add_example(
    example_id="explicit_001",
    image_path="MAST_GEMINI/01_yes.png",
    morphological_features={...},
    gemini_insights=[...],
    cell_type="explicit",
    confidence="high"
)
```

**Что происходит:**
- Формируется текстовое описание
- Создается эмбеддинг (384 числа)
- Сохраняется в ChromaDB с метаданными

### 3. Поиск похожих примеров

```python
results = kb.search_similar(
    query_text="Find implicit mast cells with blurred boundaries",
    n_results=5,
    filter_cell_type="implicit"
)
```

**Что происходит:**
- Запрос кодируется в эмбеддинг
- Поиск в векторном пространстве (cosine similarity)
- Фильтрация по метаданным
- Возврат топ-N результатов

### 4. Анализ с контекстом

```python
analyzer = EnhancedMastCellsAnalyzer()
result = await analyzer.analyze_with_context(
    image_path=Path("MAST_GEMINI/01_no.png"),
    query="Find all mast cells"
)
```

**Что происходит:**
- Поиск похожих примеров в KB
- Формирование контекстного промпта
- Отправка запроса к Gemini с контекстом
- Получение улучшенного результата

## Примеры использования

### Создание KB из результатов Gemini

```bash
python train_knowledge_base.py --action populate
```

**Результат:**
- Создается `./mast_cells_kb/`
- Добавляются примеры из анализа
- Готовность к использованию

### Поиск в KB

```bash
python train_knowledge_base.py \
    --action search \
    --query "implicit mast cell with blurred boundaries" \
    --n_results 5
```

### Анализ с использованием KB

```bash
python analyze_with_kb.py \
    --image MAST_GEMINI/01_no.png \
    --n_examples 5 \
    --filter_cell_type implicit
```

## Метрики и оценка

### Качество поиска

**Метрики:**
- Cosine similarity distance (чем меньше, тем лучше)
- Количество релевантных результатов в топ-N
- Покрытие различных типов мастоцитов

### Влияние на результаты Gemini

**Ожидаемые улучшения:**
- Более точное обнаружение неявных мастоцитов
- Меньше ложных срабатываний
- Более структурированные ответы

**Оценка:**
- Сравнение результатов с/без KB
- Метрики точности (precision, recall)
- Анализ ошибок

## Расширения и улучшения

### Краткосрочные

1. ✅ Добавление больше примеров из анализа Gemini
2. ✅ Фильтрация по различным критериям
3. ✅ Улучшение текстовых описаний

### Среднесрочные

1. ⏳ Добавление эмбеддингов изображений (CLIP)
2. ⏳ Мультимодальный поиск (текст + изображение)
3. ⏳ Автоматическое извлечение признаков из изображений

### Долгосрочные

1. ⏳ Интеграция с обученной моделью (Qwen-VL)
2. ⏳ Автоматическое обновление KB из новых анализов
3. ⏳ Обратная связь и улучшение на основе результатов

