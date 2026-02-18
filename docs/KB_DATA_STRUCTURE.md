# Структура данных в Knowledge Base

## Обзор

KB **не хранит JSON файлы** — это векторная база данных (ChromaDB), где каждый пример представлен:
- **Текстовым эмбеддингом** (384 числа) — векторное представление описания
- **CLIP эмбеддингом изображения** (512 чисел) — векторное представление кропа
- **Метаданными** (словарь) — пути к файлам, классы, координаты и т.д.
- **Текстовым документом** — полное описание для формирования промптов

Физически KB — это директория `mast_cells_kb/` с SQLite базой (`chroma.sqlite3`) и индексами.

---

## Что попадает в KB из конвейера

Когда конвейер (`pipeline_annotations_to_kb.py`) обрабатывает ваш JSON и `_no` изображения, для каждой аннотации создаётся:

### 1. Кроп (файл изображения)

**Где:** сохраняется в `--crops_dir` как `{ann_id}.png` (например `1_yes_g1.png`).

**Что:** вырезанный патч из `_no` по bbox с отступом (padding ~20px или 15% от размера).

**В KB:** хранится **путь к файлу** (`image_path`), не сам файл.

### 2. Текстовое описание (формируется автоматически)

Конвейер создаёт текст из:
- `cell_type` (explicit/implicit)
- `morphological_features` (nucleus, cytoplasm, shape, location)
- `confidence` (high/medium/low)
- `difficulty` (easy/medium/hard)

**Пример текста:**
```
Cell type: explicit mast cell | Nucleus: central, round/ovoid | 
Cytoplasm: granular, eosinophilic | Shape: round to ovoid | 
Location: stroma | Confidence: high | Difficulty: easy
```

Этот текст кодируется в **текстовый эмбеддинг** (384 числа через `all-MiniLM-L6-v2`).

### 3. CLIP эмбеддинг изображения (опционально)

Если `--no_clip` не указан и доступен CLIP:
- Кроп загружается как PIL Image
- Кодируется через `clip-ViT-B-32` → **512 чисел**
- Сохраняется в отдельной коллекции `mast_cells_images`

### 4. Метаданные (словарь)

```python
{
    "example_id": "1_yes_g1",           # ID из JSON аннотации
    "image_path": "crops/1_yes_g1.png", # Путь к кропу
    "ihc_image_path": "",                # Путь к парному ИГХ (если есть)
    "cell_type": "explicit",             # explicit или implicit
    "difficulty": "easy",                # easy/medium/hard
    "confidence": "high",                 # high/medium/low
    "coordinates_x": "400",              # Центр bbox (x)
    "coordinates_y": "300",              # Центр bbox (y)
    "has_paired_ihc": "no",              # yes/no
    "text": "Cell type: explicit mast cell | ..."  # Полное описание
}
```

---

## Структура в ChromaDB

KB содержит **две коллекции**:

### Коллекция 1: `mast_cells_text` (текстовые эмбеддинги)

**ID записи:** `{example_id}_text` (например `1_yes_g1_text`)

**Содержимое:**
- `embeddings`: список из 384 чисел (текстовый эмбеддинг)
- `documents`: строка с полным текстовым описанием
- `metadatas`: словарь метаданных (см. выше)

**Используется для:** поиска по текстовым запросам ("Find implicit mast cells with blurred boundaries").

### Коллекция 2: `mast_cells_images` (CLIP эмбеддинги)

**ID записи:** `{example_id}_he` (для Г/Э кропа), `{example_id}_ihc` (для парного ИГХ, если есть)

**Содержимое:**
- `embeddings`: список из 512 чисел (CLIP эмбеддинг изображения)
- `metadatas`: словарь с `image_type` ("H&E" или "IHC"), `image_path`, `cell_type`, `difficulty`, `confidence`

**Используется для:** визуального поиска похожих изображений (`search_similar_images`).

---

## Пример: что хранится для одной аннотации

**Входной JSON:**
```json
{
  "id": "1_yes_g1",
  "class": "explicit",
  "bbox": { "x": 400, "y": 300, "w": 80, "h": 90 }
}
```

**После конвейера:**

1. **Файл:** `crops/1_yes_g1.png` (кроп из `1_no.png`)

2. **В коллекции `mast_cells_text`:**
   ```
   ID: "1_yes_g1_text"
   Embedding: [0.123, -0.456, 0.789, ..., 0.234]  (384 числа)
   Document: "Cell type: explicit mast cell | Nucleus: central, round/ovoid | ..."
   Metadata: {
     "example_id": "1_yes_g1",
     "image_path": "crops/1_yes_g1.png",
     "cell_type": "explicit",
     "difficulty": "easy",
     "confidence": "high",
     ...
   }
   ```

3. **В коллекции `mast_cells_images` (если CLIP включён):**
   ```
   ID: "1_yes_g1_he"
   Embedding: [0.234, -0.567, 0.890, ..., 0.123]  (512 чисел)
   Metadata: {
     "example_id": "1_yes_g1",
     "image_type": "H&E",
     "image_path": "crops/1_yes_g1.png",
     "cell_type": "explicit",
     ...
   }
   ```

---

## Как это используется для Gemini

Когда вы делаете запрос через `analyze_with_kb.py`:

1. **Поиск похожих примеров:**
   ```python
   similar = kb.search_similar(
       query_text="Find implicit mast cells",
       filter_cell_type="implicit",
       n_results=5
   )
   ```

2. **Формирование промпта:**
   Из найденных примеров берутся:
   - `documents` (текстовые описания)
   - `metadata` (пути к изображениям, классы)
   - Формируется контекстный промпт для Gemini

3. **Отправка Gemini:**
   - Новое изображение для анализа
   - Расширенный промпт с примерами из KB
   - Gemini использует контекст для более точного обнаружения

---

## Что НЕ хранится в KB

- ❌ **Сами изображения** — только пути к файлам (`image_path`)
- ❌ **Исходный JSON разметки** — только извлечённые данные
- ❌ **Полные изображения `_no`** — только кропы
- ❌ **Промпты от библиотекаря** — промпты формируются динамически при поиске

---

## Как посмотреть содержимое KB

### Через код:

```python
from train_knowledge_base import MastCellsKnowledgeBase

kb = MastCellsKnowledgeBase(db_path="./mast_cells_kb")
examples = kb.get_all_examples()

for ex in examples:
    print(f"ID: {ex['id']}")
    print(f"Text: {ex['text']}")
    print(f"Metadata: {ex['metadata']}")
```

### Через библиотекаря:

```bash
python scripts/librarian_kb_helper.py kb_report --kb_path ./mast_cells_kb
```

Выведет статистику: сколько примеров по `cell_type`, `difficulty` и т.д.

---

## Резюме

**KB хранит:**
- ✅ Векторные эмбеддинги (текст + изображения)
- ✅ Метаданные (пути, классы, координаты)
- ✅ Текстовые описания для промптов

**KB НЕ хранит:**
- ❌ JSON файлы разметки
- ❌ Сами изображения (только пути)
- ❌ Готовые промпты (формируются динамически)

**Формат данных:** ChromaDB (SQLite + индексы), не JSON. Для просмотра используйте `get_all_examples()` или библиотекаря.
