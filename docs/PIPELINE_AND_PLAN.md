# Конвейер разметки и план реализации

## Порядок работ (приоритет)

1. **Сначала** — **аудит разметки через Gemini** на текущих данных (`yes_annotations.json` + пары `_no`/`_yes`). Получить от Gemini подтверждение/корректировку классов и рекомендации.
2. **После аудита** — тестировать подготовку KB, помощника библиотекаря и боевую настройку Gemini-агента (промпты, примеры, контекст из KB).

---

## Обзор шагов

- **Входные данные:** вы даёте **свой JSON разметки** (формат см. в `docs/ANNOTATION_JSON_FORMAT.md`) и соответствующие изображения **без разметки** (`*_no.png`). На их основе конвейер строит KB.
- **Экспорт из _yes** (опционально): скрипт `export_yes_annotations_to_json.py` может сгенерировать такой JSON из картинок с зелёными/синими рамками.
- **Аудит через Gemini** — проверка согласованности классов и рекомендации *(рекомендуется делать первым)*.
- **Конвейер** JSON + `_no` → кропы → добавление в KB.
- **Помощник библиотекаря** — статистика, запуск конвейера, отчёт по KB.

---

## Шаг 0: Аудит разметки через Gemini (первый шаг)

Цель: получить от Gemini оценку вашей разметки (зелёный=явные, синий=неявные) и рекомендации по единому формату и по наполнению KB. После этого переходить к наполнению KB и настройке агента.

**Требования:** в окружении или в `config.env` (или в `brats/config.env` / `brats/kb-service/config.env`) должен быть задан `GEMINI_API_KEY` (или `OPENAI_API_KEY`).

**Проверка перед запуском:** в одной папке с `yes_annotations.json` должны лежать пары `*_no.png` и `*_yes.png` (например, `data/MAST_GEMINI/test/1_no.png`, `1_yes.png`). Если JSON ещё не создан, сначала выполните шаг 1 (экспорт разметки).

**Команда:**

```bash
cd /path/to/mast
python scripts/audit_annotations_with_gemini.py \
  --annotations data/MAST_GEMINI/test/yes_annotations.json \
  --out results/audit_annotations_gemini.txt \
  --max_pairs 3
```

Скрипт возьмёт до 3 пар файлов вида `1_no.png`/`1_yes.png` из директории с `yes_annotations.json`, отправит их в Gemini и сохранит ответ в `results/audit_annotations_gemini.txt`. Полный текст ответа выводится в консоль (с обрезкой, если очень длинный).

После выполнения имеет смысл зафиксировать вывод аудита и при необходимости скорректировать правила разметки или промпты перед заполнением KB.

---

## Шаг 1: Экспорт разметки в JSON

Скрипт извлекает зелёные и синие боксы из изображений с разметкой и сохраняет их в едином JSON.

```bash
python scripts/export_yes_annotations_to_json.py \
  --mast_dir data/MAST_GEMINI/test \
  --out data/MAST_GEMINI/test/yes_annotations.json
```

Результат: `yes_annotations.json` с полями `schema` и `items[]` (для каждого изображения: `image`, `annotations[]` с `id`, `class`, `bbox`, `color`).

---

## Шаг 2: Конвейер (кропы + KB)

На входе: **ваш JSON разметки** и лежащие рядом с ним файлы `*_no.png` (имя `_no` выводится из поля `image`: например, `1_yes.png` → `1_no.png`). Формат JSON описан в **`docs/ANNOTATION_JSON_FORMAT.md`** и в примере `docs/examples/annotations_minimal_example.json`.

Скрипт читает JSON, вырезает патчи из соответствующих `_no.png` по bbox (с отступом), сохраняет кропы и при необходимости добавляет примеры в Knowledge Base.

**Только кропы** (не требует sentence_transformers/ChromaDB):

```bash
python scripts/pipeline_annotations_to_kb.py \
  --annotations data/MAST_GEMINI/test/yes_annotations.json \
  --crops_dir data/MAST_GEMINI/test/crops \
  --crops_only
```

**Кропы + добавление в KB** (нужны зависимости из `requirements.txt` и рабочее окружение для `train_knowledge_base`):

```bash
python scripts/pipeline_annotations_to_kb.py \
  --annotations data/MAST_GEMINI/test/yes_annotations.json \
  --crops_dir data/MAST_GEMINI/test/crops \
  --kb_path ./mast_cells_kb
```

Опции: `--no_clip` (без CLIP), `--no_skip_existing` (перезаписывать существующие кропы).

---

## Шаг 3: Конвейер и библиотекарь (после аудита)

После получения и учёта результатов аудита — тестирование подготовки KB и помощника библиотекаря (см. шаги 2 и 4 ниже).

---

## Шаг 4: Помощник библиотекаря

Единая точка входа для статистики, конвейера и отчёта по KB:

```bash
# Статистика по JSON (явные/неявные по файлам)
python scripts/librarian_kb_helper.py stats --annotations data/MAST_GEMINI/test/yes_annotations.json

# Запуск конвейера (кропы + KB)
python scripts/librarian_kb_helper.py pipeline \
  --annotations data/MAST_GEMINI/test/yes_annotations.json \
  --crops_dir data/MAST_GEMINI/test/crops \
  --kb_path ./mast_cells_kb

# Только кропы
python scripts/librarian_kb_helper.py pipeline --crops_only ...

# Отчёт по содержимому KB (по cell_type, difficulty)
python scripts/librarian_kb_helper.py kb_report --kb_path ./mast_cells_kb

# Всё по очереди: stats → pipeline → kb_report
python scripts/librarian_kb_helper.py full \
  --annotations data/MAST_GEMINI/test/yes_annotations.json \
  --crops_dir data/MAST_GEMINI/test/crops \
  --kb_path ./mast_cells_kb
```

---

## План реализации (чеклист)

- [x] **A1.** JSON как источник истины — экспорт `_yes` в `yes_annotations.json` (скрипт `export_yes_annotations_to_json.py`).
- [x] **A2.** Конвейер: JSON → кропы из `_no` → сохранение в `crops_dir` (режим `--crops_only`).
- [x] **A3.** Конвейер: добавление кропов в KB с `cell_type` / `difficulty` / `confidence` (без `--crops_only`).
- [x] **B1.** Аудит разметки через Gemini: скрипт `audit_annotations_with_gemini.py` (пары _no/_yes, вопросы по согласованности и формату).
- [x] **B2.** Фиксация правил: после запуска аудита принять решение по красному классу и единому формату явные/неявные.
- [x] **C1.** Помощник библиотекаря: `librarian_kb_helper.py` (stats, pipeline, kb_report, full).
- [ ] **C2.** При необходимости: донастройка промптов и примеров для Gemini на основе вывода аудита и отчётов KB.
- [ ] **C3.** Расширение датасета (50–100+ примеров по типам) и повторный прогон конвейера и аудита.
- [ ] **D.** Интеграция с веб-интерфейсом (если есть): вызов конвейера и библиотекаря через API/UI.

---

## Связанные документы

- `docs/ANNOTATION_JSON_FORMAT.md` — формат входного JSON разметки для конвейера.
- `docs/KB_DATA_STRUCTURE.md` — что именно хранится в KB (эмбеддинги, метаданные, структура ChromaDB).
- `docs/LABEL_CONSISTENCY_AND_CLUSTERING.md` — схема классов, кластеризация, методы.
- `docs/INTELLIGENT_LIBRARIAN.md` — концепция интеллектуального библиотекаря и тесты KB.
- `docs/KNOWLEDGE_BASE_ARCHITECTURE.md` — архитектура KB и эмбеддинги.
- `docs/DATASETS_FOR_KB.md` — требования к типам данных (Hard Positives, Hard Negatives и т.д.).
