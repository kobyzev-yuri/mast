# Конвейер для train/test данных

## Структура данных

- **`data/train/`** — данные для наполнения KB:
  - `annotations.json` — разметка в формате `{ "items": [ { "image": "...", "annotations": [...] } ] }`
  - `images/` — готовые патчи изображений (512x512), уже содержащие мастоциты

- **`data/test/`** — данные для контрольного тестирования:
  - `images/` — тестовые изображения (без разметки)
  - Результат тестирования — JSON в формате train (`predictions.json`)

## Шаг 1: Наполнение KB из train

Используйте `scripts/pipeline_train_to_kb.py` для добавления примеров из train в KB.

**Базовый вариант** (использует всё изображение целиком):

```bash
python scripts/pipeline_train_to_kb.py \
  --annotations data/train/annotations.json \
  --images_dir data/train/images \
  --kb_path ./mast_cells_kb
```

**С вырезкой меньших кропов по bbox** (если хотите сохранить только области мастоцитов):

```bash
python scripts/pipeline_train_to_kb.py \
  --annotations data/train/annotations.json \
  --images_dir data/train/images \
  --kb_path ./mast_cells_kb \
  --crop_bbox \
  --crops_dir data/train/crops
```

**Опции:**
- `--no_clip` — не использовать CLIP эмбеддинги
- `--crop_bbox` — вырезать меньшие кропы по bbox (иначе использует всё изображение)
- `--crops_dir` — директория для кропов (если `--crop_bbox`)

## Шаг 2: Тестирование на test

Используйте `scripts/test_with_kb.py` для анализа тестовых изображений с помощью KB и Gemini.

**Требования:**
- KB должна быть заполнена (шаг 1)
- `GEMINI_API_KEY` должен быть настроен

**Команда:**

```bash
python scripts/test_with_kb.py \
  --test_images_dir data/test/images \
  --kb_path ./mast_cells_kb \
  --out data/test/predictions.json \
  --n_examples 5
```

**Опции:**
- `--n_examples` — количество примеров из KB для контекста (по умолчанию 5)

**Результат:** `data/test/predictions.json` в формате train:
```json
{
  "items": [
    {
      "image": "test_image_001.jpeg",
      "annotations": [
        {
          "id": "test_image_001_cell1",
          "class": "explicit",
          "bbox": { "x": 100, "y": 200, "w": 40, "h": 40 }
        }
      ]
    }
  ]
}
```

## Формат train JSON

Оба скрипта работают с форматом:
```json
{
  "items": [
    {
      "image": "имя_файла.jpeg",
      "annotations": [
        {
          "id": "уникальный_id",
          "class": "explicit" | "implicit",
          "bbox": { "x": число, "y": число, "w": число, "h": число }
        }
      ]
    }
  ]
}
```

- `image` — имя файла изображения (должен существовать в `images_dir`)
- `annotations[].id` — уникальный идентификатор аннотации
- `annotations[].class` — класс мастоцита (`explicit` или `implicit`)
- `annotations[].bbox` — прямоугольник в пикселях (x, y — левый верхний угол; w, h — размеры)

## Примечания

- В train изображения уже готовые патчи (512x512), bbox относительны к этим патчам
- Конвейер train может использовать всё изображение или вырезать меньшие кропы по bbox
- Тестовый скрипт парсит ответ Gemini и извлекает координаты/типы; парсинг можно улучшить под конкретный формат ответов Gemini
