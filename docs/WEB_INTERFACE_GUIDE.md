# Руководство по веб-интерфейсу для Knowledge Base

## Обзор

Веб-интерфейс предоставляет удобный способ управления мультимодальной базой знаний для детекции мастоцитов через браузер.

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    ВЕБ-ИНТЕРФЕЙС                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Frontend (HTML/CSS/JS)                                 │
│  ├── Главная страница (Dashboard)                      │
│  ├── Сбор данных (Data Collection)                    │
│  ├── Поиск и тестирование (Search & Testing)          │
│  └── Отчеты (Reports)                                  │
│                                                          │
│  Backend (FastAPI)                                      │
│  ├── API Routes                                         │
│  │   ├── /api/kb/* - работа с KB                      │
│  │   ├── /api/search/* - поиск                        │
│  │   ├── /api/datasets/* - управление датасетами      │
│  │   └── /api/librarian/* - библиотекарь              │
│  │                                                      │
│  └── Intelligent Librarian                             │
│      ├── Data Collector                                │
│      ├── Quality Analyzer                              │
│      ├── Suggestion Engine                             │
│      └── Test Runner                                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Основные страницы

### 1. Dashboard (Главная страница)

**URL:** `/`

**Функции:**
- Общая статистика KB
- Графики распределения данных
- Быстрый доступ к основным функциям
- Последние добавленные примеры

**Визуализация:**
- Круговая диаграмма типов данных
- Столбчатая диаграмма по категориям
- Карта покрытия (heatmap)
- Метрики качества

### 2. Data Collection (Сбор данных)

**URL:** `/data-collection`

**Функции:**
- Загрузка изображений (Г/Э и ИГХ)
- Заполнение метаданных
- Выбор типа данных (Hard Positive, Hard Negative, etc.)
- Валидация перед добавлением
- Предпросмотр перед сохранением

**Форма добавления:**
```
┌─────────────────────────────────────┐
│ Тип данных: [Hard Positive ▼]     │
│                                     │
│ Г/Э изображение: [Выбрать файл]    │
│ ИГХ изображение: [Выбрать файл]    │
│                                     │
│ Морфологические признаки:          │
│ - Ядро: [________________]         │
│ - Цитоплазма: [________________]  │
│ - Форма: [________________]        │
│ - Локация: [________________]      │
│                                     │
│ Координаты:                        │
│ - X: [____] Y: [____]              │
│                                     │
│ [Валидировать] [Добавить в KB]     │
└─────────────────────────────────────┘
```

### 3. Search & Testing (Поиск и тестирование)

**URL:** `/search-testing`

**Функции:**
- Поиск по текстовому запросу
- Поиск по изображению (мультимодальный)
- Фильтрация результатов
- Тестирование качества поиска
- Визуализация результатов

**Интерфейс поиска:**
```
┌─────────────────────────────────────┐
│ Поиск:                              │
│ [Текстовый запрос или изображение] │
│                                     │
│ Фильтры:                            │
│ ☑ Hard Positives                    │
│ ☐ Hard Negatives                    │
│ ☐ Confusion Set                    │
│                                     │
│ Тип: [Все ▼]                        │
│ Сложность: [Все ▼]                 │
│                                     │
│ [Найти]                             │
│                                     │
│ Результаты (5):                     │
│ ┌─────────────────────────────────┐ │
│ │ 1. Spindle-shaped mast cell     │ │
│ │    Similarity: 0.85             │ │
│ │    [Показать изображение]       │ │
│ └─────────────────────────────────┘ │
│ ...                                 │
└─────────────────────────────────────┘
```

### 4. Testing (Тестирование)

**URL:** `/testing`

**Функции:**
- Запуск тестов KB
- Просмотр результатов тестов
- История тестов
- Настройка тестовых сценариев

**Интерфейс тестирования:**
```
┌─────────────────────────────────────┐
│ Тесты KB:                           │
│                                     │
│ ☑ Тесты поиска                     │
│ ☑ Тесты качества данных            │
│ ☑ Тесты баланса                    │
│ ☐ Тесты мультимодальности          │
│                                     │
│ [Запустить тесты]                   │
│                                     │
│ Результаты:                         │
│ ✅ Поиск Hard Positives: PASSED    │
│ ✅ Фильтрация Hard Negatives: PASSED│
│ ⚠️ Баланс данных: WARNING           │
│    Соотношение HP/HN: 0.3 (норма: 0.5-2.0)
│                                     │
│ [Экспорт отчета]                    │
└─────────────────────────────────────┘
```

### 5. Reports (Отчеты)

**URL:** `/reports`

**Функции:**
- Генерация отчетов о состоянии KB
- Экспорт отчетов (PDF, JSON, CSV)
- История изменений
- Рекомендации по улучшению

## API Endpoints

### KB Management

```python
# Получить статистику KB
GET /api/kb/statistics
Response: {
    "total_examples": 500,
    "hard_positives": 120,
    "hard_negatives": 250,
    "confusion_set": 100,
    "ambiguous": 30
}

# Получить все примеры
GET /api/kb/examples?limit=100&offset=0
Response: {
    "examples": [...],
    "total": 500,
    "limit": 100,
    "offset": 0
}

# Добавить пример
POST /api/kb/examples
Body: {
    "example_id": "hp_001",
    "image_path": "...",
    "ihc_image_path": "...",
    "morphological_features": {...},
    "dataset_category": "hard_positive",
    ...
}
Response: {"success": true, "id": "hp_001"}

# Удалить пример
DELETE /api/kb/examples/{example_id}
Response: {"success": true}
```

### Search

```python
# Текстовый поиск
POST /api/search/text
Body: {
    "query": "spindle-shaped mast cell",
    "n_results": 5,
    "filters": {
        "dataset_category": "hard_positive",
        "cell_type": "implicit"
    }
}
Response: {
    "results": [...],
    "query": "...",
    "n_results": 5
}

# Поиск по изображению
POST /api/search/image
Body: {
    "image": "base64_encoded_image",
    "n_results": 5,
    "image_type": "H&E"
}
Response: {
    "results": [...],
    "n_results": 5
}
```

### Datasets

```python
# Получить список датасетов
GET /api/datasets
Response: {
    "datasets": [
        {
            "name": "hard_positives",
            "path": "data/datasets/hard_positives/",
            "count": 120,
            "types": ["spindle_shaped", "overlapping", ...]
        },
        ...
    ]
}

# Добавить датасет в KB
POST /api/datasets/add
Body: {
    "dataset_path": "data/datasets/hard_positives/spindle_shaped/",
    "dataset_type": "hard_positives",
    "use_clip": true
}
Response: {"success": true, "added": 15}
```

### Librarian

```python
# Анализ состояния KB
GET /api/librarian/analyze
Response: {
    "statistics": {...},
    "gaps": [...],
    "suggestions": [...]
}

# Получить предложения
GET /api/librarian/suggestions
Response: {
    "data_collection": [...],
    "improvements": [...],
    "testing": [...]
}

# Валидация данных
POST /api/librarian/validate
Body: {...}
Response: {
    "valid": true,
    "errors": [],
    "warnings": []
}

# Запуск тестов
POST /api/librarian/test
Body: {
    "test_types": ["search", "quality", "balance"]
}
Response: {
    "passed": 8,
    "failed": 1,
    "total": 9,
    "results": [...]
}
```

## Запуск веб-интерфейса

### Вариант 1: FastAPI (рекомендуется)

```bash
# Установка зависимостей
pip install fastapi uvicorn python-multipart

# Запуск
cd /mnt/ai/cnn/mast
python -m web.app

# Или через uvicorn
uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
```

### Вариант 2: Streamlit (альтернатива)

```bash
# Установка
pip install streamlit

# Запуск
streamlit run web/streamlit_app.py
```

## Безопасность

- Аутентификация через токены (опционально)
- Валидация всех входных данных
- Ограничение размера загружаемых файлов
- CORS настройки для production

## Расширения

### Планируемые функции:
1. Экспорт/импорт KB
2. Версионирование KB
3. Сравнение версий KB
4. Интеграция с Gemini для автоматической аннотации
5. Batch обработка изображений
6. Автоматическая регистрация парных данных (Г/Э + ИГХ)

