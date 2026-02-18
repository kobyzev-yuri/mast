# Структура директорий проекта mast

## Общая структура проекта

```
mast/
├── README.md                          # Основной README
├── requirements.txt                   # Зависимости Python
├── .gitignore                         # Git ignore правила
├── config.env                         # Конфигурация (API ключи, пути)
│
├── # === ОСНОВНОЙ КОД ===
├── core/                              # Ядро системы
│   ├── __init__.py
│   ├── knowledge_base.py              # MastCellsKnowledgeBase (из train_knowledge_base.py)
│   ├── embeddings.py                  # Модели для эмбеддингов (текст + CLIP)
│   ├── gemini_service.py              # Сервис для работы с Gemini API
│   └── data_validator.py              # Валидация данных перед добавлением в KB
│
├── # === АНАЛИЗ И ОБРАБОТКА ===
├── analysis/                          # Скрипты анализа
│   ├── __init__.py
│   ├── analyze_gemini.py              # Первичный анализ через Gemini
│   ├── analyze_coordinates.py         # Определение координат
│   └── analyze_with_kb.py             # Анализ с использованием KB
│
├── # === УПРАВЛЕНИЕ ДАННЫМИ ===
├── data_management/                   # Управление данными для KB
│   ├── __init__.py
│   ├── dataset_loader.py              # Загрузка датасетов разных типов
│   ├── metadata_extractor.py          # Извлечение метаданных из изображений
│   ├── pair_matcher.py                # Сопоставление парных данных (Г/Э + ИГХ)
│   └── quality_checker.py             # Проверка качества данных
│
├── # === ВЕБ-ИНТЕРФЕЙС (ИНТЕЛЛЕКТУАЛЬНЫЙ БИБЛИОТЕКАРЬ) ===
├── web/                               # Веб-интерфейс для управления KB
│   ├── __init__.py
│   ├── app.py                         # Основное FastAPI приложение
│   ├── librarian/                     # Модуль интеллектуального библиотекаря
│   │   ├── __init__.py
│   │   ├── intelligent_librarian.py   # Основная логика библиотекаря
│   │   ├── data_collector.py          # Помощник по сбору данных
│   │   ├── quality_analyzer.py        # Анализатор качества KB
│   │   ├── suggestion_engine.py       # Двигатель предложений по улучшению
│   │   └── test_runner.py             # Запуск тестов KB
│   ├── api/                           # API endpoints
│   │   ├── __init__.py
│   │   ├── kb_routes.py               # Маршруты для работы с KB
│   │   ├── search_routes.py           # Маршруты поиска
│   │   ├── dataset_routes.py          # Маршруты для датасетов
│   │   └── test_routes.py             # Маршруты для тестирования
│   ├── static/                        # Статические файлы (CSS, JS)
│   │   ├── css/
│   │   └── js/
│   ├── templates/                     # HTML шаблоны
│   │   ├── index.html                 # Главная страница
│   │   ├── kb_dashboard.html          # Дашборд KB
│   │   ├── data_collection.html       # Интерфейс сбора данных
│   │   └── testing.html                # Интерфейс тестирования
│   └── streamlit_app.py               # Streamlit интерфейс (альтернатива)
│
├── # === ДАННЫЕ ===
├── data/                              # Исходные данные
│   ├── MAST_GEMINI/                   # Текущие изображения мастоцитов
│   │   ├── *.png                      # Изображения
│   │   └── .gitkeep
│   └── datasets/                      # Структурированные датасеты для KB
│       ├── hard_positives/            # Hard Positives
│       │   ├── spindle_shaped/       # Веретеновидные
│       │   │   ├── 001_he.png
│       │   │   ├── 001_ihc.png
│       │   │   └── 001_metadata.json
│       │   ├── overlapping/          # С наложением
│       │   ├── degranulated/         # Дегранулированные
│       │   ├── edge_cases/           # На краях
│       │   └── clusters/             # Кластеры
│       ├── hard_negatives/           # Hard Negatives
│       │   ├── plasmocytes_central/
│       │   │   ├── 001_he.png
│       │   │   ├── 001_metadata.json
│       │   │   └── 001_distinguishing.json
│       │   ├── fibroblasts/
│       │   ├── histiocytes/
│       │   ├── eosinophils/
│       │   └── lymphocytes/
│       ├── confusion_set/             # Confusion Set
│       │   ├── 001_he.png
│       │   ├── 001_ihc.png           # только если мастоцит
│       │   └── 001_metadata.json
│       ├── ambiguous/                 # Неоднозначные случаи
│       │   ├── 001_he.png
│       │   └── 001_metadata.json
│       └── paired_data/               # Парные данные (Г/Э + ИГХ)
│           ├── registered_pairs/
│           │   ├── 001_he.png
│           │   ├── 001_ihc.png
│           │   └── 001_registration.json
│
├── # === БАЗА ЗНАНИЙ ===
├── mast_cells_kb/                     # ChromaDB база знаний (генерируется)
│   ├── chroma.sqlite3                 # SQLite база
│   ├── index/                         # Индексы для поиска
│   └── .gitignore                     # Не коммитится в git
│
├── # === РЕЗУЛЬТАТЫ ===
├── results/                           # Результаты анализа
│   ├── mast_cells_analysis_result.txt
│   ├── mast_cells_coordinates_analysis_result.json
│   └── mast_cells_coordinates_analysis_result.txt
│
├── # === ТЕСТЫ ===
├── tests/                             # Тесты
│   ├── __init__.py
│   ├── test_knowledge_base.py         # Тесты KB
│   ├── test_embeddings.py             # Тесты эмбеддингов
│   ├── test_search.py                 # Тесты поиска
│   └── test_librarian.py              # Тесты библиотекаря
│
├── # === УТИЛИТЫ ===
├── utils/                             # Вспомогательные утилиты
│   ├── __init__.py
│   ├── image_utils.py                 # Утилиты для работы с изображениями
│   ├── metadata_utils.py               # Утилиты для метаданных
│   └── config_loader.py               # Загрузка конфигурации
│
├── # === ДОКУМЕНТАЦИЯ ===
├── docs/                              # Документация
│   ├── KNOWLEDGE_BASE_ARCHITECTURE.md  # Архитектура KB
│   ├── GEMINI_DATASET_REQUIREMENTS.md # Требования Gemini к датасетам
│   ├── DATASETS_FOR_KB.md             # Руководство по датасетам
│   ├── KB_QUICK_GUIDE.md              # Краткое руководство
│   ├── TRAINING_APPROACHES_FOR_MAST_CELLS.md
│   ├── PROJECT_DIRECTORY_STRUCTURE.md # Этот файл
│   ├── INTELLIGENT_LIBRARIAN.md       # Документация библиотекаря
│   └── WEB_INTERFACE_GUIDE.md         # Руководство по веб-интерфейсу
│
└── # === СКРИПТЫ ===
└── scripts/                           # Вспомогательные скрипты
    ├── setup_kb.sh                    # Настройка KB
    ├── populate_kb.sh                  # Пополнение KB
    └── start_web.sh                    # Запуск веб-интерфейса
```

## Принципы организации

### 1. Разделение по функциональности
- **core/** - ядро системы (KB, эмбеддинги, сервисы)
- **analysis/** - анализ и обработка изображений
- **data_management/** - управление данными
- **web/** - веб-интерфейс и библиотекарь

### 2. Модульность
- Каждый модуль имеет свой `__init__.py`
- Четкие интерфейсы между модулями
- Легко тестировать и расширять

### 3. Данные отдельно от кода
- **data/** - исходные данные (коммитятся)
- **mast_cells_kb/** - сгенерированная KB (не коммитится)
- **results/** - результаты анализа (можно коммитить)

### 4. Документация рядом с кодом
- **docs/** - подробная документация
- Комментарии в коде
- Примеры использования

## Миграция существующего кода

### Текущие файлы → Новая структура

```
analyze_mast_cells_gemini.py          → analysis/analyze_gemini.py
analyze_mast_cells_coordinates_gemini.py → analysis/analyze_coordinates.py
analyze_with_kb.py                    → analysis/analyze_with_kb.py
train_knowledge_base.py               → core/knowledge_base.py
add_datasets_to_kb.py                 → data_management/dataset_loader.py
```

## Преимущества новой структуры

1. **Четкая организация** - легко найти нужный код
2. **Масштабируемость** - легко добавлять новые модули
3. **Тестируемость** - каждый модуль можно тестировать отдельно
4. **Документированность** - структура понятна из названий
5. **Готовность к веб-интерфейсу** - отдельная папка web/

