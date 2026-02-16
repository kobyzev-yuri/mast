# Структура проекта mast

## Обзор

Проект для анализа мастоцитов через Gemini 3 Pro и Knowledge Base (RAG).

## Файловая структура

```
mast/
├── README.md                          # Основной README
├── requirements.txt                   # Зависимости Python
├── .gitignore                         # Git ignore правила
├── SETUP_GIT.md                       # Инструкции по настройке Git
├── PROJECT_STRUCTURE.md               # Этот файл
│
├── Скрипты анализа:
│   ├── analyze_mast_cells_gemini.py              # Первичный анализ
│   ├── analyze_mast_cells_coordinates_gemini.py  # Определение координат
│   └── analyze_with_kb.py                        # Анализ с KB
│
├── Скрипты Knowledge Base:
│   ├── train_knowledge_base.py        # Управление KB
│   └── add_datasets_to_kb.py          # Добавление датасетов
│
├── Документация:
│   ├── README_MAST_CELLS_ANALYSIS.md
│   ├── README_MAST_CELLS_COORDINATES_ANALYSIS.md
│   ├── README_TRAINING_MAST_CELLS.md
│   └── MAST_CELLS_COORDINATES_ANALYSIS_SUMMARY.md
│
├── docs/
│   ├── GEMINI_DATASET_REQUIREMENTS.md    # Требования Gemini к датасетам
│   ├── TRAINING_APPROACHES_FOR_MAST_CELLS.md  # Подходы к обучению
│   ├── KNOWLEDGE_BASE_ARCHITECTURE.md    # Архитектура KB
│   ├── KB_QUICK_GUIDE.md                 # Краткое руководство
│   └── DATASETS_FOR_KB.md                # Руководство по датасетам
│
├── data/
│   └── MAST_GEMINI/                    # Изображения мастоцитов
│       ├── 01_no.png, 01_yes.png
│       ├── 02_no.png, 02_yes.png
│       ├── 03_no.png, 03_yes.png
│       └── *_игх.png (ИГХ эталоны)
│
└── results/                            # Результаты анализа
    ├── mast_cells_analysis_result.txt
    ├── mast_cells_coordinates_analysis_result.json
    └── mast_cells_coordinates_analysis_result.txt
```

## Пути в проекте

### Изображения:
- `data/MAST_GEMINI/` - все изображения мастоцитов

### Результаты:
- `results/` - результаты анализа

### Knowledge Base:
- `./mast_cells_kb/` - база знаний (создается автоматически)

### Конфигурация:
- `config.env` - локальная конфигурация (приоритет)
- `../../brats/kb-service/config.env` - конфигурация из brats (резерв)

## Зависимости от других проектов

Проект может использовать конфигурацию из `../../brats/kb-service/config.env`, но может работать и независимо с локальным `config.env`.

## Дата создания

2025-02-15 (выделен из проекта sc)

