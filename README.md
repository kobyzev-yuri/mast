# Проект анализа мастоцитов через Gemini 3 Pro и Knowledge Base

Проект для обнаружения и анализа мастоцитов на гистологических изображениях с использованием Gemini 3 Pro Vision API и Knowledge Base (RAG).

## Описание проекта

Этот проект реализует:
- Анализ изображений мастоцитов через Gemini 3 Pro
- Определение координат мастоцитов (явных и неявных)
- Анализ ошибок обнаружения
- Knowledge Base (RAG) для улучшения результатов
- Мультимодальный подход: текстовые эмбеддинги + CLIP для изображений
- Поддержка различных типов датасетов (Hard Positives, Hard Negatives, Confusion Set)

## Структура проекта

```
mast/
├── README.md                          # Этот файл
├── requirements.txt                    # Зависимости
├── .gitignore                         # Git ignore правила
├── SETUP_GIT.md                       # Инструкции по настройке Git
│
├── analyze_mast_cells_gemini.py       # Первичный анализ через Gemini
├── analyze_mast_cells_coordinates_gemini.py  # Определение координат
├── analyze_with_kb.py                 # Анализ с использованием KB
├── train_knowledge_base.py             # Управление Knowledge Base
├── add_datasets_to_kb.py              # Добавление датасетов в KB
│
├── data/
│   └── MAST_GEMINI/                    # Изображения мастоцитов
│       ├── 01_no.png, 01_yes.png
│       ├── 02_no.png, 02_yes.png
│       ├── 03_no.png, 03_yes.png
│       └── *_игх.png (ИГХ эталоны)
│
├── docs/
│   ├── GEMINI_DATASET_REQUIREMENTS.md # Требования к датасетам от Gemini
│   ├── TRAINING_APPROACHES_FOR_MAST_CELLS.md  # Подходы к обучению
│   ├── KNOWLEDGE_BASE_ARCHITECTURE.md # Архитектура KB
│   ├── KB_QUICK_GUIDE.md              # Краткое руководство по KB
│   └── DATASETS_FOR_KB.md             # Руководство по датасетам
│
├── results/                            # Результаты анализа
│   ├── mast_cells_analysis_result.txt
│   ├── mast_cells_coordinates_analysis_result.json
│   └── mast_cells_coordinates_analysis_result.txt
│
└── README_*.md                        # Дополнительные README файлы
    ├── README_MAST_CELLS_ANALYSIS.md
    ├── README_MAST_CELLS_COORDINATES_ANALYSIS.md
    └── README_TRAINING_MAST_CELLS.md
```

## Быстрый старт

### Установка зависимостей

```bash
cd /mnt/ai/cnn/mast
pip install -r requirements.txt
```

### Настройка

Создайте файл `config.env` в корне проекта или используйте конфигурацию из `../../brats/kb-service/config.env`:

```env
GEMINI_API_KEY=your_proxyapi_key_here
GEMINI_BASE_URL=https://api.proxyapi.ru/google
GEMINI_MODEL=gemini-3-pro-preview
```

### Использование

1. **Создание Knowledge Base:**
```bash
python train_knowledge_base.py --action populate
```

2. **Анализ с использованием KB:**
```bash
python analyze_with_kb.py --image data/MAST_GEMINI/01_no.png
```

3. **Добавление датасетов в KB:**
```bash
python add_datasets_to_kb.py --data_dir ./datasets --dataset_type all
```

## Документация

- [README_MAST_CELLS_ANALYSIS.md](README_MAST_CELLS_ANALYSIS.md) - Первичный анализ
- [README_MAST_CELLS_COORDINATES_ANALYSIS.md](README_MAST_CELLS_COORDINATES_ANALYSIS.md) - Анализ координат
- [README_TRAINING_MAST_CELLS.md](README_TRAINING_MAST_CELLS.md) - Подходы к обучению
- [docs/KNOWLEDGE_BASE_ARCHITECTURE.md](docs/KNOWLEDGE_BASE_ARCHITECTURE.md) - Архитектура KB
- [docs/GEMINI_DATASET_REQUIREMENTS.md](docs/GEMINI_DATASET_REQUIREMENTS.md) - Требования к датасетам
- [docs/DATASETS_FOR_KB.md](docs/DATASETS_FOR_KB.md) - Руководство по датасетам

## История проекта

Этот проект был выделен из проекта `sc` (шкала патологии) для лучшей организации кода.

**Дата переноса:** 2025-02-15

**Исходный проект:** `/mnt/ai/cnn/sc`

## Связь с проектом sc

Проект `sc` фокусируется на построении шкалы патологии (0-1) для анализа WSI изображений.

Проект `mast` фокусируется на обнаружении мастоцитов через Gemini 3 Pro и Knowledge Base.

Результаты анализа мастоцитов могут использоваться в проекте `sc` для построения шкалы, но каждый проект имеет свою четкую область ответственности.
