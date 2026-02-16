# Настройка Git репозитория для проекта mast

## Инициализация репозитория

```bash
cd /mnt/ai/cnn/mast

# Инициализация git
git init

# Добавление файлов
git add .

# Первый коммит
git commit -m "Initial commit: Mast cells analysis project with Gemini 3 Pro and Knowledge Base"
```

## Создание репозитория на GitHub

1. Создайте новый репозиторий на GitHub с именем `mast`
2. Добавьте remote:

```bash
git remote add origin https://github.com/kobyzev-yuri/mast.git
git branch -M main
git push -u origin main
```

## Структура для коммита

Проект готов к коммиту со следующей структурой:

```
mast/
├── README.md
├── requirements.txt
├── .gitignore
├── analyze_mast_cells_gemini.py
├── analyze_mast_cells_coordinates_gemini.py
├── analyze_with_kb.py
├── train_knowledge_base.py
├── add_datasets_to_kb.py
├── data/
│   └── MAST_GEMINI/
│       └── .gitkeep
├── docs/
│   ├── GEMINI_DATASET_REQUIREMENTS.md
│   ├── TRAINING_APPROACHES_FOR_MAST_CELLS.md
│   ├── KNOWLEDGE_BASE_ARCHITECTURE.md
│   ├── KB_QUICK_GUIDE.md
│   └── DATASETS_FOR_KB.md
├── results/
│   ├── mast_cells_analysis_result.txt
│   ├── mast_cells_coordinates_analysis_result.json
│   └── mast_cells_coordinates_analysis_result.txt
└── README_*.md (дополнительные README файлы)
```

## Примечания

- Изображения в `data/MAST_GEMINI/` не коммитятся (добавлены в .gitignore)
- Результаты анализа можно коммитить для истории
- Knowledge Base (`mast_cells_kb/`) не коммитится (добавлена в .gitignore)

