# Инструкция по созданию репозитория на GitHub

## Вариант 1: Через веб-интерфейс GitHub (рекомендуется)

1. Перейдите на страницу создания нового репозитория:
   https://github.com/new

2. Заполните форму:
   - **Repository name**: `mast`
   - **Description**: `Multimodal knowledge base for detecting poorly expressed mast cells using Gemini 3 Pro and RAG`
   - **Visibility**: Public или Private (на ваше усмотрение)
   - **ВАЖНО**: НЕ добавляйте README, .gitignore или лицензию (они уже есть в проекте)

3. Нажмите кнопку **"Create repository"**

4. После создания репозитория выполните команду:
   ```bash
   git push -u origin main
   ```

## Вариант 2: Через GitHub API (автоматически)

Если у вас есть Personal Access Token:

1. Создайте токен на GitHub:
   - Перейдите: https://github.com/settings/tokens
   - Нажмите "Generate new token (classic)"
   - Выберите права: `repo` (полный доступ к репозиториям)
   - Скопируйте токен

2. Установите токен:
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```

3. Запустите скрипт:
   ```bash
   ./create_github_repo.sh
   ```

4. Затем запушите код:
   ```bash
   git push -u origin main
   ```

## Текущий статус

✅ Git репозиторий инициализирован  
✅ Все файлы добавлены в git  
✅ Первый коммит создан  
✅ Remote origin настроен: `https://github.com/kobyzev-yuri/mast.git`  
⏳ Осталось: создать репозиторий на GitHub и запушить код

## Проверка после создания

После успешного push проверьте репозиторий:
https://github.com/kobyzev-yuri/mast

