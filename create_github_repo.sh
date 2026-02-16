#!/bin/bash
# Скрипт для создания репозитория на GitHub через API

# Проверка наличия токена
if [ -z "$GITHUB_TOKEN" ]; then
    echo "ОШИБКА: Необходимо установить переменную окружения GITHUB_TOKEN"
    echo ""
    echo "Инструкция:"
    echo "1. Создайте Personal Access Token на GitHub:"
    echo "   https://github.com/settings/tokens"
    echo "   Нужны права: repo (полный доступ к репозиториям)"
    echo ""
    echo "2. Установите токен:"
    echo "   export GITHUB_TOKEN=your_token_here"
    echo ""
    echo "3. Запустите скрипт снова:"
    echo "   ./create_github_repo.sh"
    echo ""
    echo "ИЛИ создайте репозиторий вручную:"
    echo "1. Перейдите на https://github.com/kobyzev-yuri?tab=repositories"
    echo "2. Нажмите 'New'"
    echo "3. Название: mast"
    echo "4. Описание: Multimodal knowledge base for detecting poorly expressed mast cells"
    echo "5. Выберите Public или Private"
    echo "6. НЕ добавляйте README, .gitignore или лицензию (они уже есть)"
    echo "7. Нажмите 'Create repository'"
    echo "8. Затем выполните: git push -u origin main"
    exit 1
fi

# Создание репозитория через GitHub API
echo "Создание репозитория mast на GitHub..."

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d '{
    "name": "mast",
    "description": "Multimodal knowledge base for detecting poorly expressed mast cells using Gemini 3 Pro and RAG",
    "private": false,
    "has_issues": true,
    "has_projects": true,
    "has_wiki": true
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "201" ]; then
    echo "✓ Репозиторий успешно создан!"
    echo ""
    echo "Теперь можно запушить код:"
    echo "  git push -u origin main"
elif [ "$HTTP_CODE" = "422" ]; then
    echo "⚠ Репозиторий уже существует или имя занято"
    echo ""
    echo "Попробуйте запушить код:"
    echo "  git push -u origin main"
else
    echo "✗ Ошибка при создании репозитория (HTTP $HTTP_CODE)"
    echo "Ответ: $BODY"
    echo ""
    echo "Попробуйте создать репозиторий вручную (см. инструкции выше)"
    exit 1
fi

