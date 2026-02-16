#!/bin/bash
# Скрипт для удаления репозитория mast-about.txt на GitHub

# Проверка наличия токена
if [ -z "$GITHUB_TOKEN" ]; then
    echo "GITHUB_TOKEN не установлен."
    echo ""
    echo "Для автоматического удаления:"
    echo "1. Создайте токен: https://github.com/settings/tokens"
    echo "2. export GITHUB_TOKEN=your_token_here"
    echo "3. Запустите скрипт снова"
    echo ""
    echo "ИЛИ удалите вручную:"
    echo "https://github.com/kobyzev-yuri/mast-about.txt/settings"
    echo "-> Danger Zone -> Delete this repository"
    exit 1
fi

USERNAME="kobyzev-yuri"
REPO="mast-about.txt"

echo "Удаление репозитория $REPO..."
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$USERNAME/$REPO")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

if [ "$HTTP_CODE" = "204" ]; then
    echo "✓ Репозиторий $REPO успешно удален"
    echo ""
    echo "Теперь можно запушить код:"
    echo "  git push -u origin main"
elif [ "$HTTP_CODE" = "404" ]; then
    echo "✓ Репозиторий $REPO не найден (возможно, уже удален)"
else
    echo "✗ Ошибка при удалении (HTTP $HTTP_CODE)"
    echo ""
    echo "Удалите вручную:"
    echo "https://github.com/$USERNAME/$REPO/settings"
    echo "-> Danger Zone -> Delete this repository"
    exit 1
fi

