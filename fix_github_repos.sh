#!/bin/bash
# Скрипт для переименования и удаления репозиториев на GitHub

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
    echo "   ./fix_github_repos.sh"
    echo ""
    echo "ИЛИ выполните вручную через веб-интерфейс:"
    echo "1. Переименование: https://github.com/kobyzev-yuri/mast-about.md/settings -> Repository name -> mast"
    echo "2. Удаление: https://github.com/kobyzev-yuri/mast-about.txt/settings -> Danger Zone -> Delete this repository"
    exit 1
fi

USERNAME="kobyzev-yuri"
REPO1="mast-about.md"
REPO2="mast-about.txt"
NEW_NAME="mast"

echo "Работа с репозиториями GitHub..."
echo ""

# Шаг 1: Переименовать mast-about.md в mast
echo "1. Переименование $REPO1 в $NEW_NAME..."
RESPONSE1=$(curl -s -w "\n%{http_code}" -X PATCH \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$USERNAME/$REPO1" \
  -d "{\"name\": \"$NEW_NAME\"}")

HTTP_CODE1=$(echo "$RESPONSE1" | tail -n1)
BODY1=$(echo "$RESPONSE1" | sed '$d')

if [ "$HTTP_CODE1" = "200" ]; then
    echo "✓ Репозиторий $REPO1 успешно переименован в $NEW_NAME"
else
    echo "✗ Ошибка при переименовании $REPO1 (HTTP $HTTP_CODE1)"
    echo "Ответ: $BODY1"
    echo ""
    echo "Попробуйте переименовать вручную:"
    echo "https://github.com/$USERNAME/$REPO1/settings"
    exit 1
fi

echo ""

# Шаг 2: Удалить mast-about.txt
echo "2. Удаление репозитория $REPO2..."
read -p "Вы уверены, что хотите удалить $REPO2? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Отменено. Репозиторий $REPO2 не был удален."
    exit 0
fi

RESPONSE2=$(curl -s -w "\n%{http_code}" -X DELETE \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$USERNAME/$REPO2")

HTTP_CODE2=$(echo "$RESPONSE2" | tail -n1)

if [ "$HTTP_CODE2" = "204" ]; then
    echo "✓ Репозиторий $REPO2 успешно удален"
else
    echo "✗ Ошибка при удалении $REPO2 (HTTP $HTTP_CODE2)"
    echo ""
    echo "Попробуйте удалить вручную:"
    echo "https://github.com/$USERNAME/$REPO2/settings"
    exit 1
fi

echo ""
echo "✓ Готово! Теперь можно запушить код:"
echo "  git push -u origin main"

