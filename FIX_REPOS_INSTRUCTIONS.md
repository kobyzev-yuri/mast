# Инструкция по переименованию и удалению репозиториев

На GitHub есть два репозитория:
- `kobyzev-yuri/mast-about.md`
- `kobyzev-yuri/mast-about.txt`

Нужно:
1. Переименовать один из них в `mast`
2. Удалить второй

## Вариант 1: Через веб-интерфейс GitHub (рекомендуется)

### Шаг 1: Переименование репозитория

1. Перейдите в настройки репозитория `mast-about.md`:
   https://github.com/kobyzev-yuri/mast-about.md/settings

2. Прокрутите вниз до секции **"Repository name"**

3. Измените имя с `mast-about.md` на `mast`

4. Нажмите кнопку **"Rename"**

### Шаг 2: Удаление второго репозитория

1. Перейдите в настройки репозитория `mast-about.txt`:
   https://github.com/kobyzev-yuri/mast-about.txt/settings

2. Прокрутите вниз до секции **"Danger Zone"**

3. Нажмите кнопку **"Delete this repository"**

4. Введите имя репозитория для подтверждения: `kobyzev-yuri/mast-about.txt`

5. Нажмите кнопку **"I understand the consequences, delete this repository"**

### Шаг 3: Обновление remote и push

После переименования обновите remote и запушите код:

```bash
cd /mnt/ai/cnn/mast
git remote set-url origin https://github.com/kobyzev-yuri/mast.git
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
   cd /mnt/ai/cnn/mast
   ./fix_github_repos.sh
   ```

4. Затем запушите код:
   ```bash
   git push -u origin main
   ```

## Примечание

Рекомендуется переименовать `mast-about.md` в `mast`, так как это более подходящее имя для проекта.

