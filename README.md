# Анализ частотности слов

**Скрипт для токенизации, лемматизации и визуализации топ-5 слов на русском/английском.**

---

##  Возможности
- Поддержка **русского** и **английского** языков.
- Лемматизация с `pymorphy3` (RU) и `WordNetLemmatizer` (EN).
- Визуализация топ-5 слов.

---
## Требования
- **Python 3.8+** (рекомендуется 3.10 или новее).
- **pip** (устанавливается вместе с Python).
---

##  Установка библиотек
  ```bash
  pip install nltk pymorphy3 matplotlib
  ```
---

##  Запуск
1. Поместите текстовый файл (например, `text.txt`) в папку со скриптом.
2. Запустите:
```bash
python word_analyzer.py <файл> <язык>
```
3. Примеры:
```bash
python word_analyzer.py AIW.en.txt en
python word_analyzer.py текст.txt ru
```
---
Пример вывода:

<img width="790" height="675" alt="image" src="https://github.com/user-attachments/assets/fef6b2d2-f772-4ea2-8a91-6b697a76a590" />

