import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
import pymorphy3

# Загрузка необходимых ресурсов nltk без вывода результатов
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Ввод данных
filename = input("Введите название текстового файла: ")
language = input("Выберите язык (en/ru): ").lower().strip()

# Обработка ввода выбора языка (если не en, ru - конец программы)
if language not in ['en','ru']:
    print("Ошибка: неправильный выбор языка")
    exit()

# Основаной анализ
try:
    # Чтение файла
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # Токенизация с обработкой регистров
    word_tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]

    # Определяем стоп-слова (добавил стоп-слова в русский)
    if language == 'ru':
        stop_words = set(stopwords.words('russian'))
        stop_words.update(['это', 'свой', 'свои'])
    else:
        stop_words = set(stopwords.words('english'))

    # Удаляем стоп-слова из списка токенов
    filtered_tokens = [word for word in word_tokens if word not in stop_words]

    # Русская обработка с использованием pymorphy для корректной лемматизации
    if language == 'ru':
        # morph - анализатор
        morph = pymorphy3.MorphAnalyzer()
        lemmatized_words = []

        # Лемматизация на русском
        for word in filtered_tokens:
            try:
                # Агрумент 0 - выбираем самый вероятный разбор
                parsed = morph.parse(word)[0]
                # normal_form - получаем начальную форму слова
                normal_form = parsed.normal_form
                # Добавляем в список лемматизированных слов
                lemmatized_words.append(normal_form)

            except:
                # Вставляет изначальное слово, если слово незнакомо(нет в словаре pymorphy)
                lemmatized_words.append(word)

    # Английская обработка (лемматизация без доп. библиотеки)
    else:
        # Создаем лемматизатор lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Получаем начальные формы слов
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Подсчет частот (freqDist = [('Слово', кол-во)] - по уменьшению)
    fdist = FreqDist(lemmatized_words)

    # Выбираем топ-5 слов и их частоты
    top_words = [word for word, count in fdist.most_common(5)]
    top_counts = [count for word, count in fdist.most_common(5)]

    # Визуализация со столбчатой диаграммой и точными значениями столбцов
    bars = plt.bar(top_words, top_counts, color="green")
    plt.bar_label(bars, labels=top_counts, fontsize=10, padding=-15, color="white")
    plt.xlabel("Слова")
    plt.ylabel("Частота")
    plt.title(f"Топ-5 слов ({'русский' if language == 'ru' else 'английский'})")
    plt.show()

# Ошибки с открытием текстового файла
except FileNotFoundError:
    print(f"Ошибка: файл '{filename}' не найден!")

