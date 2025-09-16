import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
import pymorphy3

def download_nltk_resources():
    """
    Загружает необходимые ресурсы nltk без вывода логов
    """
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def read_text_from_file(filename):
    """
    Читает текст из файла

    Args:
        filename (str): название файла для чтения

    Returns:
        str: текст из файла

    Errors:
        FileNotFoundError: если файл не найден
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def tokenize_and_filter(text, language):
    """
    Токенизирует текст, приводит к нижнему регистру и удаляет стоп-слова

    Args:
        text (str): исхооный текст
        language (str): язык текста ('en' или 'ru')

    Returns:
        list: отфильтрованные токены
    """
    word_tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]

    if language == 'ru':
        stop_words = set(stopwords.words('russian'))
        stop_words.update(['это', 'свой', 'свои'])
    else:
        stop_words = set(stopwords.words('english'))

    return [word for word in word_tokens if word not in stop_words]

def lemmatize_words(filtered_tokens, language):
    """
    Лемматизирует список токенов в зависимости от языка

    Args:
        filtered_tokens (list): список отфильтрованных токенов
        language (str): язык текста ('en' или 'ru')

    Returns:
        list: лемматизированные слова
    """
    if language == 'ru':
        morph = pymorphy3.MorphAnalyzer()
        lemmatized_words = []
        for word in filtered_tokens:
            try:
                parsed = morph.parse(word)[0]
                lemmatized_words.append(parsed.normal_form)
            except:
                lemmatized_words.append(word)
        return lemmatized_words
    else:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in filtered_tokens]

def plot_top_words(fdist, language):
    """
    Строит столбчатую диаграмму топ-5 слов

    Args:
        fdist (FreqDist): распределение частот слов.
        language (str): язык текста ('en' или 'ru').
    """
    top_words = [word for word, count in fdist.most_common(5)]
    top_counts = [count for word, count in fdist.most_common(5)]

    bars = plt.bar(top_words, top_counts, color="green")
    plt.bar_label(bars, labels=top_counts, fontsize=10, padding=-15, color="white")
    plt.xlabel("Слова")
    plt.ylabel("Частота")
    plt.title(f"Топ-5 слов ({'русский' if language == 'ru' else 'английский'})")
    plt.show()

def process_text(filename, language):
    """
    Обрабатывает текст из файла: читает, токенизирует, лемматизирует и строит график топ-5 слов

    Args:
        filename (str): название текстового файла
        language (str): язык текста ('en' или 'ru')

    Errors:
        FileNotFoundError: если файл не найден
    """
    try:
        text = read_text_from_file(filename)
        filtered_tokens = tokenize_and_filter(text, language)
        lemmatized_words = lemmatize_words(filtered_tokens, language)
        fdist = FreqDist(lemmatized_words)
        plot_top_words(fdist, language)
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден!")
        raise


def main():
    """
    Основная функция для запуска анализа текста.
    """
    download_nltk_resources()

    filename = input("Введите название текстового файла: ")
    language = input("Выберите язык (en/ru): ").lower().strip()

    if language not in ['en', 'ru']:
        print("Ошибка: неправильный выбор языка")
        return

    process_text(filename, language)

if __name__ == "__main__":
    main()
