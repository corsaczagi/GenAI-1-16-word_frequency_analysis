import nltk
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pymorphy3
import argparse
import os

# Загрузка данных
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

DEFAULT_ENCODING = 'utf-8'


def setup_argparse():
    """
    Настройка аргументов командной строки
    """
    parser = argparse.ArgumentParser(description='Анализ частоты слов в текстовом файле')
    parser.add_argument('filename', help='Путь к текстовому файлу для анализа')
    return parser


def read_text_from_file(filename):
    """
    Читает текст из файла
    """
    try:
        with open(filename, 'r', encoding=DEFAULT_ENCODING) as file:
            return file.read()
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден!")
        raise
    except Exception as e:
        print(f"Неожиданная ошибка при чтении файла: {e}")
        raise


def preprocess_text(text):
    """
    Предобработка текста: токенизация, фильтрация, приведение к нижнему регистру
    Исключение стоп-слов (русских и английских)
    """
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]

    # Исключение стоп-слов (русских и английских)
    stop_words_english = set(stopwords.words('english'))
    stop_words_russian = set(stopwords.words('russian'))
    all_stop_words = stop_words_english.union(stop_words_russian)

    words = [word for word in words if word not in all_stop_words]

    return words


def normalize_words(words):
    """
    Нормализация слов с помощью pymorphy3
    """
    morph = pymorphy3.MorphAnalyzer()
    normalized_words = []

    for word in words:
        parsed = morph.parse(word)[0]
        normalized_words.append(parsed.normal_form)

    return normalized_words


def calculate_word_frequencies(words):
    """
    Подсчет частоты слов (FreqDist)
    """
    return FreqDist(words)


def plot_top_words(fdist, top_n=10):
    """
    Построение bar plot топ-N слов с подписанными осями
    """
    top_words = [word for word, count in fdist.most_common(top_n)]
    counts = [count for word, count in fdist.most_common(top_n)]

    plt.figure(figsize=(12, 6))
    plt.bar(top_words, counts, color='skyblue', edgecolor='black')

    # Подписи осей
    plt.xlabel('Слова', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.title(f'Топ-{top_n} самых частых слов', fontsize=14)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    """
    Основная функция для выполнения всех задач
    """
    parser = setup_argparse()
    args = parser.parse_args()

    try:
        # Проверка существования файла
        if not os.path.exists(args.filename):
            print(f"Ошибка: файл '{args.filename}' не найден!")
            return

        # 1. Чтение файла
        text = read_text_from_file(args.filename)

        # 2. Предобработка текста (исключение стоп-слов)
        words = preprocess_text(text)

        # 3. Нормализация слов
        normalized_words = normalize_words(words)

        # 4. Подсчет частоты слов (FreqDist)
        fdist = calculate_word_frequencies(normalized_words)

        # 5. Построение bar plot топ-10 с подписанными осями
        plot_top_words(fdist, top_n=10)

    except Exception as e:
        print(f"Ошибка: {e}")


# Запуск программы
if __name__ == "__main__":
    main()
