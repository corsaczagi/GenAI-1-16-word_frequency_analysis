import nltk
from nltk.corpus import webtext
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_nltk_resources():
    try:
        nltk.download('webtext', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        logger.error(f"Ошибка при загрузке NLTK ресурсов: {e}")
        return False

def load_text_data(filename='overheard.txt'):
    available_files = webtext.fileids()
    if filename not in available_files:
        raise ValueError(f"Файл {filename} не найден. Доступные файлы: {available_files}")

    text = webtext.raw(filename)
    return text

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return words

def calculate_frequency_distribution(words):
    return FreqDist(words)

def plot_top_words(fdist, top_n=10, title="Топ самых частых слов"):
    top_words = fdist.most_common(top_n)
    words_list, frequencies = zip(*top_words)

    plt.figure(figsize=(14, 8))
    bars = plt.bar(words_list, frequencies, color='skyblue', edgecolor='black', alpha=0.7)

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Слова', fontsize=12, fontweight='bold')
    plt.ylabel('Частота', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, frequency in zip(bars, frequencies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(frequency), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

def main():
    TEXT_FILE = 'overheard.txt'
    TOP_N_WORDS = 10

    try:
        if not download_nltk_resources():
            raise RuntimeError("Не удалось загрузить NLTK ресурсы")

        text = load_text_data(TEXT_FILE)
        processed_words = preprocess_text(text)
        frequency_distribution = calculate_frequency_distribution(processed_words)

        total_words = len(processed_words)
        unique_words = len(set(processed_words))
        lexical_diversity = unique_words / total_words if total_words > 0 else 0


        plot_top_words(
            frequency_distribution,
            top_n=TOP_N_WORDS,
            title=f'Топ-{TOP_N_WORDS} самых частых слов (без стоп-слов)\nФайл: {TEXT_FILE}'
        )

        return True

    except Exception as e:
        logger.error(f"Ошибка при выполнении анализа: {e}")
        print(f"\n❌ Ошибка: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
