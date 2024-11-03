"""
Модуль для создания кратких аннотаций текста с использованием алгоритма LexRank.

Данный модуль предназначен для суммаризации текста из файла с использованием
метода LexRank из библиотеки Sumy. Пользователь может выбрать два уровня
сжатия текста:
- 'strong' — для создания более короткой и концентрированной аннотации,
- 'weak' — для создания более полной аннотации с сохранением большего количества
  исходной информации.

Основные функции:
- `text_preprocess`: предварительная обработка текста, замена некоторых знаков
  препинания для улучшения разбиения текста на предложения.
- `sumy_lexrank_weak`: создание аннотации текста с низким уровнем сжатия (длинное резюме).
- `sumy_lexrank_strong`: создание аннотации текста с высоким уровнем сжатия (короткое резюме).

Пример использования:
    Введите путь до файла и выберите уровень сжатия (strong или weak).
    Модуль прочитает текст из указанного файла и выведет его аннотацию.

Зависимости:
- sumy
"""

from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import nltk

SENTENCES_COUNT = 5

def text_preprocess(unprepared_text):
    """
    Выполняет предварительную обработку текста, заменяя некоторые
    знаки препинания на точки для улучшения обработки текста суммаризатором.

    Args:
        unprepared_text (str): Исходный текст для обработки.

    Returns:
        str: Текст с заменёнными знаками препинания.
    """
    return unprepared_text.replace(';', '.').replace(':', '.').replace('...', '.')

def sumy_lexrank_weak(unprocessed_text: str) -> str:
    """
    Создает слабую (более длинную) аннотацию текста, используя алгоритм LexRank.

    Args:
        unprocessed_text (str): Текст для суммаризации.

    Returns:
        str: Сгенерированное слабое резюме текста.
    """
    summarizer_lex = LexRankSummarizer()
    parser = PlaintextParser.from_string(unprocessed_text, Tokenizer("russian"))
    summary = summarizer_lex(parser.document, SENTENCES_COUNT * 2)
    return ' '.join([str(sentence) for sentence in summary])

def sumy_lexrank_strong(unprocessed_text: str) -> str:
    """
    Создает сильную (более короткую) аннотацию текста, используя алгоритм LexRank.

    Args:
        unprocessed_text (str): Текст для суммаризации.

    Returns:
        str: Сгенерированное сильное резюме текста.
    """
    summarizer_lex = LexRankSummarizer()
    parser = PlaintextParser.from_string(unprocessed_text, Tokenizer("russian"))
    summary = summarizer_lex(parser.document, SENTENCES_COUNT)
    return ' '.join([str(sentence) for sentence in summary])

if __name__ == '__main__':
    nltk.download('punkt_tab')

    # Получаем путь до текстового файла от пользователя
    file = input('Введите путь до файла (txt): ')

    try:
        # Читаем содержимое файла
        with open(file, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        # Если файл не найден
        print('Такого файла не существует')

    # Запрашиваем у пользователя тип сжатия (уровень суммаризации)
    sum_type = input("Выберите уровень сжатия (strong/weak): ")

    # В зависимости от выбора выполняется соответствующая функция
    if sum_type == "strong":
        RESULT = sumy_lexrank_strong(text)
    elif sum_type == "weak":
        RESULT = sumy_lexrank_weak(text)
    else:
        # Если указан неизвестный уровень сжатия, выводится ошибка
        print(f"Неизвестный тип сжатия: {sum_type}")
        raise SystemExit

    # Вывод результата суммаризации
    print("Результат: ", RESULT)
