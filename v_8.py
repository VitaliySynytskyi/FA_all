import numbers

import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
from time import time
from scipy.optimize import curve_fit
from string import punctuation
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from os import listdir
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import re
import base64
import io
import webbrowser
from pygments import lex
from pygments.lexers import get_lexer_by_name, get_lexer_for_filename, guess_lexer, TextLexer
from pygments.token import Token
from dash.dependencies import Input, Output, State
from pygments.lexers import get_all_lexers


include_comments_state = {}

def tokenize_code_with_pygments(code_content, language=None, filename=None):
    """
    Токенізує код за допомогою Pygments, зберігаючи всі елементи коду та коментарі окремо.
    """
    # Переконаємося, що код закінчується новим рядком
    if not code_content.endswith('\n'):
        code_content += '\n'
    
    try:
        if filename:
            lexer = get_lexer_for_filename(filename)
        elif language:
            lexer = get_lexer_by_name(language)
        else:
            lexer = TextLexer()
    except:
        lexer = TextLexer()
    
    code_tokens = []
    comment_tokens = []
    
    for token_type, token_value in lex(code_content, lexer):
        # Змінюємо перевірку - не пропускаємо токени, які складаються тільки з пробілів
        if token_value == '':
            continue
            
        # Перевіряємо чи це коментар
        if token_type in Token.Comment:
            # Препроцесорні директиви - це НЕ коментарі для C/C++
            if (language in ['c', 'cpp'] and 
                token_type in (Token.Comment.Preproc, Token.Comment.PreprocFile)):
                # Розбиваємо препроцесорні директиви на токени
                sub_tokens = re.findall(r'\w+|[^\w\s]', token_value)
                for sub_token in sub_tokens:
                    if sub_token:  # Не використовуємо strip()
                        code_tokens.append((Token.Comment.Preproc, sub_token))
            else:
                # Справжні коментарі - витягуємо слова
                cleaned = token_value.strip()
                if cleaned:
                    processor = NgrammProcessor(ignore_punctuation=True)
                    processor.preprocess(cleaned)
                    words = processor.get_words()
                    comment_tokens.extend(words)
        else:
            # Всі інші токени - це код
            # Розбиваємо на окремі слова та символи
            sub_tokens = re.findall(r'\w+|[^\w\s]', token_value)
            for sub_token in sub_tokens:
                if sub_token:  # Не використовуємо strip()
                    code_tokens.append((token_type, sub_token))
    
    return code_tokens, comment_tokens


def tokenize_code_with_pygments_without_comments(code_content, language=None, filename=None, include_comments=False):
    """
    Токенізує код за допомогою Pygments, виключаючи коментарі.
    """
    # Переконаємося, що код закінчується новим рядком
    if not code_content.endswith('\n'):
        code_content += '\n'
    
    try:
        if filename:
            lexer = get_lexer_for_filename(filename)
        elif language:
            lexer = get_lexer_by_name(language)
        else:
            lexer = TextLexer()
    except:
        lexer = TextLexer()
    
    code_tokens = []
    
    for token_type, token_value in lex(code_content, lexer):
        # Змінюємо перевірку - не пропускаємо токени
        if token_value == '':
            continue
            
        # Перевіряємо чи це коментар
        if token_type in Token.Comment:
            # Препроцесорні директиви - це НЕ коментарі для C/C++
            if (language in ['c', 'cpp'] and 
                token_type in (Token.Comment.Preproc, Token.Comment.PreprocFile)):
                # Розбиваємо препроцесорні директиви на токени
                sub_tokens = re.findall(r'\w+|[^\w\s]', token_value)
                for sub_token in sub_tokens:
                    if sub_token:  # Не використовуємо strip()
                        code_tokens.append((Token.Comment.Preproc, sub_token))
            # Інші коментарі пропускаємо
            continue
        else:
            # Всі інші токени - це код
            # Розбиваємо на окремі слова та символи
            sub_tokens = re.findall(r'\w+|[^\w\s]', token_value)
            for sub_token in sub_tokens:
                if sub_token:  # Не використовуємо strip()
                    code_tokens.append((token_type, sub_token))
    
    return code_tokens, []
def get_all_code_elements(code_tokens):
    """
    Витягує всі елементи коду з токенів, включаючи ключові слова, оператори, ідентифікатори тощо.
    """
    all_elements = []
    
    for token_type, token_value in code_tokens:
        # Для директив препроцесора та інших токенів
        # просто додаємо значення як є
        if token_value.strip():
            all_elements.append(token_value)
    
    return all_elements

def process_code_improved(file_content, language=None, filename=None, include_comments=True):
    print(f"\n--- PROCESS CODE IMPROVED ---")
    print(f"Language: {language}")
    print(f"Filename: {filename}")
    print(f"Include comments: {include_comments}")
    print(f"File content length: {len(file_content)}")

    print(f"Last 50 chars: {repr(file_content[-50:])}")
    print(f"Last char: {repr(file_content[-1])}")
    
    try:
        if include_comments:
            code_tokens, comment_tokens = tokenize_code_with_pygments(file_content, language, filename)
        else:
            code_tokens, comment_tokens = tokenize_code_with_pygments_without_comments(file_content, language, filename)
        
        print(f"Code tokens: {len(code_tokens)}")
        print(f"Comment tokens: {len(comment_tokens)}")
        
        code_elements = get_all_code_elements(code_tokens)
        print(f"Code elements: {len(code_elements)}")
        
        # Виправлена логіка
        if include_comments and comment_tokens:
            all_elements = code_elements + comment_tokens
        else:
            all_elements = code_elements 
        
        print(f"All elements total: {len(all_elements)}")
        
        return code_elements, comment_tokens, all_elements
        
    except Exception as e:
        print(f"ERROR IN PROCESS CODE IMPROVED: {e}")
        import traceback
        traceback.print_exc()
        raise

def detect_programming_language(filename):
    """Визначає мову програмування за розширенням файлу"""
    extension = filename.split('.')[-1].lower()
    
    language_extensions = {
        'py': 'python',
        'js': 'javascript',
        'java': 'java',
        'c': 'c',
        'cpp': 'cpp',
        'cs': 'csharp',
        'php': 'php',
        'rb': 'ruby',
        'go': 'go',
        'rs': 'rust',
        'swift': 'swift',
        'kt': 'kotlin',
        'ts': 'typescript',
        'html': 'html',
        'css': 'css',
        'sql': 'sql'
    }
    
    return language_extensions.get(extension, 'text')

def process_code(file_content, language, include_comments=True, filename=None):
    print(f"\n*** PROCESS CODE ***")
    print(f"Language: {language}")
    print(f"Include comments: {include_comments}")
    print(f"Filename: {filename}")
    print(f"File content length: {len(file_content)}")
    
    try:
        # Використовуємо нову функцію для обробки коду
        code_elements, comment_elements, all_elements = process_code_improved(
            file_content, language, filename, include_comments
        )
        
        # Вивід для відлагодження
        print("Приклади токенів:", all_elements[:20])
        print("Загальна кількість токенів:", len(all_elements))
        
        # Підрахунок пунктуації
        punctuation = [t for t in all_elements if t in ".,(){}[]<>;:'\"!?+-*/="]
        print("Кількість токенів пунктуації:", len(punctuation))
        print("Приклади пунктуації:", punctuation[:20])
        
        # Повертаємо всі елементи, виключаючи коментарі якщо потрібно
        print(f"Returning {len(all_elements)} elements")
        return all_elements
        
    except Exception as e:
        print(f"ERROR IN PROCESS CODE: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("*** END PROCESS CODE ***\n")
def remove_punctuation_for_words(data):
    # Split the text into words using regular expression
    words = re.findall(r'\b\w+(?:[-\']\w+)*\b', data)

    # Further process the words to handle special characters
    processed_words = []
    for word in words:
        # Handle special characters and dashes within words
        processed_word = re.split(r'[^a-zA-Z0-9\']', word)
        processed_words.extend(processed_word)

    # Filter out empty strings and lowercase each word
    processed_words = [word.lower() for word in processed_words if word]

    return processed_words


def remove_punctuation(data):
    temp = []
    start_time = time()
    print()
    # print(data)
    for i in range(len(data)):
        if data[i] in punctuation:
            continue
        else:
            temp.append(data[i].lower())
    resultt = "".join(temp)
    end_time = time()

    # Calculate the execution time
    execution_time = end_time - start_time

    print("Execution time: {} seconds".format(execution_time))
    return resultt

toast_visible = False
error_visible = False
analyze_visible = False

class Ngram(dict):
    def __init__(self, iterable=None):  # Ініціалізували наш розподіл як новий об'єкт класу, додаємо наявні елементи
        super(Ngram, self).__init__()
        self.fa = {}
        self.counts = {}
        self.sums = {}
        if iterable:
            self.update(iterable)

    def update(self, iterable):  # Оновлюємо розподіл елементами з наявного ітеруємого набору даних
        for item in iterable:
            if item in self:
                self[item] += 1
            else:
                self[item] = 1

    def hist(self):
        plt.bar(self.keys(), self.values())
        plt.show()


def make_dataframe(model, fmin=0):
    filtered_data = list(
        filter(lambda x: sum(value for value in model[x].values() if isinstance(value, int)) >= fmin, model))
    if 'new_ngram' not in filtered_data:
        filtered_data.append("new_ngram")
    data = {"ngram": [],
            "F": np.empty(len(filtered_data), dtype=np.dtype(int))}

    for i, ngram in enumerate(filtered_data):
        data["ngram"].append(ngram)

        if ngram == "new_ngram":
            data['F'][i] = sum(model[ngram].bool)
            continue
        data["F"][i] = len(model[ngram].pos)

    dffff = pd.DataFrame(data=data)
    return dffff


def make_markov_chain(data, order=1):
    global model, L, V
    
    # Зберігаємо оригінальну довжину для масивів bool
    original_L = len(data)  
    L = original_L  # Використовуємо повну довжину
    
    model = dict()
    model['new_ngram'] = Ngram()
    model['new_ngram'].bool = np.zeros(original_L, dtype=np.uint8)
    model['new_ngram'].pos = []
    
    if order > 1:
        # Для n-грамів порядку > 1
        for i in range(L - order + 1):  # +1 для обробки останньої n-грами
            window = tuple(data[i: i + order])
            
            # Перевіряємо, чи це останнє вікно
            is_last_window = (i + order >= L)
            
            if window in model:
                # Додаємо наступний символ, якщо це не останнє вікно
                if not is_last_window:
                    model[window].update([data[i + order]])
                
                model[window].pos.append(i + 1)
                model[window].bool[i] = 1
            else:
                # Створюємо нову n-граму
                if not is_last_window:
                    model[window] = Ngram([data[i + order]])
                else:
                    model[window] = Ngram()  # Порожня n-грама для останнього вікна
                
                model[window].pos = []
                model[window].pos.append(i + 1)
                model[window].bool = np.zeros(original_L, dtype=np.uint8)
                model[window].bool[i] = 1
                model['new_ngram'].bool[i] = 1
                model['new_ngram'].pos.append(i + 1)
    else:
        # Для звичайних символів
        for i in range(L):
            current_char = data[i]
            
            # Перевіряємо, чи це останній символ
            is_last_char = (i == L - 1)
            
            if current_char in model:
                # Додаємо наступний символ, якщо це не останній
                if not is_last_char:
                    model[current_char].update([data[i + 1]])
                
                # Позиція - індекс самого символу для останнього, інакше індекс наступного
                pos_index = i if is_last_char else i + 1
                model[current_char].pos.append(pos_index)
                
                try:
                    model[current_char].bool[i] = 1
                except Exception as e:
                    print(f'Error setting bool for {current_char} at position {i}: {e}')
            else:
                # Створюємо нову n-граму
                if not is_last_char:
                    model[current_char] = Ngram([data[i + 1]])
                else:
                    model[current_char] = Ngram()  # Порожня n-грама для останнього символу
                
                model[current_char].pos = []
                pos_index = i if is_last_char else i + 1
                model[current_char].pos.append(pos_index)
                model[current_char].bool = np.zeros(original_L, dtype=np.uint8)
                model[current_char].bool[i] = 1
                model['new_ngram'].bool[i] = 1
                model['new_ngram'].pos.append(pos_index)
    
    V = len(model)
    return original_L


def calculate_distance(positions, L, option, ngram, min_dist=1):
    if option == "no":
        return nbc(positions, min_dist)
    if option == "ordinary":
        return obc(positions, L, min_dist)
    if option == "periodic":
        return pbc(positions, L, ngram, min_dist)


@jit(nopython=True)
def nbc(positions, min_dist=1):
    number_of_pos = len(positions)
    
    # Для Numba перевірка на порожній масив
    if number_of_pos <= 0:
        # Створюємо порожній масив через np.zeros і зменшуємо його розмір до 0
        return np.zeros(0, dtype=np.uint32)
    
    # Перевірка на один елемент  
    if number_of_pos == 1:
        return np.zeros(0, dtype=np.uint32)  # Повертаємо пустий масив
        
    dt = np.empty(number_of_pos - 1, dtype=np.uint32)
    for i in range(number_of_pos - 1):
        dt[i] = positions[i + 1] - positions[i]
        if min_dist == 0:
            dt[i] = dt[i] - 1
    return dt

@jit(nopython=True)
def obc(positions, L, min_dist=1):
    number_of_pos = len(positions)
    
    # Перевірка на порожній масив
    if number_of_pos <= 0:
        return np.zeros(0, dtype=np.uint32)
    
    # Якщо тільки один елемент
    if number_of_pos == 1:
        if min_dist == 0 and positions[0] > 0:
            dt0 = positions[0] - 1
        else:
            dt0 = positions[0]
            
        if min_dist == 0 and L - positions[0] > 0:
            dt1 = L - positions[0] - 1
        else:
            dt1 = L - positions[0]
            
        return np.array([dt0, dt1], dtype=np.uint32)
    
    dt = np.empty(number_of_pos + 1, dtype=np.uint32)
    dt[0] = positions[0]
    if min_dist == 0 and dt[0] > 0:
        dt[0] = dt[0] - 1
    for i in range(number_of_pos - 1):
        dt[i + 1] = positions[i + 1] - positions[i]
        if min_dist == 0:
            dt[i + 1] = dt[i + 1] - 1
    dt[-1] = L - positions[-1]
    if min_dist == 0 and dt[-1] > 0:
        dt[-1] = dt[-1] - 1
    return dt

@jit(nopython=True)
def pbc(positions, L, test, min_dist=1):
    number_of_pos = len(positions)
    
    # Перевірка на порожній масив
    if number_of_pos <= 0:
        return np.zeros(0, dtype=np.uint32)
    
    # Перевірка на один елемент  
    if number_of_pos == 1:
        distance = L
        if min_dist == 0 and distance > 0:
            distance = distance - 1
        return np.array([distance], dtype=np.uint32)
    
    dt = np.zeros(number_of_pos, dtype=np.uint32)
    for i in range(number_of_pos - 1):
        dt[i] = positions[i + 1] - positions[i]
        if min_dist == 0:
            dt[i] = dt[i] - 1
    dt[-1] = L - positions[-1] + positions[0]
    if min_dist == 0 and dt[-1] > 0:
        dt[-1] = dt[-1] - 1
    return dt

@jit(nopython=True)
def s(window):
    suma = 0
    for i in range(len(window)):
        suma += window[i]
    return suma


@njit(fastmath=True)
def mse(x):
    t = x.mean()
    st = np.mean(x ** 2)
    return np.sqrt(st - (t ** 2))


@jit(nopython=True, fastmath=True)
def R(x):
    if len(x) == 0:  # Додайте перевірку на порожній масив
        return 0.0
    if len(x) == 1:
        return 0.0
    t = np.mean(x)
    ts = np.std(x)
    if t == 0:  # Також перевірте ділення на нуль
        return 0.0
    return ts / t


@njit(fastmath=True)
def calc_non_overlapping_shift(k, min_window, window_expansion):
    """
    Розраховує зміщення для режиму non-overlapping
    k - номер кроку (починаючи з 1)
    """
    # Numba не працює з None значеннями, тому перевірка робиться в make_windows
    if k == 1:
        return min_window
    else:
        return min_window + (k-1) * window_expansion
    
# Додайте цю функцію для отримання всіх мов Pygments
def get_all_pygments_languages():
    
    languages = [{"label": "Шукати мову..", "value": "auto"}]
    
    for lexer_info in get_all_lexers():
        name = lexer_info[0]
        aliases = lexer_info[1]
        if aliases:
            # Беремо перший псевдонім як значення
            languages.append({"label": name, "value": aliases[0]})
    
    # Сортуємо за назвою (пропускаючи "Автоматично")
    languages[1:] = sorted(languages[1:], key=lambda x: x["label"])
    
    return languages

@njit(fastmath=True)
def make_windows(x, wi, l, wsh, overlap_mode="overlapping", min_window=None, window_expansion=None):
    sums = []
    if overlap_mode == "overlapping":
        # Стандартний режим з фіксованим зміщенням
        for i in range(0, l - wi, wsh):
            sums.append(np.sum(x[i:i + wi]))
    else:  # non-overlapping режим
        # Перевіряємо значення параметрів і встановлюємо значення за замовчуванням якщо None
        if min_window is None:
            min_window = wsh
        if window_expansion is None:
            window_expansion = wsh
            
        k = 1
        i = 0
        while i < l - wi:
            sums.append(np.sum(x[i:i + wi]))
            # Розраховуємо зміщення для наступного вікна
            shift = calc_non_overlapping_shift(k, min_window, window_expansion)
            i += shift
            k += 1
    
    return np.array(sums)


@njit(fastmath=True)
def calc_sum(x):
    sums = np.empty(len(x))
    for i, w in enumerate(x):
        sums[i] = np.sum(w)
    return sums


@jit(nopython=True, fastmath=True)
def fit(x, a, b):
    return a * (x ** b)


def prepere_data(data, n, split, file_type='regular', language=None, filename=None, include_comments=True):
    print(f"\n=== PREPERE DATA ===")
    print(f"n: {n}")
    print(f"split: {split}")
    print(f"file_type: {file_type}")
    print(f"include_comments: {include_comments}")  # Додано для відлагодження
    global L
    
    if n is None:
        print("N IS NONE - RETURNING NO UPDATE")
        return dash.no_update
    
    if file_type == 'code' and language:
        print(f"Processing code with include_comments={include_comments}")
        code_tokens = process_code(data, language, include_comments, filename)
        print(f"Got {len(code_tokens)} code tokens")
        
        # Якщо n > 1, створюємо n-грами
        if n > 1:
            temp_data = []
            L = len(code_tokens)
            for i in range(L - n + 1):
                window = tuple(code_tokens[i:i + n])
                temp_data.append(window)
            print(f"Created {len(temp_data)} n-grams")
            return temp_data
        else:
            L = len(code_tokens)
            print(f"Returning code tokens, L = {L}")
            return code_tokens
    
    # Звичайний текст - стандартна обробка
    temp_data = []
    if n == 1:
        if split == "word":
            temp = []
            data = re.sub(r'\n+', '\n', data)
            data = re.sub(r'\n\s\s', '\n', data)
            data = re.sub(r'﻿', '', data)
            data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor()
            # обробка тексту
            processor.preprocess(data)
            # Отримання слів у тексті
            data = processor.get_words()

            for i in data:
                temp.append(i)
            L = len(temp)
            return temp
        if split == 'letter':
            data = remove_punctuation(data)
            for i in data:
                for j in i:
                    if is_valid_letter(j):
                        continue
                    temp_data.append(j)
            data = temp_data
            L = len(data)
            return data
        if split == 'symbol':
            data = re.sub(r'\n+', '\n', data)
            data = re.sub(r'\n\s\s', '\n', data)
            data = re.sub(r'﻿', '', data)
            for i in data:
                for j in i:
                    if j == " ":
                        temp_data.append("space")
                        continue
                    elif i == "\n":
                        temp_data.append("space")
                        continue
                    elif i == "\ufeff":
                        temp_data.append("space")
                        continue
                    j = j.lower()
                    temp_data.append(j)
            data = temp_data
            L = len(data)
            return data
    if n > 1:
        if split == "word":
            data = re.sub(r'\n+', '\n', data)
            data = re.sub(r'\n\s\s', '\n', data)
            data = re.sub(r'﻿', '', data)
            data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor()
            # обробка тексту
            processor.preprocess(data)
            # Отримання слів у тексті
            data = processor.get_words()
            L = len(data)
            # L = len(data) - n
            for i in range(L - n + 1):
                window = tuple(data[i: i + n])
                temp_data.append(window)
            return temp_data
        if split == "letter":
            data = remove_punctuation(data.split())
            data = remove_empty_strings(data)
            for i in data:
                for j in i:
                    if is_valid_letter(j):
                        continue
                    temp_data.append(j)
            L = len(temp_data)
            data = temp_data
            temp_data = []
            for i in range(L - n + 1):
                window = tuple(data[i: i + n])
                temp_data.append(window)
            return temp_data
        if split == 'symbol':
            temp_data = []
            data = re.sub(r'\n+', '\n', data)
            data = re.sub(r'\n\s\s', '\n', data)
            data = re.sub(r'﻿', '', data)
            for i in data:
                for j in i:
                    if j == " ":
                        temp_data.append("space")
                        continue
                    elif i == "\n":
                        temp_data.append("space")
                        continue
                    elif i == "\ufeff":
                        temp_data.append("space")
                        continue
                    j = j.lower()
                    temp_data.append(j)
            data = temp_data
            temp_data = []
            L = len(data)
            # L = len(data) - n
            for i in range(L - n + 1):
                window = tuple(data[i:i + n])
                temp_data.append(window)
            return temp_data
    
    L = 0
    return []
# @jit(nopython=True)
def dfa(data, args, overlap_mode="overlapping", min_window=None, window_expansion=None):
    wi, wh, l = args
    
    if overlap_mode == "overlapping":
        # Стандартний режим з фіксованим зміщенням
        count = np.empty(len(range(0, l - wi, wh)), dtype=np.uint8)
        for index, i in enumerate(range(0, l - wi, wh)):
            temp_v = []
            x = []
            for ngram in data[i:i + wi]:
                if ngram in temp_v:
                    x.append(0)
                else:
                    temp_v.append(ngram)
                    x.append(1)
            count[index] = s(np.array(x, dtype=np.uint8))
    else:
        # Non-overlapping режим
        if min_window is None:
            min_window = wh
        if window_expansion is None:
            window_expansion = wh
            
        # Оцінюємо кількість вікон
        k = 1
        i = 0
        window_positions = []
        while i < l - wi:
            window_positions.append(i)
            shift = calc_non_overlapping_shift(k, min_window, window_expansion)
            i += shift
            k += 1
            
        count = np.empty(len(window_positions), dtype=np.uint8)
        for index, i in enumerate(window_positions):
            temp_v = []
            x = []
            for ngram in data[i:i + wi]:
                if ngram in temp_v:
                    x.append(0)
                else:
                    temp_v.append(ngram)
                    x.append(1)
            count[index] = s(np.array(x, dtype=np.uint8))
            
    return count, mse(count)


class newNgram():
    def __init__(self, data, wh, l):
        self.data = data
        self.count = {}
        self.dfa = {}
        self.wh, self.l = wh, l

    def func(self, w, overlap_mode="overlapping", min_window=None, window_expansion=None):
        if overlap_mode == "non-overlapping" and (min_window is None or window_expansion is None):
            min_window = self.wh
            window_expansion = self.wh
        self.count[w], self.dfa[w] = dfa(self.data, (w, self.wh, self.l), overlap_mode, min_window, window_expansion)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dictionary to store uploaded files
uploaded_files = {}
# Dictionary to store file lengths with structure: {filename: {'word': length, 'symbol': length, 'letter': length}}
file_lengths = {}
file_types = {}
batch_results = []
model = dict()
V = 0
L = 0
data = []
df = None
g = None
new_ngram = None
ngram = None


# Removing the corpuses list since we're using file upload now
# corpuses = listdir("corpus/")
colors = {
    "background": "#a1a1a1",
    "text": "#a1a1a1"}

import dash_bootstrap_components as dbc

layout2 = html.Div()

# Після оголошення кольорів та інших UI елементів, але перед layout1
text_type_modal = dbc.Modal(
    [
        dbc.ModalHeader("Виберіть тип тексту"),
        dbc.ModalBody([
            dbc.RadioItems(
                id="text-type-selector",
                options=[
                    {"label": "Звичайний текст", "value": "regular"},
                    {"label": "Програмний код", "value": "code"}
                ],
                value="regular",
                inline=True
            ),
            html.Div([
                html.Label("Мова програмування (якщо автоматичне визначення неправильне):", 
                          style={"marginTop": "15px", "marginBottom": "5px"}),
                dcc.Dropdown(
                id="language-selector",
                options=get_all_pygments_languages(),  # Отримуємо всі мови
                value="auto",
                searchable=True,  # Включаємо пошук
                placeholder="Шукати мову...",
                style={
                    "width": "100%",
                    "marginBottom": "10px"
                },
                optionHeight=35,  # Висота кожної опції
                clearable=False   # Заборонити очищення вибору
            )
            ], id="language-selector-container", style={"display": "none"})
        ]),
        dbc.ModalFooter(
            dbc.Button("Підтвердити", id="text-type-confirm", className="ml-auto")
        ),
    ],
    id="text-type-modal",
    centered=True,
)

layout1 = html.Div([
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Configuration:", style={"background-color": "#e9f5fe", "fontWeight": "bold"}),
                        dbc.CardBody(
                            [
                                # FILE SECTION
                                html.Div([
                                    html.H6("File Selection", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                
                                    html.Label("Upload file:"),
                                    html.Div(
                                        [
                                            # Replace dropdown with Upload component
                                            dcc.Upload(
                                                id='upload-data',
                                                children=html.Div([
                                                    'Drag and Drop or ',
                                                    html.A('Select Files', style={'fontWeight': 'bold', 'color': '#007bff'})
                                                ]),
                                                style={
                                                    'width': '100%',
                                                    'height': '60px',
                                                    'lineHeight': '60px',
                                                    'borderWidth': '1px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center',
                                                    'margin': '10px 0',
                                                    'background': '#fafafa',
                                                    'borderColor': '#007bff'
                                                },
                                                multiple=True
                                            ),
                                            html.Div(id='upload-status'),
                                            # Add dropdown for selecting files
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Select file"),
                                                    dcc.Dropdown(
                                                        id='file-selector',
                                                        options=[],
                                                        placeholder="Select a file to analyze",
                                                        style={"minWidth": "250px", "maxWidth": "100%", "whiteSpace": "nowrap", "textOverflow": "ellipsis"}
                                                    )
                                                ], 
                                                size="md", 
                                                className="mb-3",
                                                style={"marginBottom": "10px"}
                                            ),
                                        ]),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),
                                # Додайте цей код десь у layout1, наприклад після блоку вибору файлу
html.Div(id="file-type-info", children="", style={"display": "none", "margin": "10px 0", "padding": "5px", "background-color": "#f8f9fa", "borderRadius": "5px"}),
                                # ANALYSIS PARAMETERS SECTION
                                html.Div([
                                    html.H6("Analysis Parameters", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Size of ngram"),
                                            dbc.Input(id="n_size", type="number", value=1, style={"font-weight": "bold"})
                                        ], 
                                        size="md", 
                                        className="mb-2"
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Split by"),
                                            dbc.Select(
                                                id="split",
                                                options=[
                                                    {"label": "symbol", "value": "symbol"},
                                                    {"label": "word", "value": "word"},
                                                    {"label": "letter", "value": "letter"}
                                                ],
                                                value="word",
                                                style={"font-weight": "bold"}
                                            )
                                        ], 
                                        size="md", 
                                        className="mb-2"
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="condition",
                                                options=[
                                                    {"label": "no", "value": "no"},
                                                    {"label": "periodic", "value": "periodic"},
                                                    {"label": "ordinary", "value": "ordinary"}
                                                ],
                                                value="no",
                                                style={"font-weight": "bold"}
                                            ),
                                    dbc.InputGroupText("Boundary Condition:")  
                                ], 
                                size="md", 
                                className="mb-2"
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="min_dist_option",
                                                options=[
                                                    {"label": "min=1", "value": 1},
                                                    {"label": "min=0", "value": 0}
                                                ],
                                                value=1,
                                                style={"font-weight": "bold"}
                                            ),
                                    dbc.InputGroupText("Min Distance:")
                                ], 
                                size="md", 
                                className="mb-2"
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("filter"),
                                            dbc.Input(id="f_min", type="number", value=0, style={"font-weight": "bold"})
                                        ],
                                        className="mb-3"
                                    ),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),
                                # Додайте цей код після вибору типу файлу, до блоку ANALYSIS PARAMETERS SECTION
                                html.Div([
                                    dbc.Checklist(
                                        options=[
                                            {"label": "Включати коментарі при аналізі коду", "value": True}
                                        ],
                                        value=[True],
                                        id="include-comments-switch",
                                        switch=True,
                                        style={"margin": "10px 0"}
                                    ),
                                ], id="comments-switch-container", style={"display": "none", "margin": "10px 0", "padding": "5px", "background-color": "#f8f9fa", "borderRadius": "5px"}),
                                # WINDOW SETTINGS SECTION
                                html.Div([
                                    html.H6("Window Settings", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="overlap_mode",
                                                options=[
                                                    {"label": "overlapping", "value": "overlapping"},
                                                    {"label": "non-overlapping", "value": "non-overlapping"}
                                                ],
                                                value="overlapping"
                                            ),
                                            dbc.InputGroupText("Window Mode"),
                                        ], size="md", className="mb-2"
                                    ),
                                    
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="def",
                                                options=[
                                                    {"label": "static", "value": "static"},
                                                    {"label": "dynamic", "value": "dynamic"}
                                                ],
                                                value="static"
                                            ),
                                            dbc.InputGroupText("Definition", style={"background-color": "#e9f5fe"}),
                                            dbc.Tooltip(
                                                "Static: Manual window parameters. Dynamic: Auto-calculated based on data size",
                                                target="def",
                                            ),
                                        ], size="md", className="mb-3"
                                    ),

                                    html.Div([
                                        html.Small([
                                            html.Span("W = Min Window", style={"fontWeight": "bold"}), " | ",
                                            html.Span("WH = Window Shift", style={"fontWeight": "bold"}), " | ",
                                            html.Span("WE = Window Expansion", style={"fontWeight": "bold"}), " | ",
                                            html.Span("WM = Max Window", style={"fontWeight": "bold"})
                                        ], className="text-muted mb-2 d-block text-center"),
                                    ], style={"background": "#f0f8ff", "padding": "6px", "borderRadius": "5px", "marginBottom": "10px"}),

                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(html.Span(["Min", html.Br(), "Window"], style={"lineHeight": "1.2", "textAlign": "center"}), 
                                                                 style={"width": "90px", "background-color": "#e9f5fe"}),
                                            dbc.Input(id="w", type="number", style={"font-weight": "bold"}),
                                            dbc.Tooltip(
                                                "Minimum window size (W) - Starting window dimension",
                                                target="w",
                                            ),
                                        ], size="md", className="mb-2"
                                    ),

                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(html.Span(["Window", html.Br(), "Shift"], style={"lineHeight": "1.2", "textAlign": "center"}), 
                                                                 style={"width": "90px", "background-color": "#e9f5fe"}),
                                            dbc.Input(id="wh", type="number", style={"font-weight": "bold"}),
                                            dbc.Tooltip(
                                                "Window shift (WH) - How far to move window when overlapping",
                                                target="wh",
                                            ),
                                        ], size="md", className="mb-2"
                                    ),

                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(html.Span(["Window", html.Br(), "Expansion"], style={"lineHeight": "1.2", "textAlign": "center"}), 
                                                                 style={"width": "90px", "background-color": "#e9f5fe"}),
                                            dbc.Input(id="we", type="number", style={"font-weight": "bold"}),
                                            dbc.Tooltip(
                                                "Window expansion (WE) - How much window size increases per step",
                                                target="we",
                                            ),
                                        ], size="md", className="mb-2"
                                    ),

                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(html.Span(["Max", html.Br(), "Window"], style={"lineHeight": "1.2", "textAlign": "center"}), 
                                                                 style={"width": "90px", "background-color": "#e9f5fe"}),
                                            dbc.Input(id="wm", type="number", style={"font-weight": "bold"}),
                                            dbc.Tooltip(
                                                "Maximum window size (WM) - Largest window dimension",
                                                target="wm",
                                            ),
                                        ], size="md", className="mb-2"
                                    ),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),

                                # BATCH PROCESSING SECTION
                                html.Div([
                                    html.H6("Batch Processing", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Lmin: Fmin1"),
                                            dbc.Input(id="fmin1", type="number", value=1, style={"font-weight": "bold"})
                                        ],
                                        style={'marginBottom': '5px'}
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Lmax: Fmin2"),
                                            dbc.Input(id="fmin2", type="number", value=5, style={"font-weight": "bold"})
                                        ],
                                        style={'marginBottom': '5px'}
                                    ),
                                    # Add batch window settings options
                                    dbc.Collapse(
                                        [
                                            html.H6("Batch Window Settings", style={'marginTop': '10px', 'fontSize': '14px'}),
                                            dbc.InputGroup(
                                                [
                                                    dbc.Select(
                                                        id="batch_window_mode",
                                                        options=[
                                                            {"label": "Use UI settings", "value": "ui"},
                                                            {"label": "Auto per file", "value": "auto"},
                                                        ],
                                                        value="auto"
                                                    ),
                                                    dbc.InputGroupText("Window Mode")
                                                ],
                                                style={'marginBottom': '5px'}
                                            ),
                                        ],
                                        id="batch_window_controls",
                                        is_open=True
                                    ),
                                    dbc.Button("Process All Files", id="batch_process", color="success", 
                                              className="w-100", 
                                              style={'marginBottom': '10px', "fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),
                                
                                # ACTION BUTTONS SECTION
                                html.Div([
                                    html.H6("Actions", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    
                                    dbc.Button("Analyze", id="chain_button", color="primary", 
                                              className="w-100 mb-2", 
                                              style={"fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, 
                                              disabled=analyze_visible),

                                    dbc.Button("Save data", id="save", color="danger", 
                                              className="w-100",
                                              style={"fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                                    html.Div(id="temp_seve",
                                             children=[]
                                             ),
                                    html.Div(id="temp_seve_batch",
                                             children=[]
                                             )
                                ]),
                                html.Div(id="alert", children=[])
                                # html.H6("Boundary Condition:"),
                                # dcc.RadioItems(id='condition',options=[{"label":"no","value":"no"},{"label":"periodic","value":"periodic"},{"label":"ordinary","value":"ordinary"}],value="words"),
                            ]

                        ),

                    ], color="light", style={"margin-left": "0px", "margin-top": "10px", }
                ),
                width={"size": 3, "offset": 0}
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Tabs(
                                    [
                                        dbc.Tab(label="DataTable", tab_id="data_table", label_style={"font-weight": "bold"}),
                                        dbc.Tab(label="MarkovChain", tab_id="markov_chain", label_style={"font-weight": "bold"})
                                    ],
                                    id="dataframe",
                                    active_tab="data_table",
                                    card=True
                                )

                            ),
                            dbc.CardBody(
                                [
                                    # here table name
                                    html.Div(id="box_tab",
                                             style={"display": "none", "height": "400px", "minHeight": "400px"},
                                             children=[dbc.Spinner(dash_table.DataTable(
                                                 id="table",
                                                 columns=[{"name": i, "id": i} for i in
                                                          ['rank', "ngram", "F", "R", "a", "γ", "goodness"]],
                                                 style_data={'whiteSpace': 'auto', 'height': 'auto'},
                                                 editable=False,
                                                 filter_action="native",
                                                 sort_action="native",
                                                 page_size=50,
                                                 fixed_rows={'headers': True},
                                                 fixed_columns={'headers': True},
                                                 style_cell={'whiteSpace': 'normal',
                                                             'height': 'auto',
                                                             "widht": "auto",
                                                             'textAlign': 'right',
                                                             "fontSize": 15,
                                                             "font-family": "sans-serif"},
                                                 # 'minWidth': 40, 'width': 95, 'maxWidth': 95},
                                                 style_table={"height": "400px", "minWidth": "500px",
                                                              'overflowY': 'auto', "overflowX": "none",
                                                              "minHeight": "400px"},
                                                 style_header={
                                                     'backgroundColor': '#e9f5fe',
                                                     'fontWeight': 'bold',
                                                     'textAlign': 'center'
                                                 },
                                                 style_data_conditional=[
                                                     {
                                                         'if': {'row_index': 'odd'},
                                                         'backgroundColor': '#f9f9f9'
                                                     },
                                                     {
                                                         'if': {'state': 'selected'},
                                                         'backgroundColor': '#deeaff',
                                                         'border': '1px solid #aaa'
                                                     }
                                                 ]
                                             ))]),
                                    html.Div(id="box_chain",
                                             style={"display": "none"},
                                             children=[dbc.Spinner(dcc.Graph(id="chain", style={"height": "400px"}))]),

                                    dbc.CardHeader("Characteristics", style={"padding": "5px 20px", "background-color": "#f0f8ff", "font-weight": "bold"}),
                                    # here add chars
                                    dbc.CardBody(
                                        dbc.Row([
                                            # NOTE додала вивід 8-ми значень з екселю а також кнопку для копіювання всього
                                            dbc.Col([
                                                html.Div(["Length: "], id="l", style={"whiteSpace": "nowrap", "width": "100%", "overflow": "hidden", "textOverflow": "ellipsis", "fontWeight": "bold", "padding": "3px"}),
                                                html.Div(["Vocabulary: "], id="v", style={"fontWeight": "bold", "padding": "3px"}),
                                                html.Div(["Time: "], id="t", style={"fontWeight": "bold", "padding": "3px"})

                                            ], width={"size": 5}),
                                            dbc.Col([
                                                html.Div([""], id="new_output1", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output2", n_clicks=0, style={"padding": "3px"}),
                                            ], width={"size": 2}),
                                            dbc.Col([
                                                html.Div([""], id="new_output3", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output4", n_clicks=0, style={"padding": "3px"}),
                                            ], width={"size": 2}),
                                            dbc.Col([
                                                html.Div([""], id="new_output5", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output6", n_clicks=0, style={"padding": "3px"}),
                                            ], width={"size": 2}),
                                            dbc.Col([
                                                html.Div([""], id="new_output7", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output8", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="copy_all", n_clicks=0, style={"fontWeight": "bold", "color": "#007bff", "cursor": "pointer", "textDecoration": "underline", "padding": "3px"})
                                            ], width={"size": 1}),
                                        ])
                                    ),
                                    # Add batch results table
                                    html.Div([
                                        html.H5("Batch Processing Results", style={'marginTop': '20px'}),
                                        dbc.Spinner(dash_table.DataTable(
                                            id="batch_table",
                                            columns=[
                                                {"name": "No.", "id": "no"},
                                                {"name": "Filename", "id": "filename"},
                                                {"name": "F_min", "id": "f_min"},
                                                {"name": "Length (L)", "id": "length"},
                                                {"name": "Vocabulary (V)", "id": "vocabulary"},
                                                {"name": "Time (s)", "id": "time"},
                                                {"name": "R_avg", "id": "r_avg"},
                                                {"name": "dR", "id": "dr"},
                                                {"name": "Rw_avg", "id": "rw_avg"},
                                                {"name": "dRw", "id": "drw"},
                                                {"name": "γ_avg", "id": "g_avg"},
                                                {"name": "dγ", "id": "dg"},
                                                {"name": "γw_avg", "id": "gw_avg"},
                                                {"name": "dγw", "id": "dgw"}
                                            ],
                                            style_data={'whiteSpace': 'normal', 'height': 'auto'},
                                            style_cell={'textAlign': 'center'},
                                            style_header={'fontWeight': 'bold', 'backgroundColor': '#e9f5fe'},
                                            style_table={"overflowX": "auto"},
                                            style_data_conditional=[
                                                {
                                                    'if': {'row_index': 'odd'},
                                                    'backgroundColor': '#f9f9f9'
                                                },
                                                {
                                                    'if': {'row_index': -1},
                                                    'fontWeight': 'bold',
                                                    'backgroundColor': 'lightyellow'
                                                },
                                                {
                                                    'if': {'row_index': -2},
                                                    'fontWeight': 'bold',
                                                    'backgroundColor': 'lightblue'
                                                }
                                            ]
                                        )),
                                        dbc.Button("Save Batch Results", id="save_batch", color="info", 
                                                  className="mt-2", 
                                                  style={'marginTop': '10px', "fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                                    ], id="batch_results_container", style={"display": "none"})
                                ]
                            )
                        ], style={"padding": "0", "margin-right": "0px", "margin-top": "10px", "height": "auto", "minHeight": "650px"}),
                ],
                width={"size": 9, "padding": 0}
            ),
        ]
    ),
    dbc.Row([
        dbc.Col(
            width={"size": 6, "offset": 0},
            children=[
                dbc.Card(
                    [
                        dbc.CardHeader(
                            dbc.Tabs(
                                [
                                    dbc.Tab(label="distribution", tab_id="tab1", label_style={"font-weight": "bold"}),
                                ],
                                id='card-tabs1',
                                active_tab="tab1",
                                card=True
                            )
                        ),
                        dbc.CardBody([
                            dcc.Graph(id="graphs", config={'displayModeBar': True, 'displaylogo': False})

                        ], style={"background-color": "#fcfcfc"})
                    ], style={"height": "100%", "widht": "100%", "margin-right": "0%", "margin-top": "10px",
                              "margin-left": "0%"}
                )
            ]),
        dbc.Col(
            width={"size": 6},
            children=[

                dbc.Card(
                    [
                        dbc.CardHeader(
                            dbc.Tabs(
                                [
                                    dbc.Tab(label="flunctuacion", tab_id="tab2", label_style={"font-weight": "bold"}),
                                    dbc.Tab(label="alpha/R", tab_id="tab3", label_style={"font-weight": "bold"})
                                ],
                                id='card-tabs',
                                active_tab="tab2",
                                card=True
                            )
                        ),
                        dbc.CardBody([
                            dcc.RadioItems(
                                id="scale",
                                options=[
                                    {"label": "linear", "value": "linear"},
                                    {"label": "log", "value": "log"}
                                ],
                                value="linear",
                                labelStyle={"marginRight": "15px", "fontWeight": "bold"},
                                inputStyle={"marginRight": "5px"},
                                style={"marginBottom": "10px", "backgroundColor": "#f8f9fa", "padding": "8px", "borderRadius": "5px"}
                            ),
                            dcc.Graph(id="fa", config={'displayModeBar': True, 'displaylogo': False})

                        ], style={"background-color": "#fcfcfc"})

                    ], style={"height": "100%", "widht": "100%", "padding": "0", "margin-right": "0%",
                              "margin-top": "10px", "margin-left": "0%"}
                )

            ]
        )
    ]

    ),
    dbc.Row(
        children=[
            html.Br(),
            html.Br()
        ]
    ),
    dcc.Store(id='file-language-store', storage_type='memory'),
    dcc.Store(id='temp-state-holder', storage_type='memory'),
    dcc.Store(id='analysis-state', storage_type='memory'),
    # dcc.Store(id='dummy-output'),
    dcc.Store(id='stored-data'),
    html.Div(id='output-message'),
    dbc.Toast(
        id="click-toast",
        header="Attention",
        icon="danger",
        is_open=error_visible,
        dismissable=True,
        duration=6000,
        children="Length has not been calculated yet!",
        style={"position": "fixed", "top": "40%", "right": "40%", "width": 500, "zIndex": 9999}
    )
])

app.layout = html.Div([
    layout1,
    text_type_modal
])
df = None
g = None
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# NOTE клас із С# для обробки слів
class NgrammProcessor:
    def __init__(self, ignore_punctuation: bool = True):
        self.ignore_punctuation = ignore_punctuation
        self.words = []

    def preprocess(self, text: str):
        # Remove punctuation if needed
        if self.ignore_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        mixed_array = text.split()
        real_strings = [item for item in mixed_array if isinstance(item, str) and not is_number(item)]
        self.words = real_strings

    def get_words(self, remove_empty_entries: bool = False) -> list:
        words = self.words
        if remove_empty_entries:
            words = [word for word in words if word]
        words = [word.lower() for word in words]
        return words


def is_valid_letter(char):
    invalid_characters = [' ', '\n', '\ufeff', '°', '"', '„', '–']
    if is_number(char) or char in invalid_characters:
        return True
    else:
        return False


length_updated = False


@app.callback(
    [Output('upload-status', 'children'),
     Output('file-selector', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('n_size', 'value')]
)
def update_upload_status(contents, filenames, n_size):
    global uploaded_files, file_lengths
    
    if contents is None:
        # Return current options for file selector
        options = [{'label': filename, 'value': filename, 'title': filename} for filename in list(uploaded_files.keys())]
        return html.Div(["No new files uploaded"]), options
    
    # Counters for summary
    success_count = 0
    error_count = 0
    
    # Process each uploaded file
    for i, (content, filename) in enumerate(zip(contents, filenames)):
        try:
            # Parse the uploaded file
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            
            # Store the decoded content
            try:
                # Try reading as string
                file_content = decoded.decode('utf-8')
                uploaded_files[filename] = file_content

                print(f"\n=== FILE UPLOAD DEBUG ===")
                print(f"Filename: {filename}")
                print(f"Content length: {len(file_content)}")
                print(f"Last 50 chars: {repr(file_content[-50:])}")
                print(f"Last char: {repr(file_content[-1])}")
                print("========================\n")
                
                # Initialize length dictionary for this file
                file_lengths[filename] = {}
                
                # Calculate and store word length
                text_word = re.sub(r'\n+', '\n', file_content)
                text_word = re.sub(r'\n\s\s', '\n', text_word)
                text_word = re.sub(r'﻿', '', text_word)
                text_word = re.sub(r'--', ' -', text_word)
                processor = NgrammProcessor()
                processor.preprocess(text_word)
                words = processor.get_words()
                file_lengths[filename]['word'] = len(words)
                
                # Calculate and store symbol length
                text_symbol = re.sub(r'	', '', file_content)
                text_symbol = re.sub(r'\n+', '\n', text_symbol)
                text_symbol = re.sub(r'\n\s\s', '\n', text_symbol)
                text_symbol = re.sub(r'﻿', '', text_symbol)
                file_lengths[filename]['symbol'] = len(text_symbol)
                
                # Calculate and store letter length
                text_letter = remove_punctuation(file_content)
                file_lengths[filename]['letter'] = len(text_letter)
                
                # Print detailed info to console instead of adding to upload_status
                print(f"✓ Uploaded: {filename}")
                print(f"  Words: {file_lengths[filename]['word']} | Symbols: {file_lengths[filename]['symbol']} | Letters: {file_lengths[filename]['letter']}")
                
                success_count += 1
            except UnicodeDecodeError:
                print(f"✗ Error: {filename} is not a valid text file")
                error_count += 1
        except Exception as e:
             print(f"ERROR IN PREPERE DATA: {e}")
             import traceback
             traceback.print_exc()
             raise
        print("=== END PREPERE DATA ===\n")
    # Create summary message for display
    summary_message = html.Div([
        html.H5(f"Upload Summary:"),
        html.P(f"Successfully uploaded: {success_count} file(s)", style={'color': 'green'}),
        html.P(f"Files with errors: {error_count}", style={'color': 'red' if error_count > 0 else 'green'})
    ])
    
    # Create options for file selector dropdown, adding title attribute for tooltip
    options = [{'label': filename, 'value': filename, 'title': filename} for filename in list(uploaded_files.keys())]
    
    return summary_message, options


# Add callback to handle file selection
@app.callback(
    [Output('temp-state-holder', 'data'),
     Output('l', 'children'), 
     Output('w', 'value'),
     Output('wh', 'value'),
     Output('we', 'value'),
     Output('wm', 'value'),
     Output('analysis-state', 'data')],
    [Input('include-comments-switch', 'value'),
     Input('file-selector', 'value'),
     Input('split', 'value'),
     Input('file-language-store', 'data')],
    [State('n_size', 'value'),
     State('def', 'value')])
def unified_update_callback(include_comments, selected_filename, split, file_info, n_size, definition):
    global L, data, length_updated, include_comments_state, model, V, file_lengths, file_types
    
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Перевірка на валідність даних
    if selected_filename is None or selected_filename not in uploaded_files:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    file_content = uploaded_files[selected_filename]
    
    # Обробка зміни стану коментарів
    if trigger_id == 'include-comments-switch':
        if not file_info or not isinstance(file_info, dict) or file_info.get('type') != 'code':
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        print(f"=== UPDATING COMMENT STATE ===")
        print(f"Include comments changed to: {include_comments}")
        
        # Перетворюємо в boolean
        if isinstance(include_comments, list):
            include_comments_bool = True if True in include_comments else False
        else:
            include_comments_bool = bool(include_comments)
        
        # Зберігаємо стан для цього файлу
        include_comments_state[selected_filename] = include_comments_bool
        
        # Очищаємо глобальні змінні
        model = dict()
        V = 0
        
        # Перераховуємо дані з новими параметрами
        data = prepere_data(file_content, n_size, split, file_type='code', 
                           language=file_info.get('language'), 
                           filename=selected_filename, 
                           include_comments=include_comments_bool)
        
        # Оновлюємо довжину
        L = len(data) if data else 0
        print(f"Recalculated L = {L}")
        
        # Обчислюємо параметри вікна
        if L < 20:
            wm = min(10, L)
            w = min(3, L)
            wh = 1
            we = 1
        else:
            wm = int(L / 20)
            w = int(wm / 20)
            wh = w
            we = w
        
        # Забезпечуємо мінімальні значення
        wm = max(10, wm)
        w = max(5, w)
        wh = max(1, wh)
        we = max(1, we)
        
        length_updated = True
        
        # Формуємо рядок довжини
        language = file_info.get('language', 'unknown')
        lengths_str = f"Length: {L} (code tokens) | Language: {language.upper()}"
        if include_comments_bool:
            lengths_str += " (with comments)"
        else:
            lengths_str += " (without comments)"
        
        # Оновлюємо стан аналізу
        analysis_state = {
            'needs_reanalysis': True,
            'include_comments': include_comments_bool,
            'timestamp': time()
        }
        
        return ({"include_comments": include_comments_bool}, 
                [lengths_str], w, wh, we, wm, analysis_state)
    
    # Обробка зміни файлу або split (process_selected_file логіка)
    elif trigger_id in ['file-selector', 'split', 'file-language-store']:
        model = dict()
        V = 0
        length_updated = False
        
        # Перевіряємо, чи це програмний код
        if file_info and file_info['type'] == 'code':
            # Обробка коду
            language = file_info['language']
            if isinstance(include_comments, list):
                include_comments_bool = True if True in include_comments else False
            else:
                include_comments_bool = bool(include_comments)
            
            # Зберігаємо стан чекбоксу для цього файлу
            include_comments_state[selected_filename] = include_comments_bool
            
            code_tokens = process_code(file_content, language, include_comments_bool, selected_filename)
            
            # Оновлюємо глобальні змінні
            data = code_tokens
            L = len(data)
            
            # Запам'ятовуємо довжину для цього файлу
            if selected_filename not in file_lengths:
                file_lengths[selected_filename] = {}
            
            file_lengths[selected_filename]['code_tokens'] = L
            
            # Зберігаємо тип файлу
            file_types[selected_filename] = {'type': 'code', 'language': language}
            
            # Обчислюємо параметри вікна
            if L < 20:
                wm = min(10, L)
                w = min(3, L)
                wh = 1
                we = 1
            else:
                wm = int(L / 20)
                w = int(wm / 20)
                wh = w
                we = w
            
            length_updated = True
            
            # Показуємо інформацію про довжину
            lengths_str = f"Length: {L} (code tokens) | Language: {language.upper()}"
            if include_comments_bool:
                lengths_str += " (with comments)"
            else:
                lengths_str += " (without comments)"
            
            return [dash.no_update, [lengths_str], w, w, w, wm, dash.no_update]
        else:
            # Обробка звичайного тексту
            if definition == "dynamic":
                data = prepere_data(file_content, n_size, split)
                L = len(data)
                wm = int(L / 10)
                w = int(wm / 10)
                wh = w
                we = w
            else:
                temp = []
                if split == "letter":
                    file_text = re.sub(r'	', '', file_content)
                    data = remove_punctuation(file_text)
                    for word in data:
                        for i in word:
                            if is_valid_letter(i):
                                continue
                            temp.append(i)
                    data = temp
                elif split == "symbol":
                    data = file_content
                    data = re.sub(r'	', '', file_content)
                    data = re.sub(r'\n+', '\n', data)
                    data = re.sub(r'\n\s\s', '\n', data)
                    data = re.sub(r'﻿', '', data)
                    temp = []
                    for i in data:
                        if i == " ":
                            temp.append("space")
                        elif i == "\n":
                            temp.append("space")
                            continue
                        elif i == "\ufeff":
                            temp.append("space")
                            continue
                        elif i == '﻿' or is_valid_letter(i):
                            continue
                        else:
                            i = i.lower()
                            temp.append(i)
                    data = temp
                elif split == "word":
                    file_text = re.sub(r'\n+', '\n', file_content)
                    file_text = re.sub(r'\n\s\s', '\n', file_text)
                    file_text = re.sub(r'﻿', '', file_text)
                    file_text = re.sub(r'--', ' -', file_text)
                    processor = NgrammProcessor()
                    processor.preprocess(file_text)
                    data = processor.get_words()

            L = len(data)
            wm = int(L / 20)
            w = int(wm / 20)
            wh = w
            we = w
            
            # Зберігаємо тип файлу
            file_types[selected_filename] = {'type': 'regular', 'language': 'none'}
            
            length_updated = True
            
            # Забезпечуємо мінімальні значення
            wm = max(10, wm)
            w = max(5, w)
            wh = max(1, wh)
            we = max(1, we)
            
            # Показуємо всі три довжини для вибраного файлу
            lengths_str = "Length: {} ({}s) | ".format(L, split)
            for split_type in ['word', 'symbol', 'letter']:
                if split_type != split and split_type in file_lengths.get(selected_filename, {}):
                    lengths_str += "{}s: {} | ".format(split_type, file_lengths[selected_filename][split_type])
            lengths_str = lengths_str.rstrip(" | ")
            
            return [dash.no_update, [lengths_str], w, w, w, wm, dash.no_update]
    
    # Якщо не знайдено відповідного тригера
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

def remove_empty_strings(arr):
    return [item for item in arr if item != '\ufeff']

new_ngram = None

# @app.callback(
#     Output('dummy-output', 'data'),  # Додайте dummy output в layout
#     [Input('file-selector', 'value')],
#     prevent_initial_call=True
# )
# def clear_on_file_change(selected_filename):
#     global model, V, data, L
    
#     # Очищуємо глобальні змінні при зміні файлу
#     model = dict()
#     V = 0
#     data = []
#     L = 0
    
#     return None


@app.callback(
    [Output("batch_table", "data"),
     Output("batch_results_container", "style")],
    [Input("batch_process", "n_clicks")],
    [State("fmin1", "value"),
     State("fmin2", "value"),
     State("split", "value"),
     State("n_size", "value"),
     State("condition", "value"),
     State("def", "value"),
     State("min_dist_option", "value"),
     State("overlap_mode", "value"),
     State("w", "value"),
     State("wh", "value"),
     State("we", "value"),
     State("wm", "value"),
     State("batch_window_mode", "value")]
)
def process_all_files(n_clicks, fmin1, fmin2, split, n_size, condition, definition, min_dist_option, 
                      overlap_mode, w, wh, we, wm, batch_window_mode):
    global batch_results, uploaded_files, file_lengths, file_types
    global L, data, length_updated, model, V, df, new_ngram

    if n_clicks is None or not uploaded_files:
        return [], {"display": "none"}
    
    # Find Lmin and Lmax for the current split method
    lengths = []
    for filename in list(uploaded_files.keys()):
        # Перевіряємо тип файлу
        file_type_info = file_types.get(filename, {'type': 'regular', 'language': 'none'})
        
        if file_type_info['type'] == 'code':
            # Отримуємо збережене значення для цього файлу або використовуємо за замовчуванням
            include_comments_value = include_comments_state.get(filename, True)
            data = prepere_data(file_content, n_size, split, file_type='code', 
                               language=file_type_info['language'], 
                               filename=filename, 
                               include_comments=include_comments_value)
        else:
            # Для звичайного тексту використовуємо вибраний тип розбиття
            if split in file_lengths.get(filename, {}):
                lengths.append(file_lengths[filename][split])
    
    if not lengths:
        return [], {"display": "none"}
        
    lmin = min(lengths)
    lmax = max(lengths)
    
    # Initialize batch results list
    batch_results = []
    
    # Process each file
    for idx, (filename, file_content) in enumerate(list(uploaded_files.items()), 1):
        # Додайте обробку типу файлу
        file_type_info = file_types.get(filename, {'type': 'regular', 'language': 'none'})
        
        # Вибираємо потрібну довжину в залежності від типу файлу
        if file_type_info['type'] == 'code':
            file_length = file_lengths.get(filename, {}).get('code_tokens', 0)
        else:
            file_length = file_lengths.get(filename, {}).get(split, 0)
        
        # Calculate F_min based on file length
        if lmin == lmax:
            f_min = fmin1  # If all files are the same length
        else:
            # Linear interpolation between fmin1 and fmin2
            f_min = fmin1 + (fmin2 - fmin1) * (file_length - lmin) / (lmax - lmin)
            f_min = round(f_min)  # Round to nearest integer
        
        # Process the file
        start_time = time()
        
        length_updated = False
        
        # Різна підготовка даних в залежності від типу файлу
        if file_type_info['type'] == 'code':

            include_comments_value = False
            data = prepere_data(file_content, n_size, split, file_type='code', 
                         language=file_type_info['language'], 
                         filename=filename, 
                         include_comments=include_comments_value)
        else:
            if definition == "dynamic":
                data = prepere_data(file_content, n_size, split)
            else:
                temp = []
                if split == "letter":
                    file_text = re.sub(r'	', '', file_content)
                    data = remove_punctuation(file_text)
                    for word in data:
                        for i in word:
                            if is_valid_letter(i):
                                continue
                            temp.append(i)
                    data = temp
                elif split == "symbol":
                    data = file_content
                    data = re.sub(r'	', '', file_content)
                    data = re.sub(r'\n+', '\n', data)
                    data = re.sub(r'\n\s\s', '\n', data)
                    data = re.sub(r'﻿', '', data)
                    temp = []
                    for i in data:
                        if i == " ":
                            temp.append("space")
                        elif i == "\n":
                            temp.append("space")
                            continue
                        elif i == "\ufeff":
                            temp.append("space")
                            continue
                        elif i == '﻿' or is_valid_letter(i):
                            continue
                        else:
                            i = i.lower()
                            temp.append(i)
                    data = temp
                elif split == "word":
                    file_text = re.sub(r'\n+', '\n', file_content)
                    file_text = re.sub(r'\n\s\s', '\n', file_text)
                    file_text = re.sub(r'﻿', '', file_text)
                    file_text = re.sub(r'--', ' -', file_text)
                    processor = NgrammProcessor()
                    processor.preprocess(file_text)
                    data = processor.get_words()

        L = len(data)
        
        # Calculate window parameters based on batch settings
        if batch_window_mode == "ui":
            # Use the values from the UI
            wm_val = int(wm) if wm is not None else int(L / 20)
            w_val = int(w) if w is not None else int(wm_val / 10)
            wh_val = int(wh) if wh is not None else w_val
            we_val = int(we) if we is not None else w_val
        else:  # auto
            # Calculate based on file length
            if definition == "dynamic":
                wm_val = int(L / 10)
                w_val = int(wm_val / 10)
            else:
                wm_val = int(L / 20)
                w_val = int(wm_val / 20)
            wh_val = w_val
            we_val = w_val
            
        # Ensure we have valid non-zero values
        wm_val = max(10, wm_val)
        w_val = max(5, w_val)
        wh_val = max(1, wh_val)
        we_val = max(1, we_val)
        
        length_updated = True
        
        # Make Markov chain
        make_markov_chain(data, order=n_size)
        current_df = make_dataframe(model, f_min)
        
        # Process positions and calculate parameters
        for index, ngram in enumerate(current_df['ngram']):
            model[ngram].dt = calculate_distance(np.array(model[ngram].pos, dtype=np.uint32), L, condition, ngram, min_dist_option)
        windows = list(range(w_val, wm_val, we_val))
        
        temp_gamma = []
        temp_R = []
        temp_error = []
        temp_a = []
        
        # Process windows and calculate parameters
        for i, ngram in enumerate(current_df["ngram"]):
            for wind in windows:
                if overlap_mode == "overlapping":
                    model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=wh_val, overlap_mode=overlap_mode)
                else:
                    model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=wh_val, 
                                                           overlap_mode=overlap_mode, min_window=w_val, window_expansion=we_val)
                model[ngram].fa[wind] = mse(model[ngram].counts[wind])

            model[ngram].temp_fa = []
            ff = [*model[ngram].fa.values()]
            
            try:
                c, _ = curve_fit(fit, windows, ff, method='lm', maxfev=5000)
                model[ngram].a = c[0]
                model[ngram].gamma = c[1]
                for w_val in windows:
                    model[ngram].temp_fa.append(fit(w_val, c[0], c[1]))
                temp_error.append(round(r2_score(ff, model[ngram].temp_fa), 5))
                temp_gamma.append(round(c[1], 8))
                temp_a.append(round(c[0], 8))
            except:
                # Handle curve fitting errors
                temp_error.append(0)
                temp_gamma.append(0)
                temp_a.append(0)
                
            r = round(R(np.array(model[ngram].dt)), 8)
            temp_R.append(r)
            model[ngram].R = r
            
        if n_size > 1:
            temp_ngram = []
            for ng in current_df['ngram']:
                if isinstance(ng, tuple):
                    temp_ngram.append(" ".join(ng))
            temp_ngram.append("new_ngram")
            current_df["ngram"] = temp_ngram
            
        current_df['R'] = temp_R
        current_df['γ'] = temp_gamma
        current_df['a'] = temp_a
        current_df['goodness'] = temp_error
        current_df = current_df.sort_values(by="F", ascending=False)
        current_df['rank'] = range(1, len(temp_R) + 1)
        current_df = current_df.set_index(pd.Index(np.arange(len(current_df))))
        
        # Calculate the 8 parameters
        df_filtered = current_df[current_df.ngram != 'new_ngram'].copy()
        if len(df_filtered) > 0:
            df_filtered['w'] = (df_filtered['F']) / (df_filtered['F'].sum())
            
            R_avg = df_filtered['R'].mean()
            dR = df_filtered['R'].std()
            Rw_avg = (df_filtered['R'] * df_filtered['w']).sum()
            dRw = np.sqrt((((df_filtered['R'] - Rw_avg) ** 2) * df_filtered['w']).sum())
            
            gamma_avg = df_filtered['γ'].mean()
            dgamma = df_filtered['γ'].std()
            gammaw_avg = (df_filtered['γ'] * df_filtered['w']).sum()
            dgammaw = np.sqrt((((df_filtered['γ'] - gammaw_avg) ** 2) * df_filtered['w']).sum())
        else:
            R_avg = dR = Rw_avg = dRw = gamma_avg = dgamma = gammaw_avg = dgammaw = 0
        
        end_time = time()
        processing_time = end_time - start_time
        
        # Додаткова інформація про тип файлу
        file_type_info = file_types.get(filename, {'type': 'regular', 'language': 'none'})
        
        # Store results
        result = {
            "no": idx,
            "filename": filename,
            "f_min": f_min,
            "length": L,
            "vocabulary": V - 1,  # Excluding 'new_ngram'
            "time": round(processing_time, 4),
            "r_avg": round(R_avg, 8),
            "dr": round(dR, 8),
            "rw_avg": round(Rw_avg, 8),
            "drw": round(dRw, 8),
            "g_avg": round(gamma_avg, 8),
            "dg": round(dgamma, 8),
            "gw_avg": round(gammaw_avg, 8),
            "dgw": round(dgammaw, 8),
            # Add window parameters to results
            "w_val": w_val,
            "wh_val": wh_val,
            "we_val": we_val, 
            "wm_val": wm_val,
            # Add file type information
            "file_type": file_type_info['type'],
            "language": file_type_info['language']
        }
        
        batch_results.append(result)
    
    # Calculate mean and standard deviation across all files
    if batch_results:
        # Extract numeric columns
        numeric_columns = ['f_min', 'length', 'vocabulary', 'time', 
                           'r_avg', 'dr', 'rw_avg', 'drw', 
                           'g_avg', 'dg', 'gw_avg', 'dgw',
                           'w_val', 'wh_val', 'we_val', 'wm_val']
        
        # Calculate means
        means = {col: round(np.mean([result[col] for result in batch_results]), 8) for col in numeric_columns}
        means['no'] = 'Mean'
        means['filename'] = 'Average'
        means['file_type'] = ''
        means['language'] = ''
        
        # Calculate standard deviations
        stds = {col: round(np.std([result[col] for result in batch_results]), 8) for col in numeric_columns}
        stds['no'] = 'StdDev'
        stds['filename'] = 'Std. Dev.'
        stds['file_type'] = ''
        stds['language'] = ''
        
        # Add summary rows
        batch_results.append(means)
        batch_results.append(stds)
    
    return batch_results, {"display": "block"}
@app.callback(
    Output("batch_table", "columns"),
    [Input("batch_process", "n_clicks")]
)
def update_batch_table_columns(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    columns = [
        {"name": "No.", "id": "no"},
        {"name": "Filename", "id": "filename"},
        {"name": "F_min", "id": "f_min"},
        {"name": "Length (L)", "id": "length"},
        {"name": "Vocabulary (V)", "id": "vocabulary"},
        {"name": "Time (s)", "id": "time"},
        {"name": "R_avg", "id": "r_avg"},
        {"name": "dR", "id": "dr"},
        {"name": "Rw_avg", "id": "rw_avg"},
        {"name": "dRw", "id": "drw"},
        {"name": "γ_avg", "id": "g_avg"},
        {"name": "dγ", "id": "dg"},
        {"name": "γw_avg", "id": "gw_avg"},
        {"name": "dγw", "id": "dgw"},
        {"name": "W", "id": "w_val"},
        {"name": "WH", "id": "wh_val"},
        {"name": "WE", "id": "we_val"},
        {"name": "WM", "id": "wm_val"}
    ]
    
    return columns

# Add callback to save batch results
@app.callback(
    Output("temp_seve_batch", "children"),  # Changed output ID to avoid conflicts
    [Input("save_batch", "n_clicks")],
    [State("n_size", "value"),
     State("split", "value"),
     State("condition", "value"),
     State("def", "value"),
     State("min_dist_option", "value"),
     State("overlap_mode", "value")]
)
def save_batch_results(n_clicks, n_size, split, condition, definition, min_dist_option, overlap_mode):
    if n_clicks is None or not batch_results:
        return html.Div(["No batch results to save"])
    
    try:
        # Create DataFrame from batch results
        df_batch = pd.DataFrame(batch_results)
        
        # Create filename with parameters
        output_filename = "saved_data/batch_results_n={},split={},condition={},definition={},min_dist={},overlap={}.xlsx".format(
            n_size, split, condition, definition, min_dist_option, overlap_mode)
        
        # Save to Excel - modify to use older pandas style
        writer = pd.ExcelWriter(output_filename)
        df_batch.to_excel(writer, index=False)
        writer.save()
        
        return html.Div(["Saved batch results to {}".format(output_filename)])
    except Exception as e:
        return html.Div(["Error saving batch results: {}".format(str(e))])

@app.callback([Output("table", "data"), 
               Output("chain", "figure"),
               Output("box_tab", "style"),
               Output("box_chain", "style"),
               Output("alert", "children"),
               Output("v", "children"),
               Output("t", "children"),
               Output('click-toast', 'is_open')],
              [Input("chain_button", "n_clicks"),
               Input("dataframe", "active_tab"),
               Input("analysis-state", "data")],
              [State("f_min", "value"),
               State("w", "value"),
               State("wh", "value"),
               State("we", "value"),
               State("wm", "value"),
               State("def", "value"),
               State("min_dist_option", "value"),
               State("overlap_mode", "value"),
               State("n_size", "value"),
               State("split", "value"),
               State("condition", "value"),
               State("file-language-store", "data"),
               State("file-selector", "value")
               ])
def update_table(n, dataframe, analysis_state, f_min, w, wh, we, wm, definition, min_dist_option, 
                 overlap_mode, n_size, split, condition, file_info, selected_filename):
    global length_updated, data, model, df, L, V, ngram, g, new_ngram
    
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Якщо змінився стан аналізу і потрібен повторний аналіз
    if trigger_id == 'analysis-state' and analysis_state and analysis_state.get('needs_reanalysis'):
        # Очищаємо попередні результати
        df = None
        return [[], dash.no_update, {"display": "inline"}, {"display": "none"},
                dbc.Alert("Дані оновлено. Натисніть 'Analyze' для повторного аналізу.", 
                         color="info", duration=3000),
                dash.no_update, dash.no_update, dash.no_update]

    if 'V' not in globals():
        V = 0

    print("=== update_table function called ===")
    print(f"n_clicks: {n}, dataframe: {dataframe}, definition: {definition}")

    # Решта коду залишається без змін, тільки видалені output для temp-state-holder.data
    if n is None:
        print("n is None, returning all no_update")
        return (dash.no_update, dash.no_update, {"display": 'inline'}, 
                {"display": "none"}, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update)

    # Перевірка глобальних змінних
    if not globals().get('data') and not globals().get('L'):
        if selected_filename in uploaded_files:
            file_content = uploaded_files[selected_filename]
            # Спробуйте підготувати дані
            try:
                data = prepere_data(file_content, n_size, split, 
                                    file_type='code' if file_info and file_info.get('type') == 'code' else 'regular',
                                    language=file_info.get('language') if file_info else None,
                                    filename=selected_filename,
                                    include_comments=True)
                L = len(data) if data else 0
                V = 0
                model = dict()
            except Exception as e:
                print(f"Error preparing data: {e}")
                return ([], dash.no_update, {"display": "inline"}, {"display": "none"}, 
                        dbc.Alert(f"Error processing data: {str(e)}", color="danger", duration=2000),
                        ["Length: 0"], ["Time: 0"], dash.no_update)
        else:
            return ([], dash.no_update, {"display": "inline"}, {"display": "none"}, 
                    dbc.Alert("No file selected or uploaded", color="danger", duration=2000),
                    ["Length: 0"], ["Time: 0"], dash.no_update)

    if not length_updated:
        print("length_updated is False, returning toast notification")
        return (dash.no_update, dash.no_update, {"display": 'inline'}, 
                {"display": "none"}, dash.no_update, dash.no_update, dash.no_update,
                True)

    # Перевірка на data
    if data is None or (isinstance(data, list) and len(data) == 0):
        return ([], dash.no_update, {"display": "inline"}, {"display": "none"}, 
                dbc.Alert("No data to analyze", color="danger", duration=2000),
                ["Length: 0"], ["Time: 0"], dash.no_update)
    
    # Перевірка на L
    if not isinstance(L, int) or L <= 0:
        L = len(data) if data else 0
        if L <= 0:
            return ([], dash.no_update, {"display": "inline"}, {"display": "none"}, 
                    dbc.Alert("Invalid data length", color="danger", duration=2000),
                    ["Length: 0"], ["Time: 0"], dash.no_update)

    print(f"data length: {len(data)}, L: {L}, V: {globals().get('V', 'not defined')}")

    if dataframe == "markov_chain":
        print("=== Building Markov Chain Graph ===")
        ## make markov chain graph ###
        g = nx.MultiGraph()
        temp = {}
        
        print(f"Number of ngrams to process: {len(df['ngram'])}")
        for i, ngram in enumerate(df['ngram']):
            if i < 5:  # Log first 5 ngrams
                print(f"Processing ngram {i}: {ngram}")
            
            if n_size > 1:
                ngram = tuple(ngram.split())

            g.add_node(ngram)
            temp[ngram[0]] = ngram

        print(f"Graph nodes created: {len(g.nodes())}")
        
        for node in g.nodes():
            if node[0] == "new_ngram":
                node = 'new_ngram'
            for i in model[node]:
                if i in temp:
                    g.add_edge(node, temp[i], weight=model[node][i])

        print(f"Graph edges created: {len(g.edges())}")
        
        pos = nx.spring_layout(g)
        print("Spring layout calculated")

        edge_x = []
        edge_y = []
        print("Processing edges for visualization...")
        for edge in g.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        print("Processing nodes for visualization...")
        for node in g.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title=dict(
                        text='Node Connections',
                        side='right'
                    ),
                    xanchor='left'
                ),
                line_width=2))
                
        node_adjacencies = []
        node_text = []

        print("Calculating node adjacencies...")
        for node, adjacencies in enumerate(g.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            if n_size > 1:
                node_text.append(
                    '<b>' + " ".join(adjacencies[0]) + "</b>" + '<br><br>connections=' + str(len(adjacencies[1])))
                continue
            node_text.append(
                "<b>" + "".join(adjacencies[0]) + "</b>" + '<br><br>connections: ' + str(len(adjacencies[1])))

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text
        
        print("Creating Plotly figure...")
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0),
                            annotations=[dict(
                                showarrow=True,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        print("Markov chain visualization complete")
        return (dash.no_update, fig, {"display": "none"}, {
            "display": 'inline'}, dash.no_update, dash.no_update, dash.no_update,
                 dash.no_update)
                 
    if dataframe == "data_table":
        if definition == "dynamic":
            print("=== Starting Dynamic Mode ===")
            start = time()
            we = max(1, we)
            try:
              windows = list(range(w, wm, we))
            except ValueError:
              we = 1
              windows = list(range(w, wm, we))
            print(f"Windows range: {w} to {wm} step {we}")
            print(f"Total windows: {len(windows)}")
            
            # 2. create newNgram
            print("Creating newNgram...")
            new_ngram = newNgram(data, wh, L)
            
            for i, w in enumerate(windows):
                if i % 10 == 0:  # Log every 10th window
                    print(f"Processing window {i}/{len(windows)}: {w}")
                    
                if overlap_mode == "overlapping":
                    new_ngram.func(w)
                else:
                    new_ngram.func(w, overlap_mode=overlap_mode, min_window=w, window_expansion=we)
            
            # calculate coefs
            print("Calculating coefficients...")
            temp_v = []
            temp_pos = []
            for i, ngram in enumerate(data):
                if ngram not in temp_v:
                    temp_v.append(ngram)
                    temp_pos.append(i)
                    
            print(f"Unique ngrams: {len(temp_v)}")
            print("Calculating distance...")
            new_ngram.dt = calculate_distance(np.array(temp_pos, dtype=np.uint8), L, condition, ngram, min_dist_option)
            print("Calculating R...")
            new_ngram.R = round(R(new_ngram.dt), 8)
            print(f"R value: {new_ngram.R}")
            
            print("Performing curve fit...")
            c, _ = curve_fit(fit, list(new_ngram.dfa.keys()), list(new_ngram.dfa.values()), method='lm', maxfev=5000)
            new_ngram.a = round(c[0], 8)
            new_ngram.gamma = round(c[1], 8)
            print(f"Curve fit results - a: {new_ngram.a}, gamma: {new_ngram.gamma}")
            
            new_ngram.temp_dfa = []
            for w in new_ngram.dfa.keys():
                new_ngram.temp_dfa.append(fit(w, new_ngram.a, new_ngram.gamma))
                
            new_ngram.goodness = round(r2_score(list(new_ngram.dfa.values()), new_ngram.temp_dfa), 8)
            print(f"Goodness of fit: {new_ngram.goodness}")
            
            df = pd.DataFrame()
            df['rank'] = [1]
            df['ngram'] = ['new_ngram']
            df["F"] = [len(temp_pos)]
            df['R'] = [new_ngram.R]
            df["a"] = [new_ngram.a]
            df["γ"] = [new_ngram.gamma]
            df['goodness'] = [new_ngram.goodness]
            V = len(temp_v)
            print(f"Final V (vocabulary): {V}")
            print(f"Dynamic mode processing time: {time() - start:.4f}s")

        else:
            print("=== Starting Static Mode ===")
            ###  MAKE MARKOV CHAIN ####
            start = time()
            print(f"Making Markov chain with order: {n_size}")
            make_markov_chain(data, order=n_size)
            print("Creating dataframe...")
            df = make_dataframe(model, f_min)
            print(f"DataFrame shape: {df.shape}")

            for index, ngram in enumerate(df['ngram']):
                if index < 5:  # Log first 5 ngrams
                    print(f"Calculating distance for ngram {index}: {ngram}")
                model[ngram].dt = calculate_distance(np.array(model[ngram].pos, dtype=np.uint32), L, condition, ngram, min_dist_option)

            def func(wind):
                if overlap_mode == "overlapping":
                    model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=wh, overlap_mode=overlap_mode)
                else:
                    # Для non-overlapping mode
                    model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=wh, 
                                                            overlap_mode=overlap_mode, min_window=w, window_expansion=we)
                
                model[ngram].fa[wind] = mse(model[ngram].counts[wind])
            we = max(1, we)
            try:
               windows = list(range(w, wm, we))
            except ValueError:
               we = 1
               windows = list(range(w, wm, we))
            print(f"Windows range: {w} to {wm} step {we}")
            print(f"Total windows: {len(windows)}")

            temp_gamma = []
            temp_R = []
            temp_error = []
            temp_ngram = []
            temp_a = []

            # NOTE розділити на дві частини windows
            mid = len(windows) // 2
            windows_part1 = windows[:mid]
            windows_part2 = windows[mid:]

            def process_windows(windows_part):
                for _wind in windows_part:
                    func(_wind)

            # NOTE найбільш важкий цикл
            print("Processing ngrams (main loop)...")
            start_loop = time()
            for i, ngram in enumerate(df["ngram"]):
                if i % 100 == 0:  # Log every 100th ngram
                    print(f"Processing ngram {i}/{len(df)}...")

                for wind in windows:
                    func(wind)

                model[ngram].temp_fa = []
                ff = [*model[ngram].fa.values()]

                # NOTE спричиняє проблеми при паралелізації (теж вимагає виконання по порядку,
                # окрім змінних в наступній записці)
                c, _ = curve_fit(fit, windows, ff, method='lm', maxfev=5000)
                model[ngram].a = c[0]
                model[ngram].gamma = c[1]
                for w in windows:
                    model[ngram].temp_fa.append(fit(w, c[0], c[1]))
                temp_error.append(round(r2_score(ff, model[ngram].temp_fa), 5))
                temp_gamma.append(round(c[1], 8))
                temp_a.append(round(c[0], 8))

                if isinstance(ngram, tuple):
                    temp_ngram.append(" ".join(ngram))

                r = round(R(np.array(model[ngram].dt)), 8)

                temp_R.append(r)
                model[ngram].R = r

            print(f"Main loop processing time: {time() - start_loop:.4f}s")

            if n_size > 1:
                print("Adding 'new_ngram' to ngram list")
                # HERE REMOVE
                temp_ngram.append("new_ngram")
                df["ngram"] = temp_ngram

            print("Updating DataFrame with calculated values...")
            #     NOTE через ці змінні в циклі які оновлюються по порядку і потім записуються напряму ж в колонку,
            #     неможливо просто так розділити
            df['R'] = temp_R
            df['γ'] = temp_gamma
            df['a'] = temp_a
            df['goodness'] = temp_error
            
            print("Sorting DataFrame by frequency...")
            df = df.sort_values(by="F", ascending=False)
            df['rank'] = range(1, len(temp_R) + 1)
            df = df.set_index(pd.Index(np.arange(len(df))))
            
            print(f"Final DataFrame shape: {df.shape}")
            print(f"Static mode total processing time: {time() - start:.4f}s")

        voc = str(V)
        voc = int(voc) - 1
        # HERE V-1
        print(f"Final vocabulary (V-1): {voc}")
        print(f"Total execution time: {time() - start:.4f}s")

        return [df.to_dict(orient='records'), dash.no_update, {"display": "inline"}, {"display": "none"},
                dash.no_update,
                # NOTE повернення додаткових 8-ми значень на фронт-енд
                ["Vocabulary: " + str(voc)], ["Time:" + str(round(time() - start, 4))],
                 dash.no_update
                ]


clikced_ngram = None

@app.callback(
    Output("comments-switch-container", "style"),
    [Input('file-language-store', 'data')]
)
def toggle_comments_switch(file_info):
    if file_info and file_info.get('type') == 'code':
        return {"display": "block", "margin": "10px 0", "padding": "5px", "background-color": "#f8f9fa", "borderRadius": "5px"}
    return {"display": "none"}

@app.callback(
    Output("language-selector-container", "style"),
    [Input("text-type-selector", "value")]
)
def toggle_language_selector(text_type):
    if text_type == "code":
        return {"display": "block"}
    return {"display": "none"}

@app.callback(
    [Output('text-type-modal', 'is_open'),
     Output('file-language-store', 'data')],
    [Input('file-selector', 'value'),
     Input('text-type-confirm', 'n_clicks')],
    [State('text-type-selector', 'value'),
     State('text-type-modal', 'is_open')]
)
def handle_file_type_selection(selected_filename, confirm_clicks, text_type, is_open):
    # Додамо Store для збереження мови і типу файлу
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return False, {'type': 'regular', 'language': 'none'}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'file-selector' and selected_filename:
        # Файл вибрано, показуємо модальне вікно
        return True, dash.no_update
    
    elif trigger_id == 'text-type-confirm':
        # Користувач підтвердив вибір типу тексту
        if text_type == 'code':
            language = detect_programming_language(selected_filename)
        else:
            language = 'none'
        
        return False, {'type': text_type, 'language': language}
    
    return is_open, dash.no_update

@app.callback(
    [Output('file-type-info', 'children'),
     Output('file-type-info', 'style')],
    [Input('file-language-store', 'data'),
     Input('file-selector', 'value'),
     Input('include-comments-switch', 'value')]
)
def update_file_type_info(file_info, filename, include_comments):
    if not filename or not file_info:
        return "", {"display": "none"}
    
    if file_info['type'] == 'regular':
        return "Тип: Звичайний текст", {"display": "block"}
    else:
        comments_text = " (з коментарями)" if include_comments else " (без коментарів)"
        return f"Тип: Програмний код ({file_info['language'].upper()}){comments_text}", {"display": "block"}

@app.callback([Output("graphs", "figure"), Output("fa", "figure"), ],
              [Input("dataframe", "active_tab"),
               Input("card-tabs", "active_tab"),
               Input("table", "active_cell"),
                # NOTE додала параметр page_current та використала його для показу правильної інформації
               Input("table", "page_current"),
               Input("table", "derived_virtual_selected_rows"),
               Input("table", "derived_virtual_indices"),
               Input("chain", "clickData"),
               Input("scale", "value"),
               Input("fa", "clickData"),
               Input("graphs", "clickData"),
               Input("wh", "value")],
              [State("n_size", "value"),
               State("def", "value"), ])
def tab_content(active_tab2, active_tab1, active_cell, page_current, row_ids, ids, clicked_data, scale, fa_click,
                graph_click, wh, n,
                definition):
    global model, df, L, g, new_ngram, ngram
    if df is None:
        return dash.no_update, dash.no_update

    if ids is None:
        return dash.no_update, dash.no_update

    # NOTE логіка для обрання правильного рядка слова при активному номері сторінки далі ніж перша.
    # Довжина сторінки 50 слів тому множимо на 50
    if active_cell is not None and page_current is not None and page_current > 0:
        active_cell['row'] = active_cell['row'] + page_current * 50

    df = df.reindex(pd.Index(ids))
    fig = go.Figure()

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=10))
    fig1 = go.Figure()

    fig1.update_layout(margin=dict(l=0, r=0, t=0, b=15))
    if active_tab2 == "markov_chain":
        if definition == "dynamic":
            return dash.no_update, dash.no_update

        if clicked_data:
            nodes = np.array(g.nodes())
            ngram = nodes[clicked_data['points'][0]['pointNumber']]
            if n > 1:

                ngram = tuple(nodes[clicked_data['points'][0]['pointNumber']])

                if ngram[0] == 'new_ngram':
                    ngram = 'new_ngram'

            if active_tab1 == "tab2":
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool))
                if fa_click:
                    if overlap_mode == "overlapping":
                        fig.add_trace(
                            go.Bar(x=np.arange(wh, L, wh), y=model[ngram].counts[fa_click["points"][0]["x"]], name="∑∆w"))
                    else:
                        # Для non-overlapping режиму потрібно розрахувати положення барів
                        bar_positions = []
                        k = 1
                        i = 0
                        ww = fa_click["points"][0]["x"]
                        while i < L - ww:
                            bar_positions.append(i)
                            shift = calc_non_overlapping_shift(k, w, we)
                            i += shift
                            k += 1
                        fig.add_trace(go.Bar(x=bar_positions, y=model[ngram].counts[ww], name="∑∆w"))

                fa_click = None
                fig1.add_trace(
                    go.Scatter(x=[*model[ngram].fa.keys()],
                               y=[*model[ngram].fa.values()],
                               mode='markers',
                               name="∆F"))
                fig1.add_trace(go.Scatter(
                    x=[*model[ngram].fa.keys()],
                    y=model[ngram].temp_fa,
                    name="fit"))
                fig1.update_xaxes(type=scale)
                fig1.update_yaxes(type=scale)
                fig1.update_layout(hovermode="x unified")

                return fig, fig1
            if active_tab1 == "tab3":
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool))
                if fa_click:
                    if overlap_mode == "overlapping":
                        fig.add_trace(
                            go.Bar(x=np.arange(wh, L, wh), y=model[ngram].counts[fa_click["points"][0]["x"]], name="∑∆w"))
                    else:
                        # Для non-overlapping режиму потрібно розрахувати положення барів
                        bar_positions = []
                        k = 1
                        i = 0
                        ww = fa_click["points"][0]["x"]
                        while i < L - ww:
                            bar_positions.append(i)
                            shift = calc_non_overlapping_shift(k, w, we)
                            i += shift
                            k += 1
                        fig.add_trace(go.Bar(x=bar_positions, y=model[ngram].counts[ww], name="∑∆w"))
                    print(model[ngram].sums[fa_click['points'][0]['x']])
                fa_click = None

                hover_data = []
                for data in df['ngram']:
                    hover_data.append("".join(data))
                fig1.add_trace(go.Scatter(x=df["R"], y=df["γ"], mode="markers", text=hover_data))
                fig1.add_trace(go.Scatter(x=[model[ngram].R],
                                          y=[model[ngram].gamma],
                                          mode="markers",
                                          text=' '.join(ngram),
                                          marker=dict(
                                              size=20,
                                              color="red"
                                          )))
                fig1.update_layout(showlegend=False)
                fig1.update_yaxes(type=scale)
                fig1.update_xaxes(type=scale)
                fig1.update_layout(hovermode="x unified")
                return fig, fig1
            else:
                return fig, fig1

        return dash.no_update, dash.no_update
    else:
        if active_tab1 == "tab2":
            if active_cell:

                if definition == "dynamic":
                    ## add bar
                    if fa_click:
                        if overlap_mode == "overlapping":
                            fig.add_trace(go.Bar(x=np.arange(wh, L, wh), y=new_ngram.count[fa_click["points"][0]["x"]],
                                                name="∑∆w"))
                        else:
                            # Для non-overlapping режиму потрібно розрахувати положення барів
                            bar_positions = []
                            k = 1
                            i = 0
                            ww = fa_click["points"][0]["x"]
                            while i < L - ww:
                                bar_positions.append(i)
                                shift = calc_non_overlapping_shift(k, w, we)
                                i += shift
                                k += 1
                            fig.add_trace(go.Bar(x=bar_positions, y=new_ngram.count[ww], name="∑∆w"))

                    fig1.add_trace(
                        go.Scatter(x=[*new_ngram.dfa.keys()], y=[*new_ngram.dfa.values()], mode='markers', name="∆F"))
                    fig1.add_trace(go.Scatter(x=[*new_ngram.dfa.keys()], y=[*new_ngram.temp_dfa], name="fit=aw^b"))
                    fig1.update_xaxes(type=scale)
                    fig1.update_yaxes(type=scale)
                    fig1.update_layout(hovermode="x unified")

                    return fig, fig1

                if n > 1:
                    ngram = tuple(df['ngram'][ids[active_cell['row']]].split())
                    if ngram[0] == 'new_ngram':
                        ngram = 'new_ngram'
                else:
                    ngram = df['ngram'][ids[active_cell['row']]]
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool, name="positions"))

                if fa_click:
                    if overlap_mode == "overlapping":
                        fig.add_trace(go.Bar(x=np.arange(wh, L, wh), y=model[ngram].counts[fa_click["points"][0]["x"]],
                                             name="∑∆w"))
                    else:
                        # Для non-overlapping режиму потрібно розрахувати положення барів
                        bar_positions = []
                        k = 1
                        i = 0
                        while i < L - ww:
                            bar_positions.append(i)
                            shift = calc_non_overlapping_shift(k, w, we)
                            i += shift
                            k += 1
                        fig.add_trace(go.Bar(x=bar_positions, y=model[ngram].counts[ww], name="∑∆w"))
                if graph_click:
                    www = graph_click['points'][0]['x']
                graph_click = None
                fa_click = None

                temp_ww = [*model[ngram].fa.keys()]
                fig1.add_trace(
                    go.Scatter(x=temp_ww,
                               y=[*model[ngram].fa.values()],
                               mode='markers',
                               name="∆F"))
                fig1.add_trace(go.Scatter(
                    x=temp_ww,
                    y=model[ngram].temp_fa,
                    name="fit=aw^b"))
                fig1.update_xaxes(type=scale)
                fig1.update_yaxes(type=scale)
                fig1.update_layout(hovermode="x unified")
                active_cell = None
                return fig, fig1
            else:
                active_cell = None
                return fig, fig1
        else:
            hover_data = []
            if active_cell:
                if definition == "dynamic":
                    if fa_click:
                        fig.add_trace(
                            go.Bar(x=np.arange(wh, L, wh), y=new_ngram.count[fa_click["points"][0]["x"]], name="∑∆w"))

                    fig1.add_trace(go.Scatter(x=new_ngram.R, y=new_ngram.gamma, mode='marekers', hover_data=["new_ngram"]))
                    fig1.update_xaxes(type=scale)
                    fig1.update_yaxes(type=scale)
                    fig1.update_layout(hovermode="x unified")

                    return fig, fig1

                if n > 1:
                    ngram = tuple(df['ngram'][ids[active_cell['row']]].split())
                    if ngram[0] == 'new_ngram':
                        ngram = 'new_ngram'
                else:
                    ngram = df['ngram'][ids[active_cell['row']]]

                for data in df['ngram']:
                    # HERE ADDED to skip random float entities
                    if not isinstance(data, numbers.Number):
                        hover_data.append("".join(data))
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool, name="positions"))
                if fa_click:
                    ww = fa_click['points'][0]["x"]
                    # HERE ww-1
                    if overlap_mode == "overlapping":
                        fig.add_trace(go.Bar(x=np.arange(ww, L, wh), y=model[ngram].counts[ww], name="∑∆w"))
                    else:
                        # Для non-overlapping режиму потрібно розрахувати положення барів
                        bar_positions = []
                        k = 1
                        i = 0
                        while i < L - ww:
                            bar_positions.append(i)
                            shift = calc_non_overlapping_shift(k, w, we)
                            i += shift
                            k += 1
                        fig.add_trace(go.Bar(x=bar_positions, y=model[ngram].counts[ww], name="∑∆w"))

                fa_click = None
                if graph_click:
                    print(model[ngram].sums.keys())

                graph_click = None

                fig1.add_trace(go.Scatter(x=df["R"], y=df["γ"], mode="markers", text=hover_data))
                # fig1.add_trace(go.Scatter(x=[df['R'][active_cell['row']]],
                fig1.add_trace(go.Scatter(x=[df['R'][ids[active_cell['row']]]],
                                          # y=[df["b"][active_cell['row']]],
                                          y=[df["γ"][ids[active_cell['row']]]],
                                          mode="markers",
                                          text=' '.join(ngram),
                                          marker=dict(
                                              size=20,
                                              color="red"
                                          )))
                fig1.update_layout(showlegend=False)
                fig1.update_yaxes(type=scale)
                fig1.update_xaxes(type=scale)
                fig1.update_layout(hovermode="x unified")
                active_cell = None

            return fig, fig1

        return dash.no_update, dash.no_update


@app.callback([Output("temp_seve", "children")],
              [Input("save", "n_clicks"),
               Input("table", "active_cell"),
               Input("table", "page_current"),
               Input("table", "derived_virtual_indices")],
              [State("file-selector", "value"),
               State("n_size", "value"),
               State("w", "value"),
               State("wh", "value"),
               State("we", "value"),
               State("wm", "value"),
               State("f_min", "value"),
               State("condition", "value"),
               State("def", "value"),
               State("min_dist_option", "value"),
               State("overlap_mode", "value")])
def save(n, active_cell, page_current, ids, filename, n_size, w, wh, we, wm, fmin, opt, definition, min_dist_option, overlap_mode):
    if n is None or filename is None:
        return dash.no_update
    else:
        # The file parameter is now the selected filename
        file = filename
        global df, model, new_ngram

        #   2023
        #   Зміни в save
        #   - вивід без new_ngram
        #   - додаткові параметри

        # Create a copy to avoid modifying the global df directly during calculations if needed
        df_copy = df.copy()

        df_copy = df_copy[df_copy.ngram != 'new_ngram']

        # Recalculate rank if needed (ensure it's 0-based or 1-based consistently)
        # If starting from 0:
        # df_copy['rank'] = range(len(df_copy))
        # If starting from 1 (like original):
        df_copy['rank'] = range(1, len(df_copy) + 1)


        if len(df_copy) > 0: # Ensure dataframe is not empty before calculating stats
            df_copy['w'] = (df_copy['F']) / (df_copy['F'].sum())

            R_avg = df_copy['R'].mean()
            dR = df_copy['R'].std()
            Rw_avg = (df_copy['R'] * df_copy['w']).sum()
            dRw = np.sqrt((((df_copy['R'] - Rw_avg) ** 2) * df_copy['w']).sum())

            gamma_avg = df_copy['γ'].mean()
            dgamma = df_copy['γ'].std()
            gammaw_avg = (df_copy['γ'] * df_copy['w']).sum()
            dgammaw = np.sqrt((((df_copy['γ'] - gammaw_avg) ** 2) * df_copy['w']).sum())

            # Assign calculated values using .loc to avoid SettingWithCopyWarning
            df_copy.loc[:, 'R_avg'] = None
            df_copy.loc[df_copy.index[0], 'R_avg'] = R_avg
            df_copy.loc[:, 'dR'] = None
            df_copy.loc[df_copy.index[0], 'dR'] = dR
            df_copy.loc[:, 'Rw_avg'] = None
            df_copy.loc[df_copy.index[0], 'Rw_avg'] = Rw_avg
            df_copy.loc[:, 'dRw'] = None
            df_copy.loc[df_copy.index[0], 'dRw'] = dRw

            df_copy.loc[:, 'γ_avg'] = None
            df_copy.loc[df_copy.index[0], 'γ_avg'] = gamma_avg
            df_copy.loc[:, 'dγ'] = None
            df_copy.loc[df_copy.index[0], 'dγ'] = dgamma
            df_copy.loc[:, 'γw_avg'] = None
            df_copy.loc[df_copy.index[0], 'γw_avg'] = gammaw_avg
            df_copy.loc[:, 'dγw'] = None
            df_copy.loc[df_copy.index[0], 'dγw'] = dgammaw

            # Remove temporary 'w' column if not needed in the final output
            df_copy = df_copy.drop(columns=['w'])

        else:
             # Handle empty dataframe case if necessary
             # Maybe return an alert or log a message
             print("Warning: DataFrame is empty after filtering 'new_ngram'. Cannot save stats.")
             # Decide how to handle df_copy columns if it's empty
             pass


        if definition == "dynamic":
            output_filename = "saved_data/{0} condition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8},min_dist={9},overlap={10}.xlsx".format(file, fmin, n_size, w, wh, we, wm, opt, definition, min_dist_option, overlap_mode)
            # Changed to older pandas style without with context
            writer = pd.ExcelWriter(output_filename)
            df_copy.to_excel(writer, index=False)
            writer.save()

            if active_cell and new_ngram: # Check if new_ngram exists
                # Existing logic for saving new_ngram data...
                # NOTE: Ensure that 'active_cell' logic correctly identifies the row AFTER filtering 'new_ngram'
                # This part might need review depending on whether active_cell refers to the original df or df_copy
                # Assuming it refers to the state *before* this function modified df globally

                # Handle potential errors if ids or active_cell['row'] are invalid for the *original* df
                try:
                    # Original logic used global df, let's assume we still need info based on the original selection state
                    original_df = df # Reference the global df as it was upon entering the function
                    current_ids = ids # Use the passed ids

                    # Correct row index considering pagination
                    row_index = active_cell['row']
                    if page_current is not None and page_current > 0:
                         row_index += page_current * 50 # Assuming page size is 50

                    # Get the ngram based on the original selection state
                    # Check if the selected index is valid in the *original* derived indices
                    if current_ids is not None and row_index < len(current_ids):
                        selected_original_index = current_ids[row_index]
                        # Check if this index exists in the original df before filtering
                        if selected_original_index < len(original_df):
                             ngram_to_save_details = original_df.iloc[selected_original_index]['ngram']

                             # Ensure it's not the filtered 'new_ngram' (though unlikely if active_cell logic is sound)
                             if ngram_to_save_details != 'new_ngram':

                                 details_filename = "saved_data/{} {}_details.xlsx".format(file, ngram_to_save_details)
                                 # Changed to older pandas style without with context
                                 writer_details = pd.ExcelWriter(details_filename)
                                 df1 = pd.DataFrame()
                                 # Check if the ngram exists in the global model (might have been filtered)
                                 if ngram_to_save_details in model:
                                     df1["w"] = list(model[ngram_to_save_details].fa.keys())
                                     df1['∆F'] = list(model[ngram_to_save_details].fa.values()) # Original code had '∆F', assuming this is correct?
                                     df1['fit=a*w^b'] = model[ngram_to_save_details].temp_fa
                                     df1.to_excel(writer_details, index=False)
                                     writer_details.save()
                                 # Also save new_ngram specific data
                                 if new_ngram: # Save new_ngram details if definition is dynamic
                                     new_ngram_details_filename = "saved_data/{} new_ngram_dynamic_details.xlsx".format(file)
                                     writer_new_ngram = pd.ExcelWriter(new_ngram_details_filename)
                                     df_new = pd.DataFrame()
                                     df_new["w"] = list(new_ngram.dfa.keys())
                                     df_new['∆F'] = list(new_ngram.dfa.values()) # Original used ∆F here
                                     df_new['fit=a*w^b'] = new_ngram.temp_dfa
                                     df_new.to_excel(writer_new_ngram, index=False)
                                     writer_new_ngram.save()

                        else:
                            print("Warning: Selected index {} out of bounds for original DataFrame.".format(selected_original_index))
                    else:
                        print("Warning: Calculated row index {} is invalid for derived indices.".format(row_index))

                except Exception as e:
                    print("Error saving detailed ngram file (dynamic): {}".format(e))
                    # Potentially add a Dash alert to inform the user


            return [html.Div("Saved data to {}".format(output_filename))] # Provide feedback

        # Static definition part
        output_filename_static = "saved_data/{0} condition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8},min_dist={9},overlap={10}.xlsx".format(
                file, fmin, n_size, w, wh, we, wm, opt, definition, min_dist_option, overlap_mode
            )
        # Changed to older pandas style without with context 
        writer = pd.ExcelWriter(output_filename_static)
        df_copy.to_excel(writer, index=False)
        writer.save()

        if active_cell:
            # Similar logic as above to get the correct ngram based on original selection state
            try:
                original_df = df
                current_ids = ids
                row_index = active_cell['row']
                if page_current is not None and page_current > 0:
                    row_index += page_current * 50

                if current_ids is not None and row_index < len(current_ids):
                     selected_original_index = current_ids[row_index]
                     if selected_original_index < len(original_df):
                        ngram = original_df.iloc[selected_original_index]['ngram']

                        # Ensure it's not 'new_ngram' (already filtered in df_copy, but check original selection)
                        if ngram != 'new_ngram':
                            # Check if ngram exists in the model dictionary
                            if ngram in model:
                                details_filename_static = "saved_data/{} {}.xlsx".format(file, ngram)
                                # Changed to older pandas style without with context
                                writer_details = pd.ExcelWriter(details_filename_static)
                                df1 = pd.DataFrame()
                                df1["w"] = list(model[ngram].fa.keys())
                                df1['∆F'] = list(model[ngram].fa.values()) # Original used ∆F here
                                df1['fit=a*w^b'] = model[ngram].temp_fa
                                df1.to_excel(writer_details, index=False)
                                writer_details.save()
                            else:
                                print("Warning: Ngram '{}' selected but not found in model for detail saving.".format(ngram))
                     else:
                        print("Warning: Selected index {} out of bounds for original DataFrame (static).".format(selected_original_index))
                else:
                    print("Warning: Calculated row index {} is invalid for derived indices (static).".format(row_index))

            except Exception as e:
                print("Error saving detailed ngram file (static): {}".format(e))
                # Potentially add a Dash alert

    # Use the modified df_copy for saving, keep global df potentially unchanged if needed elsewhere
    # Or update global df if necessary: df = df_copy
    # For now, just provide feedback
    return [html.Div("Saved data.")] # Generic feedback if filename isn't always generated


# import webbrowser # Commented out as it might cause issues if run non-interactively

if __name__ == "__main__":
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run_server(host='0.0.0.0', port=8050, debug=False)
# Add callback to toggle batch window settings
@app.callback(
    Output("batch_custom_controls", "is_open"),
    [Input("batch_window_mode", "value")]
)
def toggle_batch_window_controls(mode):
    return mode in ["ui", "auto"]
