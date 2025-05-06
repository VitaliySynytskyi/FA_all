import numbers
from typing import List, Tuple, Optional, Dict, Any, Union
import gc  # Garbage Collector для кращого управління пам'яттю

import numpy as np
from numba import jit, njit, prange
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Обробка даних і тексту
import re
from string import punctuation
from time import time
import openpyxl

# Dash і візуалізація
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# Системні і допоміжні бібліотеки
import base64
import io
from os import listdir
import webbrowser
from dash.dependencies import Input, Output, State
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numba
import os

# tkinter for selecting browsing folder
import tkinter as tk
from tkinter import filedialog

# lexers to process computer code texts
from pygments.lexers import TextLexer
from pygments import lex
from pygments.lexers import guess_lexer_for_filename, get_lexer_by_name
from pygments.token import Token

# Функція для очищення пам'яті
def clear_memory(keep: List[str] = []):
    """
    Очищує пам'ять від великих структур даних, які більше не потрібні.
    
    Args:
        keep: Список назв змінних, які потрібно зберегти
    """
    global model, df, new_ngram, data, uploaded_files, file_lengths, batch_results
    
    # Зберігаємо лише необхідні дані для таблиці
    variables_to_keep = keep + ['uploaded_files', 'file_lengths', 'batch_results']
    
    # Очищення великих глобальних структур даних
    if 'model' not in variables_to_keep and 'model' in globals():
        if isinstance(model, dict):
            model.clear()
        model = {}
    
    # Очищення DataFrame
    if 'df' not in variables_to_keep and 'df' in globals() and df is not None:
        df = None
    
    # Очищення даних тексту
    if 'data' not in variables_to_keep and 'data' in globals() and data is not None:
        data = None
    
    # Очищення об'єкта newNgram
    if 'new_ngram' not in variables_to_keep and 'new_ngram' in globals() and new_ngram is not None:
        new_ngram = None
    
    # Очищення кешу мемоізованих функцій
    if hasattr(prepare_data, 'clear_cache') and 'prepare_data_cache' not in variables_to_keep:
        prepare_data.clear_cache()
    
    if hasattr(make_markov_chain, 'clear_cache') and 'make_markov_chain_cache' not in variables_to_keep:
        make_markov_chain.clear_cache()
    
    # Додаємо агресивне очищення пам'яті за допомогою Python gc
    import gc
    gc.collect(generation=2) # Запуск повного збирання сміття
    gc.collect(generation=1)
    gc.collect(generation=0)

# Кешування для покращення продуктивності
def memoize(func):
    """
    Декоратор для кешування результатів функцій, щоб уникнути повторних обчислень.
    """
    cache = {}
    
    def wrapper(*args, **kwargs):
        # Створюємо унікальний ключ на основі аргументів
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    # Додаємо функцію для очищення кешу
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper


def remove_punctuation_for_words(data):
    """
    Розбиває текст на слова та видаляє знаки пунктуації.
    
    Args:
        data: Вхідний текст
        
    Returns:
        List[str]: Список оброблених слів
    """
    # Використовуємо ефективніший регулярний вираз один раз
    words = re.findall(r'\b[a-zA-Z0-9]+(?:[-\'][a-zA-Z0-9]+)*\b', data.lower())
    
    # Обробляємо слова з дефісами та апострофами
    result = []
    for word in words:
        if '-' in word or '\'' in word:
            # Розділяємо слово на підчастини за спеціальними символами
            parts = re.split(r'[-\']', word)
            # Додаємо лише непорожні частини
            result.extend([part for part in parts if part])
        else:
            result.append(word)
    
    return result


def remove_punctuation(data):
    """
    Видаляє знаки пунктуації з тексту.
    """
    temp = []
    for i in range(len(data)):
        if data[i] in punctuation:
            continue
        else:
            temp.append(data[i].lower())
    return "".join(temp)

toast_visible = False
error_visible = False
analyze_visible = False
isComputerText = False
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


def make_dataframe(model, fmin=3):
    """
    Створює DataFrame для відображення результатів аналізу.

    Args:
        model: Словник моделі з n-грамами
        fmin: Мінімальна частота для включення n-грами в аналіз

    Returns:
        pd.DataFrame: DataFrame з результатами
    """
    # Фільтруємо n-грами за мінімальною частотою
    filtered_data = list(
        filter(lambda x: sum(value for value in model[x].values() if isinstance(value, int)) >= fmin, model))

    # Додаємо new_ngram, якщо вона існує в моделі
    if 'new_ngram' not in filtered_data and 'new_ngram' in model:
        filtered_data.append("new_ngram")

    # Створюємо структуру даних для DataFrame
    data = {"ngram": [],
            "F": np.empty(len(filtered_data), dtype=np.dtype(int))}

    # Заповнюємо дані
    for i, ngram in enumerate(filtered_data):
        data["ngram"].append(ngram)

        if ngram == "new_ngram" and hasattr(model[ngram], 'bool'):
            data['F'][i] = sum(model[ngram].bool)
        elif ngram == "new_ngram":
            # Якщо атрибут bool відсутній, встановлюємо значення за замовчуванням
            data['F'][i] = 0
        elif hasattr(model[ngram], 'pos'):
            data["F"][i] = len(model[ngram].pos)
        else:
            data["F"][i] = 0

    # Створюємо DataFrame з даних
    dffff = pd.DataFrame(data=data)
    return dffff


@memoize
def make_markov_chain(data: List, order: int = 1) -> Dict[str, Ngram]:
    """
    Створює ланцюг Маркова з вхідних даних.
    
    Args:
        data: Список елементів для побудови ланцюга Маркова
        order: Порядок ланцюга Маркова (кількість попередніх елементів для прогнозу)
        
    Returns:
        Dict[str, Ngram]: Модель ланцюга Маркова у вигляді словника n-грам
    """
    global model, L, V
    
    # Створюємо новий словник моделі
    model = dict()
    L = len(data) - order
    
    # Ініціалізуємо спеціальну n-граму для нових елементів
    model['new_ngram'] = Ngram()
    model['new_ngram'].bool = np.zeros(L, dtype=np.uint8)  # використовуємо uint8 для зменшення пам'яті
    model['new_ngram'].pos = []
    
    # Використовуємо більш ефективний алгоритм для побудови ланцюга Маркова
    if order > 1:
        for i in range(L - 1):
            window = tuple(data[i: i + order])  # Додаємо в словник
            
            if window in model:  # Приєднуємо до вже існуючого розподілу
                model[window].update([data[i + order]])
                model[window].pos.append(i + 1)
                model[window].bool[i] = 1
            else:
                model[window] = Ngram([data[i + order]])
                model[window].pos = []
                model[window].pos.append(i + 1)
                model[window].bool = np.zeros(L, dtype=np.uint8)
                model[window].bool[i] = 1
                model['new_ngram'].bool[i] = 1
                model['new_ngram'].pos.append(i + 1)
    else:
        # Попередньо визначаємо множину унікальних елементів для оптимізації
        unique_items = set(data)
        
        # Ініціалізуємо модель для кожного унікального елемента
        for item in unique_items:
            model[item] = Ngram()
            model[item].pos = []
            model[item].bool = np.zeros(L, dtype=np.uint8)
        
        # Заповнюємо модель
        for i in range(L):
            item = data[i]
            next_item = data[i + order]
            
            model[item].update([next_item])
            model[item].pos.append(i + order)
            model[item].bool[i] = 1
            
            if i == 0:  # Перший елемент
                model['new_ngram'].bool[i] = 1
                model['new_ngram'].pos.append(i + order)

        # З'єднуємо останнє слово з першим та перше з останнім
        model[data[L]].update([data[0]])
        if data[L] not in model[data[L]].pos:
            model[data[L]].pos.append(L + order)
            model[data[L]].bool = np.zeros(L, dtype=np.uint8)
            model[data[L]].bool[L-1] = 1
        
        model[data[0]].update([data[L]])
        
    V = len(model)
    return model


def calculate_distance(positions: np.ndarray, L: int, option: str, ngram: str, min_dist: int = 1) -> np.ndarray:
    """
    Розраховує відстані між позиціями елементів з урахуванням граничних умов.
    
    Оптимізована для роботи з великими наборами даних за допомогою паралельної обробки.
    
    Args:
        positions: Масив позицій елементів
        L: Довжина тексту
        option: Тип граничних умов ("no", "ordinary", "periodic")
        ngram: Назва n-грами
        min_dist: Мінімальна відстань (0 або 1)
        
    Returns:
        np.ndarray: Масив відстаней між елементами
    """
    # Оптимізуємо обробку масиву позицій
    positions = np.array(positions, dtype=np.int32)
    
    # Переконуємося, що min_dist є цілим числом
    if not isinstance(min_dist, int):
        try:
            min_dist = int(min_dist)
        except (ValueError, TypeError):
            print(f"Warning: min_dist '{min_dist}' is not an integer. Using default min_dist=1")
            min_dist = 1
    
    # Використовуємо оптимізовані функції відповідно до граничних умов
    if option == "no":
        distances = nbc(positions, L, min_dist)
    elif option == "periodic":
        distances = pbc(positions, L, min_dist)
    else:  # "ordinary"
        distances = obc(positions, L, min_dist)
    
    return distances


@njit(parallel=True)
def nbc(pos, L, min_dist=1):
    """
    Обчислює відстані без граничних умов.
    
    Оптимізовано за допомогою Numba JIT з паралельною обробкою.
    
    Args:
        pos: Масив позицій елементів
        L: Довжина послідовності
        min_dist: Мінімальна відстань
        
    Returns:
        np.ndarray: Масив відстаней
    """
    n = len(pos)
    dt = np.zeros(n - 1, dtype=np.int32)
    
    for i in prange(n - 1):
        dt[i] = pos[i + 1] - pos[i]
        if min_dist==0:
            dt[i] -= 1
    
    return dt


@njit(parallel=True)
def pbc(pos, L, min_dist=1):
    """
    Обчислює відстані з періодичними граничними умовами.
    
    Оптимізовано за допомогою Numba JIT з паралельною обробкою.
    
    Args:
        pos: Масив позицій елементів
        L: Довжина послідовності
        min_dist: Мінімальна відстань
        
    Returns:
        np.ndarray: Масив відстаней
    """
    n = len(pos)
    dt = np.zeros(n, dtype=np.int32)
    
    for i in prange(n - 1):
        dt[i] = pos[i + 1] - pos[i]
        if dt[i] > L // 2:
            dt[i] = L - dt[i]
        if min_dist==0:
            dt[i] -= -1
    
    # Останній елемент обчислюємо окремо через періодичність
    dt[n - 1] = L - pos[n - 1] + pos[0]
    if dt[n - 1] > L // 2:
        dt[n - 1] = L - dt[n - 1]
    if min_dist==0:
        dt[n - 1] -= 1
    
    return dt


@njit(parallel=True)
def obc(pos, L, min_dist=1):
    """
    Обчислює відстані зі звичайними граничними умовами.
    
    Оптимізовано за допомогою Numba JIT з паралельною обробкою.
    
    Args:
        pos: Масив позицій елементів
        L: Довжина послідовності
        min_dist: Мінімальна відстань
        
    Returns:
        np.ndarray: Масив відстаней
    """
    n = len(pos)
    dt = np.zeros(n, dtype=np.int32)
    
    for i in prange(n - 1):
        dt[i] = pos[i + 1] - pos[i]
        if min_dist==0:
            dt[i] -= 1
    
    # Останній елемент обчислюємо окремо
    dt[n - 1] = L - pos[n - 1] + pos[0]
    if dt[n - 1] < min_dist:
        dt[n - 1] = min_dist
    if min_dist==0:
        dt[n - 1] -= 1
    
    return dt


@jit(nopython=True)
def s(window: np.ndarray) -> int:
    """
    Обчислює суму значень вікна.
    
    Args:
        window: Масив значень
        
    Returns:
        int: Сума значень
    """
    # Використовуємо оптимізовану NumPy функцію
    return np.sum(window)


@njit(fastmath=True)
def mse(x: np.ndarray) -> float:
    """
    Обчислює середньоквадратичну похибку (MSE) набору значень.
    
    Args:
        x: Масив значень
        
    Returns:
        float: Значення MSE
    """
    if len(x) == 0:
        return 0.0
        
    # Оптимізоване обчислення MSE
    mean_x = np.mean(x)
    return np.sqrt(np.mean((x - mean_x) ** 2))


@jit(nopython=True, fastmath=True)
def R(x: np.ndarray) -> float:
    """
    Обчислює коефіцієнт варіації.
    
    Args:
        x: Масив значень
        
    Returns:
        float: Значення коефіцієнта варіації
    """
    if len(x) <= 1:
        return 0.0
        
    # Оптимізоване обчислення коефіцієнта варіації
    mean_x = np.mean(x)
    if mean_x == 0:  # Запобігаємо діленню на нуль
        return 0.0
    std_x = np.std(x)
    return std_x / mean_x


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

@njit(fastmath=True)
def make_windows(x: np.ndarray, wi: int, l: int, wsh: int, 
                overlap_mode: str = "overlapping", 
                min_window: Optional[int] = None, 
                window_expansion: Optional[int] = None) -> np.ndarray:
    """
    Створює вікна для аналізу даних.
    
    Args:
        x: Вхідний масив даних
        wi: Розмір вікна
        l: Довжина даних
        wsh: Величина зсуву вікна
        overlap_mode: Режим перекриття вікон ("overlapping" або "non-overlapping")
        min_window: Мінімальний розмір вікна для режиму non-overlapping
        window_expansion: Значення розширення вікна для режиму non-overlapping
        
    Returns:
        np.ndarray: Масив сум у вікнах
    """
    # Використовуємо Numba для оптимізації
    if overlap_mode == "overlapping":
        # Визначаємо кількість вікон заздалегідь для уникнення повторного обчислення
        num_windows = (l - wi) // wsh + 1
        sums = np.zeros(num_windows, dtype=np.float64)
        
        # Використовуємо ефективніший цикл
        for i in range(num_windows):
            start_idx = i * wsh
            end_idx = start_idx + wi
            # Використовуємо вбудовану функцію sum у NumPy
            sums[i] = np.sum(x[start_idx:end_idx])
            
    else:  # non-overlapping режим
        # Використовуємо правильні значення за замовчуванням
        min_win = wi if min_window is None else min_window
        win_exp = wi if window_expansion is None else window_expansion
        
        # Визначаємо кількість вікон
        num_windows = (l - wi) // wi + 1
        sums = np.zeros(num_windows, dtype=np.float64)
        
        # Використовуємо ефективніший цикл для non-overlapping
        for i in range(num_windows):
            start_idx = i * wi
            end_idx = start_idx + wi
            if end_idx > l:
                end_idx = l
            sums[i] = np.sum(x[start_idx:end_idx])
    
    return sums


@njit(fastmath=True)
def calc_sum(x):
    sums = np.empty(len(x))
    for i, w in enumerate(x):
        sums[i] = np.sum(w)
    return sums


@jit(nopython=True, fastmath=True)
def fit(x, a, b):
    return a * (x ** b)


@memoize
def prepare_data(data: str, n: int, split: str, filename: str) -> List:
    """
    Підготовка даних для аналізу, розбиття на n-грами залежно від вказаних параметрів.
    
    Args:
        data: Вхідний текст для обробки
        n: Розмір n-грами
        split: Метод розбиття тексту ("word", "letter", "symbol")
        
    Returns:
        List: Список підготовлених даних
    """
    global L
    if n is None:
        return dash.no_update
    
    # Використовуємо спільний код попередньої обробки для всіх типів
    data = re.sub(r'\n+', '\n', data)
    data = re.sub(r'\n\s\s', '\n', data)
    data = re.sub(r'﻿', '', data)
    
    # Для n=1 (одиничні елементи)
    if n == 1:
        if split == "word":
            if isComputerText:
                lexer = guess_lexer_or_text(data, filename)
                if not isinstance(lexer, TextLexer): 
                    result = tokenize_code(data, lexer)
                else:
                    print(f"could not guess language of file {filename}")
                    data = re.sub(r'--', ' -', data)
                    processor = NgrammProcessor()
                    processor.preprocess(data)
                    result = processor.get_words()
            else:
                # Обробка тексту для слів
                data = re.sub(r'--', ' -', data)
                processor = NgrammProcessor()
                processor.preprocess(data)
                result = processor.get_words()
 
            L = len(result)
            return result
            
        elif split == 'letter':
            # Обробка для літер і чисел
            temp = []
            data = remove_punctuation(data)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            L = len(temp)
            return temp
            
        elif split == 'symbol':
            # Обробка для символів
            result = []
            for char in data:
                if char == " " or char == "\n" or char == "\ufeff":
                    result.append("space")
                else:
                    result.append(char.lower())
            L = len(result)
            return result
    
    # Для n>1 (n-грами)
    else:
        if split == "word":
            if isComputerText:
                lexer = guess_lexer_or_text(data, filename)
                if not isinstance(lexer, TextLexer): 
                    words = tokenize_code(data, lexer)
                else:
                    print(f"could not guess language of file {filename}")
                    data = re.sub(r'--', ' -', data)
                    processor = NgrammProcessor()
                    processor.preprocess(data)
                    words = processor.get_words()
            else:
                # Обробка тексту для слів
                data = re.sub(r'--', ' -', data)
                processor = NgrammProcessor()
                processor.preprocess(data)
                words = processor.get_words()
            
            # Обробка для n-грам слів
            L = len(words)
            
            # Створюємо n-грами з слів
            result = []
            for i in range(L - n + 1):
                window = tuple(words[i:i + n])
                result.append(window)
            
            return result
                
        elif split == "letter":
            # Обробка для n-грам літер і чисел
            temp = []
            data = remove_punctuation(data)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            L = len(temp)
            data = temp
            temp = []
            for i in range(L - n + 1):
                window = tuple(data[i:i + n])
                temp.append(window)
            return temp
                
        elif split == 'symbol':
            # Обробка для n-грам символів
            temp = []
            for char in data:
                if char == " " or char == "\n" or char == "\ufeff":
                    temp.append("space")
                else:
                    temp.append(char.lower())
            data = temp
            L = len(data)
            temp = []
            for i in range(L - n + 1):
                window = tuple(data[i:i + n])
                temp.append(window)
            return temp

    return []

def dfa(data: List, args: Tuple[int, int, int], 
       overlap_mode: str = "overlapping", 
       min_window: Optional[int] = None, 
       window_expansion: Optional[int] = None) -> np.ndarray:
    """
    Виконує аналіз флуктуацій (DFA) для даних.
    
    Args:
        data: Вхідні дані для аналізу
        args: Кортеж (розмір вікна, зсув вікна, довжина даних)
        overlap_mode: Режим перекриття вікон ("overlapping" або "non-overlapping")
        min_window: Мінімальний розмір вікна для режиму non-overlapping
        window_expansion: Значення розширення вікна для режиму non-overlapping
        
    Returns:
        np.ndarray: Масив результатів DFA аналізу
    """
    wi, wh, l = args
    
    if overlap_mode == "overlapping":
        # Стандартний режим з фіксованим зміщенням
        window_count = len(range(0, l - wi, wh))
        count = np.zeros(window_count, dtype=np.uint8)
        
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
            
        # Оцінюємо кількість і розташування вікон
        k = 1
        i = 0
        window_positions = []
        while i < l - wi:
            window_positions.append(i)
            shift = calc_non_overlapping_shift(k, min_window, window_expansion)
            i += shift
            k += 1
            
        count = np.zeros(len(window_positions), dtype=np.uint8)
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
    
    return count


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
        count = dfa(self.data, (w, self.wh, self.l), overlap_mode, min_window, window_expansion)
        self.count[w] = count
        self.dfa[w] = float(mse(count)) # Окремо обчислюємо MSE для count


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dictionary to store uploaded files
uploaded_files = {}
# Dictionary to store file lengths with structure: {filename: {'word': length, 'symbol': length, 'letter': length}}
file_lengths = {}
# List to store batch processing results
batch_results = []

# Removing the corpuses list since we're using file upload now
# corpuses = listdir("corpus/")
colors = {
    "background": "#a1a1a1",
    "text": "#a1a1a1"}

import dash_bootstrap_components as dbc

layout2 = html.Div()

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
                                    html.H6("Text Type Selection", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                
                                    html.Label("Choose Text Type:"),
                                    html.Div(
                                        [
                                            dbc.InputGroup(
                                            [
                                                html.Div(
                                                    dbc.RadioItems(
                                                        options=[
                                                            {"label": "Natural text", "value": False},
                                                            {"label": "PC text (Computer code)", "value": True}
                                                        ],
                                                        value=False,
                                                        id="computerTextRadio",
                                                        inline=True
                                                    ),
                                                    style={"marginLeft": "10px"} 
                                                ),
                                            ]
                                        ),
                                        html.Div(id='isComputerTextOutput', style={'display': 'none'})
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),
                                ]),

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
                                                    {"label": "word", "value": "word"},
                                                    {"label": "letter&number", "value": "letter"},
                                                    {"label": "symbol", "value": "symbol"},
                                                ],
                                                value="word"
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
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Min Tau:"),
                                        dbc.Select(
                                            id="min_dist_option",
                                            options=[
                                                {"label": "0", "value": "0"},
                                                {"label": "1", "value": "1"}
                                            ],
                                            value="1",
                                            style={"font-weight": "bold"}
                                        )
                                    ], className="mb-1"),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("filter"),
                                            dbc.Input(id="f_min", type="number", value=3, min=1, style={"font-weight": "bold"})
                                        ],
                                        className="mb-3"
                                    ),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),
                                
                                # WINDOW SETTINGS SECTION
                                html.Div([
                                    html.H6("Sliding Window Settings",
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
                                            html.Span("w_min = Min Window", style={"fontWeight": "bold"}), " | ",
                                            html.Span("w_s = Window Shift", style={"fontWeight": "bold"}), " | ",
                                            html.Span("w_e = Window Expansion", style={"fontWeight": "bold"}), " | ",
                                            html.Span("w_max = Max Window", style={"fontWeight": "bold"})
                                        ], className="text-muted mb-2 d-block text-center"),
                                    ], style={"background": "#f0f8ff", "padding": "6px", "borderRadius": "5px", "marginBottom": "10px"}),

                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Min", html.Br(), "Window"], style={"lineHeight": "1.2", "textAlign": "center"}),
                                                         style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_min", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-2"),
                                    
                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Window", html.Br(), "Shift"], style={"lineHeight": "1.2", "textAlign": "center"}),
                                                         style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_s", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-2"),
                                    
                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Window", html.Br(), "Expansion"], style={"lineHeight": "1.2", "textAlign": "center"}),
                                                         style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_e", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-2"),
                                    
                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Max", html.Br(), "Window"], style={"lineHeight": "1.2", "textAlign": "center"}),
                                                         style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_max", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-3"),
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
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),

                                # BATCH PROCESSING SECTION
                                html.Div([
                                    html.H6("Batch Processing", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    # Add the min-max info Div here
                                    html.Div(id='min-max-length-info', style={"marginTop": "5px", "fontSize": "small", "textAlign": "center", "marginBottom": "10px"}),
                                    
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Lmin: Fmin1"),
                                            dbc.Input(id="fmin1", type="number", value=3, min=1, style={"font-weight": "bold"})
                                        ],
                                        style={'marginBottom': '5px'}
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Lmax: Fmin2"),
                                            dbc.Input(id="fmin2", type="number", value=5, min=1, style={"font-weight": "bold"})
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
                                    dbc.Button("Save Batch Results", id="save_batch", color="primary",
                                              className="w-100",
                                              style={'marginBottom': '10px', "fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                                    html.Div(id="temp_seve_batch", style={'marginBottom': '10px'}),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),
                                
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
                                        dbc.Tab(label="DataTable", tab_id="data_table", label_style={"font-weight": "bold"})
                                    ],
                                    id="dataframe",
                                    active_tab="data_table"
                                )

                            ),
                            dbc.CardBody(
                                [
                                    # here table
                                    html.Div(id="box_tab",
                                             style={"display": "none", "height": "400px", "minHeight": "400px"},
                                             children=[dbc.Spinner(dash_table.DataTable(
                                                 id="table",
                                                 columns=[{"name": i, "id": i} for i in
                                                          ['rank', "ngram", "F", "R", "a", "gamma", "goodness"]],
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
                                                {"name": "gamma_avg", "id": "gamma_avg"},
                                                {"name": "dgamma", "id": "dgamma"},
                                                {"name": "gammaw_avg", "id": "gammaw_avg"},
                                                {"name": "dgammaw", "id": "dgammaw"}
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
                                        # Remove the old batch save button
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
                                active_tab="tab1"
                                #active_tab="tab1",
                                #card=True
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
                                active_tab="tab2"
                                #active_tab="tab2",
                                #card=True
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
    ),
])
from dash.dependencies import Input, Output, State

app.layout = layout1
df = None
g = None
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numba
import os

def is_number(s: str) -> bool:
    """
    Перевіряє, чи можна рядок перетворити в число.
    
    Args:
        s: Рядок для перевірки
        
    Returns:
        bool: True, якщо рядок може бути перетворений у число, інакше False
    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

# NOTE клас із С# для обробки слів
class NgrammProcessor:
    """
    Клас для обробки тексту і отримання n-грам.
    """
    def __init__(self, ignore_punctuation: bool = True):
        """
        Ініціалізує процесор n-грам.
        
        Args:
            ignore_punctuation: Чи ігнорувати пунктуацію при обробці
        """
        self.ignore_punctuation = ignore_punctuation
        self.words = []
        self.processed_text = ""
        
    def preprocess(self, text: str) -> None:
        """
        Попередня обробка тексту.
        
        Args:
            text: Вхідний текст для обробки
        """
        # Видаляємо пунктуацію, якщо потрібно
        if self.ignore_punctuation:
            # Використовуємо оптимізований метод видалення пунктуації
            self.processed_text = ''.join(char for char in text if char not in punctuation or char == '-' or char == "'")
        else:
            self.processed_text = text
            
        # Розбиваємо текст на слова
        self.words = [word.lower() for word in re.findall(r'\b\w+(?:[-\']\w+)*\b', self.processed_text)]
        
    def get_words(self, remove_empty_entries: bool = False) -> List[str]:
        """
        Отримує список слів із обробленого тексту.
        
        Args:
            remove_empty_entries: Чи видаляти порожні рядки
            
        Returns:
            List[str]: Список слів
        """
        if remove_empty_entries:
            return [word for word in self.words if word]
        return self.words


def is_valid_letter(char: str) -> bool:
    """
    Check if a character should be skipped.
    Returns True if character should be skipped.
    """
    invalid_characters = [' ', '\n', '\ufeff', '°', '"', '„', '–']
    return char in invalid_characters


length_updated = False


@app.callback(
    [Output('upload-status', 'children'),
     Output('file-selector', 'options'),
     Output('min-max-length-info', 'children')], # Added new output
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('n_size', 'value'),
     State('split', 'value')] # Added split state
)
def update_upload_status(contents, filenames, n_size, split_mode):
    global uploaded_files, file_lengths
    
    min_max_info = ""
    options = [{'label': filename, 'value': filename, 'title': filename} for filename in list(uploaded_files.keys())]
    
    if contents is None:
        # Calculate min/max even if no new files are uploaded, but existing ones are present
        if file_lengths and split_mode:
            lengths = [file_lengths[filename].get(split_mode, 0) for filename in file_lengths]
            if lengths:
                min_len = min(lengths)
                max_len = max(lengths)
                split_label = "letters&numbers" if split_mode == 'letter' else f"{split_mode}s"
                min_max_info = f"Min/Max Length ({split_label}): {min_len} / {max_len}"
        return html.Div(["No new files uploaded"]), options, html.Div(min_max_info)
    
    success_count = 0
    error_count = 0
    
    for i, (content, filename) in enumerate(zip(contents, filenames)):
        try:
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            
            try:
                file_content = decoded.decode('utf-8')
                uploaded_files[filename] = file_content
                file_lengths[filename] = {}
                
                # Word length
                text_word = re.sub(r'\n+', '\n', file_content)
                text_word = re.sub(r'\n\s\s', '\n', text_word)
                text_word = re.sub(r'﻿', '', text_word)
                text_word = re.sub(r'--', ' -', text_word)
                processor = NgrammProcessor()
                processor.preprocess(text_word)
                words = processor.get_words()
                file_lengths[filename]['word'] = len(words)
                
                # Symbol length
                symbols = []
                for char in file_content:
                    if char == " " or char == "\n" or char == "\ufeff":
                        symbols.append("space")
                    else:
                        symbols.append(char.lower())
                file_lengths[filename]['symbol'] = len(symbols)
                
                # Letter length
                text_letter = remove_punctuation(file_content)
                letters = []
                for word in text_letter:
                    for i in word:
                        if i == ' ':
                            continue
                        letters.append(i)
                file_lengths[filename]['letter'] = len(letters)
                
                print(f"✓ Uploaded: {filename}")
                print(f"  Words: {file_lengths[filename]['word']} | Symbols: {file_lengths[filename]['symbol']} | Letters: {file_lengths[filename]['letter']}")
                
                success_count += 1
            except UnicodeDecodeError:
                print(f"✗ Error: {filename} is not a valid text file")
                error_count += 1
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
            error_count += 1
    
    summary_message = html.Div([
        html.H5(f"Upload Summary:"),
        html.P(f"Successfully uploaded: {success_count} file(s)", style={'color': 'green'}),
        html.P(f"Files with errors: {error_count}", style={'color': 'red' if error_count > 0 else 'green'})
    ])
    
    options = [{'label': filename, 'value': filename, 'title': filename} for filename in list(uploaded_files.keys())]

    # Calculate Min/Max length based on the current split mode
    if file_lengths and split_mode:
        lengths = [file_lengths[filename].get(split_mode, 0) for filename in file_lengths]
        if lengths:
            min_len = min(lengths)
            max_len = max(lengths)
            split_label = "letters&numbers" if split_mode == 'letter' else f"{split_mode}s"
            min_max_info = f"Min/Max Length ({split_label}): {min_len} / {max_len}"
    
    return summary_message, options, html.Div(min_max_info)


# Add callback to handle file selection
@app.callback(
    [Output('l', 'children'),
     Output('w_min', 'value'),
     Output('w_s', 'value'),
     Output('w_e', 'value'),
     Output('w_max', 'value')],
    [Input('file-selector', 'value'),
     Input('split', 'value')],
    [State('def', 'value'),
     State('n_size', 'value')]
)
def process_selected_file(selected_filename, split, definition, n):
    global L, data, length_updated, isComputerText
    
    if selected_filename is None or selected_filename not in uploaded_files:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    file = uploaded_files[selected_filename]
    length_updated = False
    
    # Calculate L based on split type (dynamic or static handles data differently)
    if definition == "dynamic":
        data = prepare_data(file, n, split, selected_filename)
        L = len(data)
        w_max = int(L / 10)
        w_min = int(w_max / 10)
    else:
        # Static mode calculation based on selected split
        if split == "letter":
            temp = []
            data = remove_punctuation(file)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            data = temp
            L = len(data)
        elif split == "symbol":
            temp = []
            for char in file:
                if char == " " or char == "\n" or char == "\ufeff":
                    temp.append("space")
                else:
                    temp.append(char.lower())
            data = temp
            L = len(data)
        elif split == "word":
            if isComputerText:
                lexer = guess_lexer_or_text(file, selected_filename)
                if not isinstance(lexer, TextLexer): 
                    data = tokenize_code(file, lexer)
                else:
                    print(f"could not guess language of file {selected_filename}")
                    file = re.sub(r'\n+', '\n', file)
                    file = re.sub(r'\n\s\s', '\n', file)
                    file = re.sub(r'﻿', '', file)
                    file = re.sub(r'--', ' -', file)
                    processor = NgrammProcessor()
                    processor.preprocess(file)
                    data = processor.get_words()
            else:
                file = re.sub(r'\n+', '\n', file)
                file = re.sub(r'\n\s\s', '\n', file)
                file = re.sub(r'﻿', '', file)
                file = re.sub(r'--', ' -', file)
                processor = NgrammProcessor()
                processor.preprocess(file)
                data = processor.get_words()
            L = len(data)

        w_max = int(L / 20)
        w_min = int(w_max / 20)
        length_updated = True
    
    # Format the lengths into a multi-line Div
    length_elements = [html.Strong("Length:")]
    
    # Get all lengths from the stored dictionary
    lengths = file_lengths[selected_filename]
    
    # Add each length type on a new line
    if 'word' in lengths:
        length_elements.append(html.Div(f"words: {lengths['word']}"))
    if 'symbol' in lengths:
        length_elements.append(html.Div(f"symbols: {lengths['symbol']}"))
    if 'letter' in lengths:
        length_elements.append(html.Div(f"letters&numbers: {lengths['letter']}"))
        
    return length_elements, w_min, w_min, w_min, w_max


def remove_empty_strings(arr: List[str]) -> List[str]:
    """
    Видаляє порожні рядки та спеціальні символи з списку.
    
    Args:
        arr: Список рядків для обробки
        
    Returns:
        List[str]: Список без порожніх рядків та спеціальних символів
    """
    return [item for item in arr if item and item != '\ufeff']

new_ngram = None


# Add callback for batch processing
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
     State("w_min", "value"),
     State("w_s", "value"),
     State("w_e", "value"),
     State("w_max", "value"),
     State("batch_window_mode", "value")]
)
def process_all_files(n_clicks, fmin1, fmin2, split, n_size, condition, definition, min_dist_option, 
                      overlap_mode, w_min, w_s, w_e, w_max, batch_window_mode):
    global batch_results, uploaded_files, file_lengths, isComputerText
    
    if n_clicks is None or not uploaded_files:
        return [], {"display": "none"}
    
    # Find Lmin and Lmax for the current split method
    lengths = [file_lengths[filename][split] for filename in list(uploaded_files.keys())]
    if not lengths:
        return [], {"display": "none"}
        
    lmin = min(lengths)
    lmax = max(lengths)
    
    # Initialize batch results list
    batch_results = []
    
    # Process each file sequentially and clear memory after each
    file_list = list(uploaded_files.items())
    for idx, (filename, file_content) in enumerate(file_list, 1):
        # Force garbage collection before starting new file
        gc.collect()
        
        # Calculate F_min based on file length
        file_length = file_lengths[filename][split]
        if lmin == lmax:
            f_min = fmin1  # If all files are the same length
        else:
            # Linear interpolation between fmin1 and fmin2
            f_min = fmin1 + (fmin2 - fmin1) * (file_length - lmin) / (lmax - lmin)
            f_min = round(f_min)  # Round to nearest integer
        
        # Process the file
        start_time = time()
        
        # Initialize all variables locally to avoid memory leaks
        data = None
        local_model = {}
        
        # Process data based on definition mode
        if definition == "dynamic":
            data = prepare_data(file_content, n_size, split, filename)
        else:
            if split == "letter":
                file_text = re.sub(r'	', '', file_content)
                processed_data = remove_punctuation(file_text)
                temp = []
                current_number = ""
                
                for char in processed_data:
                    if char.isspace() or char == '\n' or char == '\ufeff':
                        if current_number:
                            temp.append(current_number)
                            current_number = ""
                        continue
                    if char.isdigit() or char.isalpha():
                        temp.append(char)
                
                data = temp
                # Free memory immediately
                del processed_data
                del temp
                del file_text
                gc.collect()
            elif split == "symbol":
                # Optimize processing
                clean_text = re.sub(r'	', '', file_content)
                clean_text = re.sub(r'\n+', '\n', clean_text)
                clean_text = re.sub(r'\n\s\s', '\n', clean_text)
                clean_text = re.sub(r'﻿', '', clean_text)
                temp = []
                for i in clean_text:
                    if i == " " or i == "\n" or i == "\ufeff":
                        temp.append("space")
                    elif i == '﻿' or is_valid_letter(i):
                        continue
                    else:
                        temp.append(i.lower())
                data = temp
                del temp
                del clean_text
                gc.collect()
            elif split == "word":
                if isComputerText:
                    lexer = guess_lexer_or_text(file_content, filename)
                    if not isinstance(lexer, TextLexer): 
                        data = tokenize_code(file_content, lexer)
                    else:
                        print(f"could not guess language of file {filename}")
                        file_text = re.sub(r'\n+', '\n', file_content)
                        file_text = re.sub(r'\n\s\s', '\n', file_text)
                        file_text = re.sub(r'﻿', '', file_text)
                        file_text = re.sub(r'--', ' -', file_text)
                        processor = NgrammProcessor()
                        processor.preprocess(file_text)
                        data = processor.get_words()
                        del processor
                        del file_text
                        gc.collect()
                else:
                    file_text = re.sub(r'\n+', '\n', file_content)
                    file_text = re.sub(r'\n\s\s', '\n', file_text)
                    file_text = re.sub(r'﻿', '', file_text)
                    file_text = re.sub(r'--', ' -', file_text)
                    processor = NgrammProcessor()
                    processor.preprocess(file_text)
                    data = processor.get_words()
                    del processor
                    del file_text
                    gc.collect()

        L = len(data)
        
        # Calculate window parameters based on batch settings
        if batch_window_mode == "ui":
            # Use the values from the UI
            wm_val = int(w_max) if w_max is not None else int(L / 20)
            w_val = int(w_s) if w_s is not None else int(wm_val / 10)
            wh_val = int(w_s) if w_s is not None else w_val
            we_val = int(w_e) if w_e is not None else w_val
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
        
        # Make Markov chain (use local variables)
        for i in range(L - n_size + 1):
            if i + n_size <= len(data):
                if n_size == 1:
                    ngram = data[i]
                else:
                    ngram = tuple(data[i:i + n_size])
                
                if ngram not in local_model:
                    local_model[ngram] = Ngram()
                    local_model[ngram].pos = []
                
                local_model[ngram].pos.append(i)
        
        # Build a vocabulary count for statistics
        V = len(local_model)
        
        # Make DataFrame locally instead of using global function to save memory
        filtered_data = list(filter(lambda x: len(local_model[x].pos) >= f_min, local_model))
        
        data_df = {"ngram": [], "F": np.empty(len(filtered_data), dtype=np.int32)}
        for i, ngram in enumerate(filtered_data):
            data_df["ngram"].append(ngram)
            data_df["F"][i] = len(local_model[ngram].pos)
        
        current_df = pd.DataFrame(data=data_df)
        
        # Process positions and calculate parameters
        temp_gamma = []
        temp_R = []
        temp_error = []
        temp_a = []
        
        windows = list(range(w_val, wm_val, we_val))
        
        # Process each ngram
        for i, row in current_df.iterrows():
            ngram = row['ngram']
            
            # Generate boolean array for this ngram
            local_model[ngram].bool = np.zeros(L, dtype=np.int8)
            for pos in local_model[ngram].pos:
                local_model[ngram].bool[pos] = 1
            
            # Calculate distances
            min_dist_int = int(min_dist_option) if isinstance(min_dist_option, (str, float)) else min_dist_option
            local_model[ngram].dt = calculate_distance(np.array(local_model[ngram].pos, dtype=np.uint32), L, condition, ngram, min_dist_int)
            
            # Process windows
            local_model[ngram].fa = {}
            local_model[ngram].counts = {}
            
            for wind in windows:
                if overlap_mode == "overlapping":
                    local_model[ngram].counts[wind] = make_windows(local_model[ngram].bool, wi=wind, l=L, wsh=wh_val, overlap_mode=overlap_mode)
                else:
                    local_model[ngram].counts[wind] = make_windows(local_model[ngram].bool, wi=wind, l=L, wsh=wh_val, 
                                                                overlap_mode=overlap_mode, min_window=w_val, window_expansion=we_val)
                local_model[ngram].fa[wind] = mse(local_model[ngram].counts[wind])
            
            ff = [local_model[ngram].fa[wind] for wind in windows]
            
            try:
                c, _ = curve_fit(fit, windows, ff, method='lm', maxfev=5000)
                a_val = c[0]
                gamma_val = c[1]
                temp_fa = [fit(w_val, c[0], c[1]) for w_val in windows]
                temp_error.append(round(r2_score(ff, temp_fa), 5))
                temp_gamma.append(round(gamma_val, 8))
                temp_a.append(round(a_val, 8))
            except:
                # Handle curve fitting errors
                temp_error.append(0)
                temp_gamma.append(0)
                temp_a.append(0)
            
            r = round(R(np.array(local_model[ngram].dt)), 8)
            temp_R.append(r)
        
        # Handle n-grams formatting if needed
        if n_size > 1:
            temp_ngram = []
            for ng in current_df['ngram']:
                if isinstance(ng, tuple):
                    temp_ngram.append(" ".join(ng))
                else:
                    temp_ngram.append(ng)
            current_df["ngram"] = temp_ngram
        
        # Add calculated parameters to DataFrame
        current_df['R'] = temp_R
        current_df['gamma'] = temp_gamma
        current_df['a'] = temp_a
        current_df['goodness'] = temp_error
        current_df = current_df.sort_values(by="F", ascending=False)
        current_df['rank'] = range(1, len(current_df) + 1)
        
        # Calculate the 8 parameters
        df_filtered = current_df.copy()
        if len(df_filtered) > 0:
            df_filtered['w'] = df_filtered['F'] / df_filtered['F'].sum()
            
            R_avg = df_filtered['R'].mean()
            dR = df_filtered['R'].std()
            Rw_avg = (df_filtered['R'] * df_filtered['w']).sum()
            dRw = np.sqrt((((df_filtered['R'] - Rw_avg) ** 2) * df_filtered['w']).sum())
            
            gamma_avg = df_filtered['gamma'].mean()
            dgamma = df_filtered['gamma'].std()
            gammaw_avg = (df_filtered['gamma'] * df_filtered['w']).sum()
            dgammaw = np.sqrt((((df_filtered['gamma'] - gammaw_avg) ** 2) * df_filtered['w']).sum())
        else:
            # Default values if no data
            R_avg = dR = Rw_avg = dRw = gamma_avg = dgamma = gammaw_avg = dgammaw = 0
        
        # Calculate execution time
        end_time = time()
        execution_time = end_time - start_time
        
        # Create batch result (store only what's needed for the table)
        batch_result = {
            "no": idx,
            "filename": filename,
            "f_min": f_min,
            "length": L,
            "vocabulary": V,
            "time": round(execution_time, 3),
            "r_avg": round(R_avg, 8),
            "dr": round(dR, 8),
            "rw_avg": round(Rw_avg, 8),
            "drw": round(dRw, 8),
            "gamma_avg": round(gamma_avg, 8),
            "dgamma": round(dgamma, 8),
            "gammaw_avg": round(gammaw_avg, 8),
            "dgammaw": round(dgammaw, 8)
        }
        
        batch_results.append(batch_result)
        
        # Explicitly clean up all local variables
        del data
        del local_model
        del current_df
        del df_filtered
        del temp_gamma
        del temp_R
        del temp_error
        del temp_a
        
        # Force garbage collection multiple times
        gc.collect()
        gc.collect()
    
    # Calculate statistics if we have results
    if batch_results:
        # Add the mean and standard deviation rows after all files processed
        add_batch_statistics(batch_results)
    
    return batch_results, {"display": "block"}
    
def add_batch_statistics(results):
    """
    Adds mean and standard deviation rows to batch results
    
    Args:
        results: List of batch results to add statistics to
    """
    if not results:
        return
    
    # Extract only numerical data for statistics
    data_for_stats = []
    numeric_fields = ["length", "vocabulary", "time", "r_avg", "dr", "rw_avg", "drw", 
                    "gamma_avg", "dgamma", "gammaw_avg", "dgammaw"]
    
    for item in results:
        # Skip statistics rows (if this function is called multiple times)
        if item["filename"] in ["MEAN", "STDDEV"]:
            continue
        
        data_point = {}
        for field in numeric_fields:
            if field in item:
                data_point[field] = item[field]
        
        data_for_stats.append(data_point)
    
    # Calculate means
    if not data_for_stats:
        return
        
    df_stats = pd.DataFrame(data_for_stats)
    
    # Calculate means
    means = {
        "no": len(results) + 1,
        "filename": "MEAN",
        "f_min": "-",
    }
    
    # Calculate standard deviations
    stddevs = {
        "no": len(results) + 2,
        "filename": "STDDEV",
        "f_min": "-",
    }
    
    # Fill in statistics for all numeric fields
    for field in numeric_fields:
        if field in df_stats.columns:
            means[field] = round(df_stats[field].mean(), 
                               8 if field in ["r_avg", "dr", "rw_avg", "drw", "gamma_avg", "dgamma", "gammaw_avg", "dgammaw"] else 
                               3 if field == "time" else 0)
            
            stddevs[field] = round(df_stats[field].std(), 
                                 8 if field in ["r_avg", "dr", "rw_avg", "drw", "gamma_avg", "dgamma", "gammaw_avg", "dgammaw"] else 
                                 3 if field == "time" else 0)
    
    # Remove old statistics rows if present
    results[:] = [r for r in results if r["filename"] not in ["MEAN", "STDDEV"]]
    
    # Add statistics to results
    results.append(means)
    results.append(stddevs)

# Update the batch results table to show window parameters too
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
        {"name": "gamma_avg", "id": "gamma_avg"},
        {"name": "dgamma", "id": "dgamma"},
        {"name": "gammaw_avg", "id": "gammaw_avg"},
        {"name": "dgammaw", "id": "dgammaw"}
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
     State("overlap_mode", "value"),
     State("batch_window_mode", "value")]
)
def save_batch_results(n_clicks, n_size, split, condition, definition, min_dist_option, overlap_mode, batch_window_mode):
    if n_clicks is None:
        return dash.no_update
    if not batch_results:
        return html.Div(["No batch results to save"])
    
    try:
        save_folder = pick_folder()
        if save_folder is None or save_folder == "":
            return dash.no_update
        # Create DataFrame from batch results
        df_batch = pd.DataFrame(batch_results)
        
        # Ensure column names match the display columns for consistency
        # This ensures the saved file has the same data structure as what's shown in the UI
        column_mapping = {}
        
        # Create filename with parameters
        output_filename = "{}/batch_results_n={},split={},condition={},definition={},min_dist={},overlap={},window_mode={}.xlsx".format(
            save_folder, n_size, split, condition, definition, min_dist_option, overlap_mode, batch_window_mode)
        
        # Ensure directory exists
        os.makedirs(save_folder, exist_ok=True)
        
        # Save to Excel - modify to use older pandas style
        writer = pd.ExcelWriter(output_filename)
        df_batch.to_excel(writer, index=False)
        writer.save()
        #writer.close()
        
        return html.Div(["Saved batch results to {}".format(output_filename)])
    except Exception as e:
        return html.Div(["Error saving batch results: {}".format(str(e))])

@app.callback([Output("table", "data"), Output("chain", "figure"),
               Output("box_tab", "style"),
               Output("box_chain", "style"),
               Output("alert", "children"),
               Output("v", "children"),
               Output("t", "children"),
                Output('click-toast', 'is_open'),
               ],
              [Input("chain_button", "n_clicks"),
               Input("dataframe", "active_tab")],
              [State("f_min", "value"),
               State("w_min", "value"),
               State("w_s", "value"),
               State("w_e", "value"),
               State("w_max", "value"),
               State("def", "value"),
               State("min_dist_option", "value"),
               State("overlap_mode", "value"),
               State("n_size", "value"),
               State("split", "value"),
               State("condition", "value")
               ])
def update_table(n, dataframe, f_min, w_min, w_s, w_e, w_max, definition, min_dist_option, overlap_mode, n_size, split, condition):
    """
    Оновлює таблицю та графік на основі вибраних параметрів.
    
    Використовує паралельну обробку для інтенсивних обчислень і оптимізоване управління пам'яттю
    для зменшення навантаження.
    """
    global model, L, V, df, new_ngram
    
    # Очищуємо кеш для мемоізованих функцій
    if hasattr(prepare_data, 'clear_cache'):
        prepare_data.clear_cache()
    if hasattr(make_markov_chain, 'clear_cache'):
        make_markov_chain.clear_cache()
    
    # Викликаємо збирач сміття для звільнення пам'яті
    clear_memory(keep=['data', 'uploaded_files', 'file_lengths'])
    
    if n is None or dataframe is None:
        return (dash.no_update, dash.no_update, {"display": "none"}, {"display": "none"},
                dash.no_update, dash.no_update, dash.no_update,
                dash.no_update)
                
    # Вже нема вкладки MarkovChain, тому використовуємо тільки data_table
    if definition == "dynamic":
        start = time()
        
        # Додаємо перевірку на None для безпеки
        w_s_val = int(w_s) if w_s is not None else 5
        w_max_val = int(w_max) if w_max is not None else 100
        w_e_val = int(w_e) if w_e is not None else 5
        
        # Запобігання ValueError: range() arg 3 must not be zero
        if w_e_val == 0:
            w_e_val = 5
            print("Warning: Window expansion (w_e) was 0, set to default value 5")
        
        windows = list(range(w_s_val, w_max_val, w_e_val))
        
        # Створення нового n-граму та його обробка
        new_ngram = newNgram(data, w_s_val, L)
        
        # Визначаємо функцію для паралельної обробки вікон
        def process_window(w):
            if overlap_mode == "overlapping":
                return new_ngram.func(w)
            else:
                return new_ngram.func(w, overlap_mode=overlap_mode, min_window=w_s_val, window_expansion=w_e_val)
        
        # Паралельна обробка вікон (якщо їх достатньо багато)
        if len(windows) > 4:  # Паралелізуємо лише якщо є достатня кількість вікон
            with ThreadPoolExecutor(max_workers=min(4, len(windows))) as executor:
                list(executor.map(process_window, windows))
        else:
            # Послідовна обробка для малої кількості вікон
            for w in windows:
                process_window(w)
        
        # Оптимізоване створення списків для елементів та їх позицій
        temp_v = []
        temp_pos = []
        unique_items = set()  # Використовуємо множину для швидшого пошуку
        
        for i, ngram in enumerate(data):
            if ngram not in unique_items:
                unique_items.add(ngram)
                temp_v.append(ngram)
                temp_pos.append(i)
        
        # Використовуємо numpy масиви для ефективнішої обробки
        temp_pos_array = np.array(temp_pos, dtype=np.uint32)
        # Використовуємо перший елемент або "new_ngram" для розрахунку відстаней
        ngram_for_calc = temp_v[0] if temp_v else "new_ngram"
        new_ngram.dt = calculate_distance(temp_pos_array, L, condition, ngram_for_calc, min_dist_option)
        new_ngram.R = round(R(new_ngram.dt), 8)
        
        # Обробка помилок при підгонці кривої
        try:
            dfa_keys = list(new_ngram.dfa.keys())
            dfa_values = list(new_ngram.dfa.values())
            
            # Перевірка наявності достатньої кількості даних для підбору кривої
            if len(dfa_keys) < 2 or len(dfa_values) < 2:
                print("Недостатньо даних для підбору кривої")
                new_ngram.a = 1.0
                new_ngram.gamma = 0.5
                new_ngram.temp_dfa = [1.0] * (len(dfa_keys) if dfa_keys else 1)
                new_ngram.goodness = 0.0
            else:
                c, _ = curve_fit(fit, dfa_keys, dfa_values, method='lm', maxfev=5000)
                new_ngram.a = round(c[0], 8)
                new_ngram.gamma = round(c[1], 8)
                
                # Оптимізуємо обчислення temp_dfa
                new_ngram.temp_dfa = [fit(w, new_ngram.a, new_ngram.gamma) for w in dfa_keys]
                new_ngram.goodness = round(r2_score(dfa_values, new_ngram.temp_dfa), 8)
            
            # Звільняємо пам'ять від тимчасових змінних
            del dfa_keys, dfa_values
        except Exception as e:
            print(f"Помилка при підборі кривої: {e}")
            new_ngram.a = 1.0
            new_ngram.gamma = 0.5
            new_ngram.temp_dfa = []
            new_ngram.goodness = 0.0
        
        # Створення DataFrame для представлення результатів
        df = pd.DataFrame({
            'rank': [1],
            'ngram': ['new_ngram'],
            'F': [len(temp_pos)],
            'R': [new_ngram.R],
            'a': [new_ngram.a],
            'gamma': [new_ngram.gamma],
            'goodness': [new_ngram.goodness]
        })
        
        V = len(temp_v)
        
        end_time = time()
        execution_time = end_time - start
        
        # Підготовка даних для відображення
        df_table = df.to_dict("records")
        
        # Додаємо інформацію про розмір словника і час виконання
        vocab_info = f"Vocabulary: {V}"
        time_info = f"Time: {execution_time:.4f} s"
        
        # Звільняємо пам'ять від тимчасових змінних
        del temp_v, temp_pos, unique_items, temp_pos_array
        gc.collect()
        
        return (df_table, dash.no_update, {"display": "inline"}, {"display": "none"},
                dash.no_update, vocab_info, time_info, False)
    else:
        # Markov Chain обробка
        start = time()
        
        # Створення ланцюга Маркова та DataFrame
        make_markov_chain(data, order=n_size)
        df = make_dataframe(model, f_min)
        
        # Перевірка безпеки для None значень
        w_s_val = int(w_s) if w_s is not None else 5
        w_max_val = int(w_max) if w_max is not None else 100
        w_e_val = int(w_e) if w_e is not None else 5
        
        # Запобігання ValueError: range() arg 3 must not be zero
        if w_e_val == 0:
            w_e_val = 5
            print("Warning: Window expansion (w_e) was 0, set to default value 5")
        
        windows = list(range(w_s_val, w_max_val, w_e_val))
        
        # Функція для обробки окремого n-грама
        def process_ngram(ngram_data):
            ngram, index = ngram_data
            
            # Розрахунок відстаней
            dt = calculate_distance(np.array(model[ngram].pos, dtype=np.uint32), L, condition, ngram, min_dist_option)
            model[ngram].dt = dt
            
            # Обробка вікон для цього n-грама
            for wind in windows:
                if overlap_mode == "overlapping":
                    model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=w_s_val, overlap_mode=overlap_mode)
                else:
                    model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=w_s_val, 
                                                            overlap_mode=overlap_mode, min_window=w_s_val, window_expansion=w_e_val)
                
                model[ngram].fa[wind] = mse(model[ngram].counts[wind])
            
            # Підгонка кривої та обробка помилок
            try:
                ff = [*model[ngram].fa.values()]
                c, _ = curve_fit(fit, windows, ff, method='lm', maxfev=5000)
                
                a_val = c[0]
                gamma_val = c[1]
                temp_fa = [fit(w_val, a_val, gamma_val) for w_val in windows]
                
                # Зберігаємо результати в моделі
                model[ngram].a = a_val
                model[ngram].gamma = gamma_val
                model[ngram].temp_fa = temp_fa
                
                r_val = round(R(dt), 8)
                model[ngram].R = r_val
                
                return {
                    'ngram': ngram,
                    'a': round(a_val, 8),
                    'gamma': round(gamma_val, 8),
                    'error': round(r2_score(ff, temp_fa), 5),
                    'R': r_val
                }
            except Exception as e:
                print(f"Error in curve fitting for {ngram}: {e}")
                model[ngram].a = 0
                model[ngram].gamma = 0
                model[ngram].temp_fa = [0] * len(windows)
                r_val = round(R(dt), 8)
                model[ngram].R = r_val
                
                return {
                    'ngram': ngram,
                    'a': 0,
                    'gamma': 0,
                    'error': 0,
                    'R': r_val
                }
        
        # Підготовка даних для паралельної обробки
        ngram_items = [(ngram, i) for i, ngram in enumerate(df["ngram"])]
        
        # Визначаємо кількість робітників на основі кількості n-грамів
        max_workers = min(4, len(ngram_items))
        
        # Паралельна обробка для великої кількості n-грамів, інакше послідовна
        results = []
        if len(ngram_items) >= 4:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_ngram, ngram_items))
        else:
            results = [process_ngram(item) for item in ngram_items]
        
        # Витягуємо результати
        temp_a = [result['a'] for result in results]
        temp_gamma = [result['gamma'] for result in results]
        temp_error = [result['error'] for result in results]
        temp_R = [result['R'] for result in results]
        
        # Обробка n-грамів для відображення
        if n_size > 1:
            temp_ngram = []
            for ng in df['ngram']:
                if isinstance(ng, tuple):
                    temp_ngram.append(" ".join(ng))
                else:
                    temp_ngram.append(ng)
            df["ngram"] = temp_ngram
        
        # Оновлення DataFrame результатами
        df['R'] = temp_R
        df['gamma'] = temp_gamma
        df['a'] = temp_a
        df['goodness'] = temp_error
        df = df.sort_values(by="F", ascending=False)
        df['rank'] = range(1, len(temp_R) + 1)
        
        end_time = time()
        execution_time = end_time - start
        
        # Підготовка даних для відображення
        df_table = df.to_dict("records")
        
        # Додаємо інформацію про розмір словника і час виконання
        vocab_info = f"Vocabulary: {V}"
        time_info = f"Time: {execution_time:.4f} s"
        
        # Звільняємо пам'ять від тимчасових змінних
        del temp_gamma, temp_R, temp_error, temp_a, results, ngram_items
        gc.collect()
        
        return (df_table, dash.no_update, {"display": "inline"}, {"display": "none"},
                dash.no_update, vocab_info, time_info, False)


clikced_ngram = None


@app.callback([Output("graphs", "figure"), Output("fa", "figure"), ],
              [Input("dataframe", "active_tab"),
               Input("card-tabs", "active_tab"),
               Input("table", "active_cell"),
               Input("table", "page_current"),
               Input("table", "derived_virtual_selected_rows"),
               Input("table", "derived_virtual_indices"),
               Input("chain", "clickData"),
               Input("scale", "value"),
               Input("fa", "clickData"),
               Input("graphs", "clickData"),
               Input("w_max", "value"),
               Input("table", "derived_virtual_data")],
              [State("n_size", "value"),
               State("def", "value"), ])
def tab_content(active_tab2, active_tab1, active_cell, page_current, row_ids, ids, clicked_data, scale, fa_click,
                graph_click, w_max, derived_virtual_data, n,
                definition):
    if active_tab2 == "data_table":
        fig = go.Figure()
        fig1 = go.Figure()
        if active_tab1 == "tab2":
            if active_cell and derived_virtual_data:
                selected_row = derived_virtual_data[active_cell['row']]
                ngram = selected_row['ngram']
                if definition == "dynamic":
                    if fa_click:
                        if overlap_mode == "overlapping":
                            fig.add_trace(go.Bar(x=np.arange(w_s, L, w_s), y=new_ngram.count[fa_click["points"][0]["x"]],
                                                name="∑∆w"))
                        else:
                            bar_positions = []
                            k = 1
                            i = 0
                            ww = fa_click["points"][0]["x"]
                            while i < L - ww:
                                bar_positions.append(i)
                                shift = calc_non_overlapping_shift(k, w_s, w_e)
                                i += shift
                                k += 1
                            fig.add_trace(go.Bar(x=bar_positions, y=new_ngram.count[ww], name="∑∆w"))

                    if new_ngram is not None and hasattr(new_ngram, 'dfa') and new_ngram.dfa:
                        fig1.add_trace(
                            go.Scatter(x=[*new_ngram.dfa.keys()], y=[*new_ngram.dfa.values()], mode='markers', name="∆F"))

                        if hasattr(new_ngram, 'temp_dfa') and new_ngram.temp_dfa:
                            fig1.add_trace(go.Scatter(x=[*new_ngram.dfa.keys()], y=[*new_ngram.temp_dfa], name="fit=aw^b"))

                        fig1.update_xaxes(type=scale)
                        fig1.update_yaxes(type=scale)
                        fig1.update_layout(hovermode="x unified")

                    return fig, fig1

                if n > 1:
                    if ngram == 'new_ngram':
                        ngram = 'new_ngram'
                    else:
                        ngram = tuple(ngram.split())
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool, name="positions"))

                if fa_click:
                    if overlap_mode == "overlapping":
                        fig.add_trace(go.Bar(x=np.arange(w_s, L, w_s), y=model[ngram].counts[fa_click["points"][0]["x"]],
                                             name="∑∆w"))
                    else:
                        bar_positions = []
                        k = 1
                        i = 0
                        ww = fa_click["points"][0]["x"]
                        while i < L - ww:
                            bar_positions.append(i)
                            shift = calc_non_overlapping_shift(k, w_s, w_e)
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
            if active_cell and derived_virtual_data:
                selected_row = derived_virtual_data[active_cell['row']]
                ngram = selected_row['ngram']
                if definition == "dynamic":
                    if fa_click:
                        fig.add_trace(
                            go.Bar(x=np.arange(w_s, L, w_s), y=new_ngram.count[fa_click["points"][0]["x"]], name="∑∆w"))

                    if new_ngram is not None and hasattr(new_ngram, 'R') and hasattr(new_ngram, 'gamma'):
                        fig1.add_trace(go.Scatter(x=new_ngram.R, y=new_ngram.gamma, mode='markers', hover_data=["new_ngram"]))
                        fig1.update_xaxes(type=scale)
                        fig1.update_yaxes(type=scale)
                        fig1.update_layout(hovermode="x unified")

                    return fig, fig1

                if n > 1:
                    if ngram == 'new_ngram':
                        ngram = 'new_ngram'
                    else:
                        ngram = tuple(ngram.split())

                for data in df['ngram']:
                    if not isinstance(data, numbers.Number):
                        hover_data.append("".join(data))
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool, name="positions"))
                if fa_click:
                    ww = fa_click['points'][0]["x"]
                    if overlap_mode == "overlapping":
                        fig.add_trace(go.Bar(x=np.arange(ww, L, w_s), y=model[ngram].counts[ww], name="∑∆w"))
                    else:
                        bar_positions = []
                        k = 1
                        i = 0
                        while i < L - ww:
                            bar_positions.append(i)
                            shift = calc_non_overlapping_shift(k, w_s, w_e)
                            i += shift
                            k += 1
                        fig.add_trace(go.Bar(x=bar_positions, y=model[ngram].counts[ww], name="∑∆w"))

                fa_click = None
                if graph_click:
                    print(model[ngram].sums.keys())

                graph_click = None

                fig1.add_trace(go.Scatter(x=df["R"], y=df["gamma"], mode="markers", text=hover_data))
                fig1.add_trace(go.Scatter(x=[selected_row["R"]],
                                          y=[selected_row["gamma"]],
                                          mode="markers",
                                          text=' '.join(ngram) if isinstance(ngram, tuple) else ngram,
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
               State("w_min", "value"),
               State("w_s", "value"),
               State("w_e", "value"),
               State("w_max", "value"),
               State("f_min", "value"),
               State("condition", "value"),
               State("def", "value"),
               State("min_dist_option", "value"),
               State("overlap_mode", "value")])
def save(n, active_cell, page_current, ids, filename, n_size, w_min, w_s, w_e, w_max, fmin, opt, definition, min_dist_option, overlap_mode):
    if n is None:
        return dash.no_update
    if filename is None:
        return [html.Div(["No file selected to save"])]
    
    try:
        save_folder = pick_folder()
        if save_folder is None or save_folder == "":
            return dash.no_update
        file = filename
        global df, model, new_ngram

        # For dynamic mode, we want to save exactly what's shown in the table
        if definition == "dynamic":
            # Create DataFrame with the new_ngram row
            df_to_save = df.copy()  # This will include the new_ngram row
            
            output_filename = "{11}/{0} condition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8},min_dist={9},overlap={10}.xlsx".format(
                file, fmin, n_size, w_min, w_s, w_e, w_max, opt, definition, min_dist_option, overlap_mode, save_folder)
            
            # Ensure save directory exists
            os.makedirs(save_folder, exist_ok=True)
            
            # Save the main file with new_ngram data
            writer = pd.ExcelWriter(output_filename)
            df_to_save.to_excel(writer, index=False)
            writer.save()

            # If new_ngram exists and we have its details, save them too
            if new_ngram and hasattr(new_ngram, 'dfa'):
                details_filename = "{}/{} new_ngram_details.xlsx".format(save_folder, file)
                writer_details = pd.ExcelWriter(details_filename)
                df_details = pd.DataFrame()
                df_details["w"] = list(new_ngram.dfa.keys())
                df_details['∆F'] = list(new_ngram.dfa.values())
                df_details['fit=a*w^b'] = new_ngram.temp_dfa
                df_details.to_excel(writer_details, index=False)
                writer_details.save()
                return [html.Div([
                    "Saved main data to {}".format(output_filename),
                    html.Br(),
                    "Saved new_ngram details to {}".format(details_filename)
                ])]
            
            return [html.Div(["Saved data to {}".format(output_filename)])]
        
        else:  # For non-dynamic mode, keep existing logic
            df_copy = df.copy()
            df_copy = df_copy[df_copy.ngram != 'new_ngram']
            df_copy['rank'] = range(1, len(df_copy) + 1)

            if len(df_copy) > 0:
                df_copy['w'] = (df_copy['F']) / (df_copy['F'].sum())

                R_avg = df_copy['R'].mean()
                dR = df_copy['R'].std()
                Rw_avg = (df_copy['R'] * df_copy['w']).sum()
                dRw = np.sqrt((((df_copy['R'] - Rw_avg) ** 2) * df_copy['w']).sum())

                gamma_avg = df_copy['gamma'].mean()
                dgamma = df_copy['gamma'].std()
                gammaw_avg = (df_copy['gamma'] * df_copy['w']).sum()
                dgammaw = np.sqrt((((df_copy['gamma'] - gammaw_avg) ** 2) * df_copy['w']).sum())

                df_copy.loc[:, 'R_avg'] = None
                df_copy.loc[df_copy.index[0], 'R_avg'] = R_avg
                df_copy.loc[:, 'dR'] = None
                df_copy.loc[df_copy.index[0], 'dR'] = dR
                df_copy.loc[:, 'Rw_avg'] = None
                df_copy.loc[df_copy.index[0], 'Rw_avg'] = Rw_avg
                df_copy.loc[:, 'dRw'] = None
                df_copy.loc[df_copy.index[0], 'dRw'] = dRw

                df_copy.loc[:, 'gamma_avg'] = None
                df_copy.loc[df_copy.index[0], 'gamma_avg'] = gamma_avg
                df_copy.loc[:, 'dgamma'] = None
                df_copy.loc[df_copy.index[0], 'dgamma'] = dgamma
                df_copy.loc[:, 'gammaw_avg'] = None
                df_copy.loc[df_copy.index[0], 'gammaw_avg'] = gammaw_avg
                df_copy.loc[:, 'dgammaw'] = None
                df_copy.loc[df_copy.index[0], 'dgammaw'] = dgammaw

                df_copy = df_copy.drop(columns=['w'])

            output_filename = "{11}/{0} condition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8},min_dist={9},overlap={10}.xlsx".format(
                file, fmin, n_size, w_min, w_s, w_e, w_max, opt, definition, min_dist_option, overlap_mode, save_folder)
            
            os.makedirs(save_folder, exist_ok=True)
            
            writer = pd.ExcelWriter(output_filename)
            df_copy.to_excel(writer, index=False)
            writer.save()

            if active_cell:
                try:
                    row_index = active_cell['row']
                    if page_current is not None and page_current > 0:
                        row_index += page_current * 50

                    if ids is not None and row_index < len(ids):
                        selected_index = ids[row_index]
                        if selected_index < len(df):
                            ngram = df.iloc[selected_index]['ngram']
                            if ngram != 'new_ngram' and ngram in model:
                                details_filename = "{}/{} {}_details.xlsx".format(save_folder, file, ngram)
                                writer_details = pd.ExcelWriter(details_filename)
                                df1 = pd.DataFrame()
                                df1["w"] = list(model[ngram].fa.keys())
                                df1['∆F'] = list(model[ngram].fa.values())
                                df1['fit=a*w^b'] = model[ngram].temp_fa
                                df1.to_excel(writer_details, index=False)
                                writer_details.save()
                                return [html.Div([
                                    "Saved main data to {}".format(output_filename),
                                    html.Br(),
                                    "Saved details to {}".format(details_filename)
                                ])]
                except Exception as e:
                    print("Error saving ngram details: {}".format(e))

            return [html.Div(["Saved data to {}".format(output_filename)])]
            
    except Exception as e:
        return [html.Div(["Error saving data: {}".format(str(e))])]


def pick_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_selected = filedialog.askdirectory()
    root.destroy() 

    if folder_selected:
        return folder_selected
    else:
        return None

@app.callback(
    [Output('isComputerTextOutput', 'children'),
    Output('file-selector', 'value')],
    Input('computerTextRadio', 'value')
)
def update_isComputerText(selected_value):
    global isComputerText
    isComputerText = selected_value
    return "", None

def guess_lexer_or_text(code, file_path):
    try:
        lexer = guess_lexer_for_filename(file_path, code)
    except Exception:
        lexer = get_lexer_by_name("text")
    return lexer

def tokenize_code(code, lexer):
    comm, whitespace, other = 0, 0, 0
    tokens = lex(code, lexer)
    token_list = []
    for token_type, token_value in tokens:
        if token_type in Token.Comment:
            comm += 1
            file_text = re.sub(r'\n+', '\n', token_value)
            file_text = re.sub(r'\n\s\s', '\n', file_text)
            file_text = re.sub(r'﻿', '', file_text)
            file_text = re.sub(r'--', ' -', file_text)
            processor = NgrammProcessor()
            processor.preprocess(file_text)
            for x in processor.get_words():
                token_list.append(x)
            del processor
            del file_text
        elif token_type in Token.Text or token_type in Token.Whitespace:
            whitespace += 1
            pass 
        else: 
            other += 1
            token_list.append(token_value)
    print("\n comm", comm,"\n whitespace", whitespace,"\n other", other)
    return token_list

# import webbrowser # Commented out as it might cause issues if run non-interactively

if __name__ == "__main__":
    webbrowser.open_new("http://127.0.0.1:8050/") # Автоматично відкриває браузер
    # Replace app.run() with the older style Flask server run for Dash < 2.0
    app.server.run(host='0.0.0.0', port=8050, debug=False)

# Add callback to toggle batch window settings
@app.callback(
    Output("batch_custom_controls", "is_open"),
    [Input("batch_window_mode", "value")]
)
def toggle_batch_window_controls(mode):
    return mode in ["ui", "auto"]