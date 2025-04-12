import numbers
from typing import List, Tuple, Optional

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


def make_dataframe(model, fmin=3):
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
    model = dict()
    L = len(data) - order
    model['new_ngram'] = Ngram()
    model['new_ngram'].bool = np.zeros(L, dtype=np.uint8)
    model['new_ngram'].pos = []
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
        for i in range(L):
            if data[i] in model:  # Приєднуємо до вже існуючого розподілу
                model[data[i]].update([data[i + order]])
                model[data[i]].pos.append(i + order)
                try:
                    model[data[i]].bool[i] = 1
                except Exception:
                    print('Wait for symbol calculation')
            else:
                model[data[i]] = Ngram([data[i + order]])
                model[data[i]].pos = []
                model[data[i]].pos.append(i + order)
                model[data[i]].bool = np.zeros(L, dtype=np.uint8)
                model[data[i]].bool[i] = 1

                model['new_ngram'].bool[i] = 1
                model['new_ngram'].pos.append(i + order)

            # Connect the last word with the first one
        if data[L] in model:
            model[data[L]].update({data[0]: 1})
        else:
            model[data[L]] = Ngram({data[0]: 1})
            model[data[L]].pos = []
            model[data[L]].pos.append(L + order)
            model[data[L]].bool = np.zeros(L, dtype=np.uint8)
            model[data[L]].bool[L-1] = 1

            # Connect the first word with the last one
        if data[0] in model:
            model[data[0]].update({data[L]: 1})
        else:
            model[data[0]] = {data[L]: 1}
    V = len(model)


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
    if number_of_pos == 1:
        return positions
    dt = np.empty(number_of_pos - 1, dtype=np.uint32)
    for i in range(number_of_pos - 1):
        dt[i] = positions[i + 1] - positions[i]
        if min_dist == 0:
            dt[i] = dt[i] - 1
    return dt


@jit(nopython=True)
def obc(positions, L, min_dist=1):
    number_of_pos = len(positions)
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
    if len(x) == 1:
        return 0.0
    t = np.mean(x)
    ts = np.std(x)
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

#@njit(fastmath=True)
def make_windows(x, wi, l, wsh, overlap_mode="overlapping", min_window=None, window_expansion=None):
    sums = []
    if overlap_mode == "overlapping":
        # Стандартний режим з фіксованим зміщенням
        for i in range(0, l - wi, wsh):
            sums.append(np.sum(x[i:i + wi]))
    else:  # non-overlapping режим
        # Перевіряємо значення параметрів і встановлюємо значення за замовчуванням якщо None
        """
        здається, це не зовсім той алгоритм, тут виходить, що лівий край вікна зсовується далі від правого краю попереднього вікна,
        а не стає впритик

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
            print(i)
            k += 1"""

        for i in range(0, l - wi, wi):
            sums.append(np.sum(x[i:i + wi]))
    
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


def prepare_data(data: str, n: int, split: str) -> List:
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
    
    # Попередня обробка тексту
    data = re.sub(r'\n+', '\n', data)
    data = re.sub(r'\n\s\s', '\n', data)
    data = re.sub(r'﻿', '', data)
    
    # Для n=1 (одиничні елементи)
    if n == 1:
        if split == "word":
            # Обробка тексту для слів
            data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor()
            processor.preprocess(data)
            temp = processor.get_words()
            L = len(temp)
            return temp
            
        elif split == 'letter':
            # Обробка для літер
            data = remove_punctuation(data)
            temp_data = []
            for i in data:
                for j in i:
                    if not is_valid_letter(j):
                        temp_data.append(j)
            L = len(temp_data)
            return temp_data
            
        elif split == 'symbol':
            # Обробка для символів
            temp_data = []
            for i in data:
                for j in i:
                    if j == " " or j == "\n" or j == "\ufeff":
                        temp_data.append("space")
                    else:
                        temp_data.append(j.lower())
            L = len(temp_data)
            return temp_data
    
    # Для n>1 (n-грами)
    else:
        temp_data = []
        
        if split == "word":
            # Обробка для n-грам слів
            data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor()
            processor.preprocess(data)
            data = processor.get_words()
            L = len(data)
            
            for i in range(L - n + 1):
                window = tuple(data[i: i + n])
                temp_data.append(window)
                
        elif split == "letter":
            # Обробка для n-грам літер
            data = remove_punctuation(data.split())
            data = remove_empty_strings(data)
            letter_data = []
            
            for word in data:
                for char in word:
                    if not is_valid_letter(char):
                        letter_data.append(char)
                        
            L = len(letter_data)
            
            for i in range(L - n + 1):
                window = tuple(letter_data[i: i + n])
                temp_data.append(window)
                
        elif split == 'symbol':
            # Обробка для n-грам символів
            symbol_data = []
            
            for i in data:
                for j in i:
                    if j == " " or j == "\n" or j == "\ufeff":
                        symbol_data.append("space")
                    else:
                        symbol_data.append(j.lower())
                        
            L = len(symbol_data)
            
            for i in range(L - n + 1):
                window = tuple(symbol_data[i: i + n])
                temp_data.append(window)
                
        return temp_data


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
        self.count[w], self.dfa[w] = dfa(self.data, (w, self.wh, self.l), overlap_mode, min_window, window_expansion)


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

                                # BATCH PROCESSING SECTION
                                html.Div([
                                    html.H6("Batch Processing", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    
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
                                                {"name": "gamma_avg", "id": "g_avg"},
                                                {"name": "dgamma", "id": "dg"},
                                                {"name": "gammaw_avg", "id": "gw_avg"},
                                                {"name": "dgammaw", "id": "dgw"}
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
    )
])
from dash.dependencies import Input, Output, State

app.layout = layout1
df = None
g = None
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx

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
    Перевіряє, чи є символ допустимою літерою для аналізу.
    
    Args:
        char: Символ для перевірки
        
    Returns:
        bool: True, якщо символ НЕ є допустимою літерою (тобто, має бути пропущений),
              False, якщо символ Є допустимою літерою (тобто, має бути збережений)
              
    Note:
        Функція має зворотну логіку: повертає True для символів, які слід ПРОПУСТИТИ,
        і False для символів, які слід ВКЛЮЧИТИ в аналіз.
    """
    invalid_characters = [' ', '\n', '\ufeff', '°', '"', '„', '–']
    return is_number(char) or char in invalid_characters


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
            print(f"✗ Error processing {filename}: {str(e)}")
            error_count += 1
    
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
    global L, data, length_updated
    
    if selected_filename is None or selected_filename not in uploaded_files:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Get the file content
    file = uploaded_files[selected_filename]
    
    length_updated = False
    
    if definition == "dynamic":
        data = prepare_data(file, n, split)
        w_max = int(L / 10)
        w_min = int(w_max / 10)
    else:
        temp = []
        if split == "letter":
            file = re.sub(r'	', '', file)
            data = remove_punctuation(file)
            for word in data:
                for i in word:
                    if is_valid_letter(i):
                        continue
                    temp.append(i)
            data = temp
        if split == "symbol":
            data = file
            data = re.sub(r'	', '', file)
            data = re.sub(r'\n+', '\n', data)
            data = re.sub(r'\n\s\s', '\n', data)
            data = re.sub(r'﻿', '', data)
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

        if split == "word":
            file = re.sub(r'\n+', '\n', file)
            file = re.sub(r'\n\s\s', '\n', file)
            file = re.sub(r'﻿', '', file)
            file = re.sub(r'--', ' -', file)

            processor = NgrammProcessor()
            # обробка тексту
            processor.preprocess(file)

            # Отримання слів у тексті
            data = processor.get_words()

        L = len(data)
        w_max = int(L / 20)
        w_min = int(w_max / 20)
        length_updated = True
    
    # Show all three lengths for the selected file
    lengths_str = "Length: {} ({}s) | ".format(L, split)
    for split_type in ['word', 'symbol', 'letter']:
        if split_type != split:
            lengths_str += "{}s: {} | ".format(split_type, file_lengths[selected_filename][split_type])
    lengths_str = lengths_str.rstrip(" | ")
    
    return [lengths_str], w_min, w_min, w_min, w_max


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
    global batch_results, uploaded_files, file_lengths
    
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
    
    # Process each file
    for idx, (filename, file_content) in enumerate(list(uploaded_files.items()), 1):
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
        
        # Prepare data
        global L, data, length_updated, model, V, df, new_ngram
        
        length_updated = False
        
        if definition == "dynamic":
            data = prepare_data(file_content, n_size, split)
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
        current_df['gamma'] = temp_gamma
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
            
            gamma_avg = df_filtered['gamma'].mean()
            dgamma = df_filtered['gamma'].std()
            gammaw_avg = (df_filtered['gamma'] * df_filtered['w']).sum()
            dgammaw = np.sqrt((((df_filtered['gamma'] - gammaw_avg) ** 2) * df_filtered['w']).sum())
        else:
            R_avg = dR = Rw_avg = dRw = gamma_avg = dgamma = gammaw_avg = dgammaw = 0
        
        end_time = time()
        processing_time = end_time - start_time
        
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
            "wm_val": wm_val
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
        
        # Calculate standard deviations
        stds = {col: round(np.std([result[col] for result in batch_results]), 8) for col in numeric_columns}
        stds['no'] = 'StdDev'
        stds['filename'] = 'Std. Dev.'
        
        # Add summary rows
        batch_results.append(means)
        batch_results.append(stds)
    
    return batch_results, {"display": "block"}

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
        {"name": "gamma_avg", "id": "g_avg"},
        {"name": "dgamma", "id": "dg"},
        {"name": "gammaw_avg", "id": "gw_avg"},
        {"name": "dgammaw", "id": "dgw"},
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
    global model, L, V, df, new_ngram
    if n is None or dataframe is None:
        return (dash.no_update, dash.no_update, {"display": "none"}, {"display": "none"},
                dash.no_update, dash.no_update, dash.no_update,
                dash.no_update)
    # Завжди використовуємо data_table, оскільки вкладку MarkovChain було видалено
    if definition == "dynamic":
        start = time()
        # Add safety checks for None values
        w_s_val = w_s if w_s is not None else 5
        w_max_val = w_max if w_max is not None else 100
        w_e_val = w_e if w_e is not None else 5
        
        windows = list(range(w_s_val, w_max_val, w_e_val))
        # 2. create newNgram

        new_ngram = newNgram(data, w_s_val, L)
        for w in windows:
            if overlap_mode == "overlapping":
                new_ngram.func(w)
            else:
                new_ngram.func(w, overlap_mode=overlap_mode, min_window=w_s_val, window_expansion=w_e_val)
        # calculate coefs
        temp_v = []
        temp_pos = []
        for i, ngram in enumerate(data):
            if ngram not in temp_v:
                temp_v.append(ngram)
                temp_pos.append(i)
        new_ngram.dt = calculate_distance(np.array(temp_pos, dtype=np.uint8), L, condition, ngram, min_dist_option)
        new_ngram.R = round(R(new_ngram.dt), 8)
        c, _ = curve_fit(fit, list(new_ngram.dfa.keys()), list(new_ngram.dfa.values()), method='lm', maxfev=5000)
        new_ngram.a = round(c[0], 8)
        new_ngram.gamma = round(c[1], 8)
        new_ngram.temp_dfa = []
        for w in new_ngram.dfa.keys():
            new_ngram.temp_dfa.append(fit(w, new_ngram.a, new_ngram.gamma))
        new_ngram.goodness = round(r2_score(list(new_ngram.dfa.values()), new_ngram.temp_dfa), 8)
        df = pd.DataFrame()
        df['rank'] = [1]
        df['ngram'] = ['new_ngram']
        df["F"] = [len(temp_pos)]
        df['R'] = [new_ngram.R]
        df["a"] = [new_ngram.a]
        df["gamma"] = [new_ngram.gamma]
        df['goodness'] = [new_ngram.goodness]
        V = len(temp_v)

    else:
        ###  MAKE MARKOV CHAIN ####
        start = time()
        make_markov_chain(data, order=n_size)
        df = make_dataframe(model, f_min)

        for index, ngram in enumerate(df['ngram']):
            model[ngram].dt = calculate_distance(np.array(model[ngram].pos, dtype=np.uint32), L, condition, ngram, min_dist_option)

        def func(wind):
            # Safety checks for None values
            w_s_val = w_s if w_s is not None else 5
            w_e_val = w_e if w_e is not None else 5
            
            if overlap_mode == "overlapping":
                model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=w_s_val, overlap_mode=overlap_mode)
            else:
                # Для non-overlapping mode
                model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=w_s_val, 
                                                        overlap_mode=overlap_mode, min_window=w_s_val, window_expansion=w_e_val)
            
            model[ngram].fa[wind] = mse(model[ngram].counts[wind])

        # Add safety checks for None values
        w_s_val = w_s if w_s is not None else 5
        w_max_val = w_max if w_max is not None else 100
        w_e_val = w_e if w_e is not None else 5
        
        windows = list(range(w_s_val, w_max_val, w_e_val))

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
        for i, ngram in enumerate(df["ngram"]):

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

        if n_size > 1:
            # HERE REMOVE
            temp_ngram.append("new_ngram")
            df["ngram"] = temp_ngram

        #     NOTE через ці змінні в циклі які оновлюються по порядку і потім записуються напряму ж в колонку,
        #     неможливо просто так розділити
        df['R'] = temp_R
        df['gamma'] = temp_gamma
        df['a'] = temp_a
        df['goodness'] = temp_error
        df = df.sort_values(by="F", ascending=False)
        df['rank'] = range(1, len(temp_R) + 1)
        df = df.set_index(pd.Index(np.arange(len(df))))

    voc = str(V)
    voc = int(voc) - 1
    # HERE V-1

    return [df.to_dict(orient='records'), dash.no_update, {"display": "inline"}, {"display": "none"},
            dash.no_update,
            # NOTE повернення додаткових 8-ми значень на фронт-енд
            ["Vocabulary: " + str(voc)], ["Time:" + str(round(time() - start, 4))],
             dash.no_update
            ]


clikced_ngram = None


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
               Input("w_max", "value")],
              [State("n_size", "value"),
               State("def", "value"), ])
def tab_content(active_tab2, active_tab1, active_cell, page_current, row_ids, ids, clicked_data, scale, fa_click,
                graph_click, w_max, n,
                definition):
    # Тільки для вкладки DataTable, оскільки MarkovChain було видалено
    if active_tab2 == "data_table":
        fig = go.Figure()
        fig1 = go.Figure()
        if active_tab1 == "tab2":
            if active_cell:
                if definition == "dynamic":
                    ## add bar
                    if fa_click:
                        if overlap_mode == "overlapping":
                            fig.add_trace(go.Bar(x=np.arange(w_s, L, w_s), y=new_ngram.count[fa_click["points"][0]["x"]],
                                                name="∑∆w"))
                        else:
                            # Для non-overlapping режиму потрібно розрахувати положення барів
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
                        fig.add_trace(go.Bar(x=np.arange(w_s, L, w_s), y=model[ngram].counts[fa_click["points"][0]["x"]],
                                             name="∑∆w"))
                    else:
                        # Для non-overlapping режиму потрібно розрахувати положення барів
                        bar_positions = []
                        k = 1
                        i = 0
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
            if active_cell:
                if definition == "dynamic":
                    if fa_click:
                        fig.add_trace(
                            go.Bar(x=np.arange(w_s, L, w_s), y=new_ngram.count[fa_click["points"][0]["x"]], name="∑∆w"))

                    fig1.add_trace(go.Scatter(x=new_ngram.R, y=new_ngram.gamma, mode='markers', hover_data=["new_ngram"]))
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
                        fig.add_trace(go.Bar(x=np.arange(ww, L, w_s), y=model[ngram].counts[ww], name="∑∆w"))
                    else:
                        # Для non-overlapping режиму потрібно розрахувати положення барів
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
                # fig1.add_trace(go.Scatter(x=[df['R'][active_cell['row']]],
                fig1.add_trace(go.Scatter(x=[df['R'][ids[active_cell['row']]]],
                                          # y=[df["b"][active_cell['row']]],
                                          y=[df["gamma"][ids[active_cell['row']]]],
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

            gamma_avg = df_copy['gamma'].mean()
            dgamma = df_copy['gamma'].std()
            gammaw_avg = (df_copy['gamma'] * df_copy['w']).sum()
            dgammaw = np.sqrt((((df_copy['gamma'] - gammaw_avg) ** 2) * df_copy['w']).sum())

            # Assign calculated values using .loc to avoid SettingWithCopyWarning
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

            # Remove temporary 'w' column if not needed in the final output
            df_copy = df_copy.drop(columns=['w'])

        else:
             # Handle empty dataframe case if necessary
             # Maybe return an alert or log a message
             print("Warning: DataFrame is empty after filtering 'new_ngram'. Cannot save stats.")
             # Decide how to handle df_copy columns if it's empty
             pass


        if definition == "dynamic":
            output_filename = "saved_data/{0} condition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8},min_dist={9},overlap={10}.xlsx".format(file, fmin, n_size, w_s, w_s, w_e, w_max, opt, definition, min_dist_option, overlap_mode)
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
                file, fmin, n_size, w_s, w_s, w_e, w_max, opt, definition, min_dist_option, overlap_mode
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
