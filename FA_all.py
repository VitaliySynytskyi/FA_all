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
from dash import dcc, html, dash_table as dt
from os import listdir
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import re


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

    print(f"Execution time: {execution_time} seconds")
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
            "ƒ": np.empty(len(filtered_data), dtype=np.dtype(int))}

    for i, ngram in enumerate(filtered_data):
        data["ngram"].append(ngram)

        if ngram == "new_ngram":
            data['ƒ'][i] = sum(model[ngram].bool)
            continue
        data["ƒ"][i] = len(model[ngram].pos)

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
            model[data[L]] = {data[0]: 1}

            # Connect the first word with the last one
        if data[0] in model:
            model[data[0]].update({data[L]: 1})
        else:
            model[data[0]] = {data[L]: 1}
    V = len(model)


def calculate_distance(positions, L, option, ngram):
    if option == "no":
        return nbc(positions)
    if option == "ordinary":
        return obc(positions, L)
    if option == "periodic":
        return pbc(positions, L, ngram)


@jit(nopython=True)
def nbc(positions):
    number_of_pos = len(positions)
    if number_of_pos == 1:
        return positions
    dt = np.empty(number_of_pos - 1, dtype=np.uint32)
    for i in range(number_of_pos - 1):
        dt[i] = positions[i + 1] - positions[i]
    return dt


@jit(nopython=True)
def obc(positions, L):
    number_of_pos = len(positions)
    dt = np.empty(number_of_pos + 1, dtype=np.uint32)
    dt[0] = positions[0]
    for i in range(number_of_pos - 1):
        dt[i + 1] = positions[i + 1] - positions[i]
    dt[-1] = L - positions[-1]
    return dt


@jit(nopython=True)
def pbc(positions, L, test):
    number_of_pos = len(positions)
    dt = np.zeros(number_of_pos, dtype=np.uint32)
    for i in range(number_of_pos - 1):
        dt[i] = positions[i + 1] - positions[i]
    dt[-1] = L - positions[-1] + positions[0]
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
def make_windows(x, wi, l, wsh):
    sums = []
    for i in range(0, l - wi, wsh):
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


def prepere_data(data, n, split):
    global L
    if n is None:
        return dash.no_update
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
            data = temp
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
            # data = data.split()
            # data = remove_empty_strings(data)
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
            for i in range(L):
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
            for i in range(L):
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
            for i in range(L):
                window = tuple(data[i:i + n])
                temp_data.append(window)
            return temp_data


# @jit(nopython=True)
def dfa(data, args):
    wi, wh, l = args
    count = np.empty(len(range(wi, l, wh)), dtype=np.uint8)
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
        return count, mse(count)


class newNgram():
    def __init__(self, data, wh, l):
        self.data = data
        self.count = {}
        self.dfa = {}
        self.wh, self.l = wh, l

    def func(self, w):
        self.count[w], self.dfa[w] = dfa(self.data, (w, self.wh, self.l))


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])



corpuses = listdir("corpus/")
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
                        dbc.CardHeader("Configuration:"),
                        dbc.CardBody(
                            [
                                html.Label("Choose file:"),
                                html.Div(
                                    [
                                        dcc.Dropdown(id="corpus", options=[{"label": i, "value": i} for i in corpuses]),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Size of ngram"),
                                                dbc.Input(id="n_size", type="number", value=1),
                                            ], size="md", className="config"
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
                                                    value="word"
                                                )
                                            ], size="md", className="config"
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
                                                    value="no"
                                                ),
                                                dbc.InputGroupText("Boundary Condition:"),
                                            ], size="md", className="config"
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("filter"),
                                                dbc.Input(id="f_min", type="number", value=0)
                                            ]
                                        ),
                                        html.Label("Sliding window"),

                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Min window"),
                                                dbc.Input(id="w", type="number"),
                                            ], size="md", className="window"
                                        ),

                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Window shift"),
                                                dbc.Input(id="wh", type="number"),
                                            ], size="md", className="window"
                                        ),

                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Window expansion"),
                                                dbc.Input(id="we", type="number"),
                                            ], size="md", className="window"
                                        ),

                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Max window"),
                                                dbc.Input(id="wm", type="number"),
                                            ], size="md", className="window"
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
                                                dbc.InputGroupText("Definition")
                                            ], size="md", className="window"
                                        ),

                                        # dbc.Input(placeholder="size of ngram",type="number"),
                                        # html.H6("Size of ngram:"),
                                        # dcc.Slider(id="n_size",min=1,max=9,value=1,marks={i:"{}".format(i)for i in range(1,10)}),
                                        # html.H6("Split by:"),
                                        # dcc.RadioItems(id='split',options=[{"label":"symbol","value":"symbol"},{"label":"word","value":"word"}],value="word"),
                                        # html.H6("Boundary Condition:"),
                                        # dcc.RadioItems(id='condition',options=[{"label":"no","value":"no"},{"label":"periodic","value":"periodic"},{"label":"ordinary","value":"ordinary"}],value="words"),
                                        html.Br(),
                                        dbc.Button("Analyze", id="chain_button", color="primary", className="w-100", disabled=analyze_visible),

                                        dbc.Button("Save data", id="save", color="danger", className="w-100"),
                                        html.Div(id="temp_seve",
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
                                        dbc.Tab(label="DataTable", tab_id="data_table"),
                                        dbc.Tab(label="MarkovChain", tab_id="markov_chain")
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
                                             children=[dbc.Spinner(dt.DataTable(
                                                 id="table",
                                                 columns=[{"name": i, "id": i} for i in
                                                          ['rank', "ngram", "ƒ", "R", "a", "b", "goodness"]],
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
                                                              "minHeight": "400px"}
                                             ))]),
                                    html.Div(id="box_chain",
                                             style={"display": "none"},
                                             children=[dbc.Spinner(dcc.Graph(id="chain", style={"height": "400px"}))]),

                                    dbc.CardHeader("Characteristics", style={"padding": "5px 20px"}),
                                    # here add chars
                                    dbc.CardBody(
                                        dbc.Row([
                                            # NOTE додала вивід 8-ми значень з екселю а також кнопку для копіювання всього
                                            dbc.Col([
                                                html.Div(["Length: "], id="l"),
                                                html.Div(["Vocabulary: "], id="v"),
                                                html.Div(["Time: "], id="t")

                                            ]),
                                            dbc.Col([
                                                html.Div([""], id="new_output1", n_clicks=0),
                                                html.Div([""], id="new_output2", n_clicks=0),
                                            ]),
                                            dbc.Col([
                                                html.Div([""], id="new_output3", n_clicks=0),
                                                html.Div([""], id="new_output4", n_clicks=0),
                                            ]),
                                            dbc.Col([
                                                html.Div([""], id="new_output5", n_clicks=0),
                                                html.Div([""], id="new_output6", n_clicks=0),
                                            ]),
                                            dbc.Col([
                                                html.Div([""], id="new_output7", n_clicks=0),
                                                html.Div([""], id="new_output8", n_clicks=0),
                                                html.Div([""], id="copy_all", n_clicks=0, style={"fontWeight": "bold", "color": "blue", "cursor": "pointer"})
                                            ])
                                        ])
                                    )

                                ]
                            )
                        ], style={"padding": "0", "margin-right": "0px", "margin-top": "10px", "height": "650px"}),
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
                                    dbc.Tab(label="distribution", tab_id="tab1"),
                                ],
                                id='card-tabs1',
                                active_tab="tab1"
                            )
                        ),
                        dbc.CardBody([
                            dcc.Graph(id="graphs")

                        ])

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
                                    dbc.Tab(label="flunctuacion", tab_id="tab2"),
                                    dbc.Tab(label="alpha/R", tab_id="tab3")
                                ],
                                id='card-tabs',
                                active_tab="tab2"
                            )
                        ),
                        dbc.CardBody([
                            dcc.RadioItems(
                                id="scale",
                                options=[
                                    {"label": "linear", "value": "linear"},
                                    {"label": "log", "value": "log"}
                                ],
                                value="linear"

                            ),
                            dcc.Graph(id="fa")

                        ])

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
    invalid_characters = [' ', '\n', '\ufeff', '°', '“', '„', '–']
    if is_number(char) or char in invalid_characters:
        return True
    else:
        return False


length_updated = False


@app.callback([Output("w", "value"),
               Output("wh", "value"),
               Output("we", "value"),
               Output("wm", "value"),
               Output("l", "children")],
              [Input("corpus", "value"), Input("split", "value"),
               Input("def", "value"), Input("n_size", "value")])
def calc_window(corpus, split, definition, n):
    if corpus is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    global L, data, length_updated
    length_updated = False
    with open("corpus/" + corpus, encoding='utf-8') as f:
        file = f.read()
    if definition == "dynamic":
        data = prepere_data(file, n, split)
        wm = int(L / 10)
        w = int(wm / 10)
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
            # NOTE попередня версія розрахування слів
            # data = remove_punctuation(file)
            # data = remove_punctuation_for_words(file)
            # data = data.split()
            # data = remove_empty_strings(data)

            processor = NgrammProcessor()
            # обробка тексту
            processor.preprocess(file)

            # Отримання слів у тексті
            data = processor.get_words()

        L = len(data)
        wm = int(L / 20)
        w = int(wm / 20)
        length_updated = True
    return [w, w, w, wm, ["Lenght: " + str(L)]]


def remove_empty_strings(arr):
    return [item for item in arr if item != '\ufeff']


new_ngram = None



# @app.callback(
#     Output('chain_button', 'disabled'),
#     Input("split", "value")
# )
# def update_output(selected_value):
#     global length_updated
#     if length_updated:
#         length_updated = False
#         return False
#     else:
#         return True

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
              [State("corpus", "value"),
               State("n_size", "value"),
               State("split", "value"),
               State("condition", "value"),
               State("f_min", "value"),
               State("w", "value"),
               State("wh", "value"),
               State("we", "value"),
               State("wm", "value"),
               State("def", "value")
               ])
def update_table(n, dataframe, corpus, n_size, split, condition, f_min, w, wh, we, wm, definition):
    global length_updated

    if n is None:
        # кількість dash.no_update відповідає кількостю output значень в методі контроллера
        return (dash.no_update, dash.no_update, {"display": 'inline'}, {
            "display": "none"}, dash.no_update, dash.no_update, dash.no_update,
                 dash.no_update,
                )

    if not length_updated:
        # кількість dash.no_update відповідає кількостю output значень в методі контроллера
        return (dash.no_update, dash.no_update, {"display": 'inline'}, {
            "display": "none"}, dash.no_update, dash.no_update, dash.no_update,
                 True
                )

    # add alert corpus if not selected
    if corpus is None:
        return (dash.no_update, dash.no_update, {"display": "inline"}, {"display": "none"}, dbc.Alert(
            "Please choose corpus", color="danger", duration=2000,
            dismissable=False), dash.no_update, dash.no_update,
                 dash.no_update)

    global data, L, V, model, ngram, df, g, new_ngram

    if dataframe == "markov_chain":
        ## make markov chain graph ###
        g = nx.MultiGraph()
        temp = {}
        for ngram in df['ngram']:
            if n_size > 1:
                ngram = tuple(ngram.split())

            g.add_node(ngram)
            temp[ngram[0]] = ngram

        for node in g.nodes():
            if node[0] == "new_ngram":
                node = 'new_ngram'
            for i in model[node]:
                if i in temp:
                    g.add_edge(node, temp[i], weight=model[node][i])

        pos = nx.spring_layout(g)

        edge_x = []
        edge_y = []
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
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        node_adjacencies = []
        node_text = []

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

        return (dash.no_update, fig, {"display": "none"}, {
            "display": 'inline'}, dash.no_update, dash.no_update, dash.no_update,
                 dash.no_update)
    if dataframe == "data_table":
        if definition == "dynamic":
            start = time()
            windows = list(range(w, wm, we))
            # 2. create newNgram

            new_ngram = newNgram(data, wh, L)
            for w in windows:
                new_ngram.func(w)
            # calculate coefs
            temp_v = []
            temp_pos = []
            for i, ngram in enumerate(data):
                if ngram not in temp_v:
                    temp_v.append(ngram)
                    temp_pos.append(i)
            new_ngram.dt = calculate_distance(np.array(temp_pos, dtype=np.uint8), L, condition, ngram)
            new_ngram.R = round(R(new_ngram.dt), 8)
            c, _ = curve_fit(fit, [*new_ngram.dfa.keys()], [*new_ngram.dfa.values()], method='lm', maxfev=5000)
            new_ngram.a = round(c[0], 8)
            new_ngram.b = round(c[1], 8)
            new_ngram.temp_dfa = []
            for w in new_ngram.dfa.keys():
                new_ngram.temp_dfa.append(fit(w, new_ngram.a, new_ngram.b))
            new_ngram.goodness = round(r2_score([*new_ngram.dfa.values()], new_ngram.temp_dfa), 8)
            df = pd.DataFrame()
            df['rank'] = [1]
            df['ngram'] = ['new_ngram']
            df["ƒ"] = [len(temp_pos)]
            df['R'] = [new_ngram.R]
            df["a"] = [new_ngram.a]
            df["b"] = [new_ngram.b]
            df['goodness'] = [new_ngram.goodness]
            V = len(temp_v)

        else:
            ###  MAKE MARKOV CHAIN ####
            start = time()
            make_markov_chain(data, order=n_size)
            df = make_dataframe(model, f_min)

            for index, ngram in enumerate(df['ngram']):
                model[ngram].dt = calculate_distance(np.array(model[ngram].pos, dtype=np.uint32), L, condition, ngram)

            def func(wind):
                model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=wh)
                model[ngram].fa[wind] = mse(model[ngram].counts[wind])

            windows = list(range(w, wm, we))

            temp_b = []
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
                model[ngram].b = c[1]
                for w in windows:
                    model[ngram].temp_fa.append(fit(w, c[0], c[1]))
                temp_error.append(round(r2_score(ff, model[ngram].temp_fa), 5))
                temp_b.append(round(c[1], 8))
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
            df['b'] = temp_b
            df['a'] = temp_a
            df['goodness'] = temp_error
            df = df.sort_values(by="ƒ", ascending=False)
            df['rank'] = range(1, len(temp_R) + 1)
            df = df.set_index(pd.Index(np.arange(len(df))))

        voc = str(V)
        voc = int(voc) - 1
        # HERE V-1

        # copy_df = df.copy()
        # copy_df = copy_df[copy_df.ngram != 'new_ngram']
        # copy_df['rank'] = copy_df['rank'] - 1
        #
        # copy_df['w'] = (copy_df['ƒ']) / (copy_df['ƒ'].sum())
        #
        # copy_df['R_avg'] = copy_df['R'].mean()
        # R_avg = copy_df.iloc[0, copy_df.columns.get_loc('R_avg')]
        # del copy_df['R_avg']
        #
        # copy_df['dR'] = copy_df['R'].std()
        # dR = copy_df.iloc[0, copy_df.columns.get_loc('dR')]
        # del copy_df['dR']
        #
        # copy_df['Rw'] = (copy_df['R']) * (copy_df['w'])
        #
        # copy_df['Rw_avg'] = copy_df['Rw'].sum()
        # Rw_avg = copy_df.iloc[0, copy_df.columns.get_loc('Rw_avg')]
        # del copy_df['Rw_avg']
        #
        # copy_df['dRw'] = np.sqrt((((copy_df['R'] - Rw_avg) ** 2) * (copy_df['w'])).sum())
        # dRw = copy_df.iloc[0, copy_df.columns.get_loc('dRw')]
        # del copy_df['dRw']
        # del copy_df['Rw']
        #
        # copy_df['b_avg'] = copy_df['b'].mean()
        # b_avg = copy_df.iloc[0, copy_df.columns.get_loc('b_avg')]
        # del copy_df['b_avg']
        #
        # copy_df['db'] = copy_df['b'].std()
        # db = copy_df.iloc[0, copy_df.columns.get_loc('db')]
        # del copy_df['db']
        #
        # copy_df['bw'] = (copy_df['b']) * (copy_df['w'])
        #
        # copy_df['bw_avg'] = copy_df['bw'].sum()
        # bw_avg = copy_df.iloc[0, copy_df.columns.get_loc('bw_avg')]
        # del copy_df['bw_avg']
        #
        # copy_df['dbw'] = np.sqrt((((copy_df['b'] - bw_avg) ** 2) * (copy_df['w'])).sum())
        # dbw = copy_df.iloc[0, copy_df.columns.get_loc('dbw')]
        # del copy_df['dbw']
        # del copy_df['bw']
        #
        # copy_df['R_avg'] = R_avg
        # copy_df['R_avg'].iloc[1:] = None
        # copy_df['dR'] = dR
        # copy_df['dR'].iloc[1:] = None
        # copy_df['Rw_avg'] = Rw_avg
        # copy_df['Rw_avg'].iloc[1:] = None
        # copy_df['dRw'] = dRw
        # copy_df['dRw'].iloc[1:] = None
        #
        # copy_df['b_avg'] = b_avg
        # copy_df['b_avg'].iloc[1:] = None
        # copy_df['db'] = db
        # copy_df['db'].iloc[1:] = None
        # copy_df['bw_avg'] = bw_avg
        # copy_df['bw_avg'].iloc[1:] = None
        # copy_df['dbw'] = dbw
        # copy_df['dbw'].iloc[1:] = None

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
                    fig.add_trace(
                        go.Bar(x=np.arange(wh, L, wh), y=model[ngram].counts[fa_click["points"][0]["x"]], name="∑∆w"))
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
                    fig.add_trace(
                        go.Bar(x=np.arange(wh, L, wh), y=model[ngram].counts[fa_click["points"][0]["x"]], name="∑∆w"))
                    print(model[ngram].sums[fa_click['points'][0]['x']])
                fa_click = None

                hover_data = []
                for data in df['ngram']:
                    hover_data.append("".join(data))
                fig1.add_trace(go.Scatter(x=df["R"], y=df["b"], mode="markers", text=hover_data))
                fig1.add_trace(go.Scatter(x=[model[ngram].R],
                                          y=[model[ngram].b],
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
                        fig.add_trace(go.Bar(x=np.arange(wh, L, wh), y=new_ngram.count[fa_click["points"][0]["x"]],
                                             name="‚àë‚àÜw"))

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
                    ww = fa_click['points'][0]["x"]
                    fig.add_trace(go.Bar(x=np.arange(0, L, wh), y=model[ngram].counts[ww], name="∑∆w"))
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

                    fig1.add_trace(go.Scatter(x=new_ngram.R, y=new_ngram.b, mode='marekers', hover_data=["new_ngram"]))
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
                    fig.add_trace(go.Bar(x=np.arange(ww, L, wh), y=model[ngram].counts[ww], name="∑∆w"))

                fa_click = None
                if graph_click:
                    print(model[ngram].sums.keys())

                graph_click = None

                fig1.add_trace(go.Scatter(x=df["R"], y=df["b"], mode="markers", text=hover_data))
                # fig1.add_trace(go.Scatter(x=[df['R'][active_cell['row']]],
                fig1.add_trace(go.Scatter(x=[df['R'][ids[active_cell['row']]]],
                                          # y=[df["b"][active_cell['row']]],
                                          y=[df["b"][ids[active_cell['row']]]],
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
            else:
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool))
                for data in df["ngram"]:
                    hover_data.append("".join(data))
                fig1.add_trace(go.Scatter(x=df["R"], y=df["b"], mode="markers", text=hover_data))
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
              [State("corpus", "value"),
               State("n_size", "value"),
               State("w", "value"),
               State("wh", "value"),
               State("we", "value"),
               State("wm", "value"),
               State("f_min", "value"),
               State("condition", "value"),
               State("def", "value")])
def save(n, active_cell, page_current, ids, file, n_size, w, wh, we, wm, fmin, opt, definition):
    if n is None:
        return dash.no_update
    else:
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
            df_copy['w'] = (df_copy['ƒ']) / (df_copy['ƒ'].sum())

            R_avg = df_copy['R'].mean()
            dR = df_copy['R'].std()
            Rw_avg = (df_copy['R'] * df_copy['w']).sum()
            dRw = np.sqrt((((df_copy['R'] - Rw_avg) ** 2) * df_copy['w']).sum())

            b_avg = df_copy['b'].mean()
            db = df_copy['b'].std()
            bw_avg = (df_copy['b'] * df_copy['w']).sum()
            dbw = np.sqrt((((df_copy['b'] - bw_avg) ** 2) * df_copy['w']).sum())

            # Assign calculated values using .loc to avoid SettingWithCopyWarning
            df_copy.loc[:, 'R_avg'] = None
            df_copy.loc[df_copy.index[0], 'R_avg'] = R_avg
            df_copy.loc[:, 'dR'] = None
            df_copy.loc[df_copy.index[0], 'dR'] = dR
            df_copy.loc[:, 'Rw_avg'] = None
            df_copy.loc[df_copy.index[0], 'Rw_avg'] = Rw_avg
            df_copy.loc[:, 'dRw'] = None
            df_copy.loc[df_copy.index[0], 'dRw'] = dRw

            df_copy.loc[:, 'b_avg'] = None
            df_copy.loc[df_copy.index[0], 'b_avg'] = b_avg
            df_copy.loc[:, 'db'] = None
            df_copy.loc[df_copy.index[0], 'db'] = db
            df_copy.loc[:, 'bw_avg'] = None
            df_copy.loc[df_copy.index[0], 'bw_avg'] = bw_avg
            df_copy.loc[:, 'dbw'] = None
            df_copy.loc[df_copy.index[0], 'dbw'] = dbw

            # Remove temporary 'w' column if not needed in the final output
            df_copy = df_copy.drop(columns=['w'])

        else:
             # Handle empty dataframe case if necessary
             # Maybe return an alert or log a message
             print("Warning: DataFrame is empty after filtering 'new_ngram'. Cannot save stats.")
             # Decide how to handle df_copy columns if it's empty
             pass


        if definition == "dynamic":
            output_filename = "saved_data/{0} contition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8}.xlsx".format(file, fmin, n_size, w, wh, we, wm, opt, definition)
            with pd.ExcelWriter(output_filename) as writer:
                df_copy.to_excel(writer, index=False) # Use df_copy here
            # writer.save() # Deprecated

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

                                 details_filename = f"saved_data/{file} {ngram_to_save_details}_details.xlsx"
                                 with pd.ExcelWriter(details_filename) as writer_details:
                                     df1 = pd.DataFrame()
                                     # Check if the ngram exists in the global model (might have been filtered)
                                     if ngram_to_save_details in model:
                                         df1["w"] = [*model[ngram_to_save_details].fa.keys()]
                                         df1['‚àÜF'] = [*model[ngram_to_save_details].fa.values()] # Original code had '‚àÜF', assuming this is correct?
                                         df1['fit=a*w^b'] = model[ngram_to_save_details].temp_fa
                                         df1.to_excel(writer_details, index=False)
                                     # Also save new_ngram specific data
                                     if new_ngram: # Save new_ngram details if definition is dynamic
                                         new_ngram_details_filename = f"saved_data/{file} new_ngram_dynamic_details.xlsx"
                                         with pd.ExcelWriter(new_ngram_details_filename) as writer_new_ngram:
                                             df_new = pd.DataFrame()
                                             df_new["w"] = [*new_ngram.dfa.keys()]
                                             df_new['∆F'] = [*new_ngram.dfa.values()] # Original used ∆F here
                                             df_new['fit=a*w^b'] = new_ngram.temp_dfa
                                             df_new.to_excel(writer_new_ngram, index=False)

                        else:
                            print(f"Warning: Selected index {selected_original_index} out of bounds for original DataFrame.")
                    else:
                         print(f"Warning: Calculated row index {row_index} is invalid for derived indices.")

                except Exception as e:
                    print(f"Error saving detailed ngram file (dynamic): {e}")
                    # Potentially add a Dash alert to inform the user


            return [html.Div(f"Saved data to {output_filename}")] # Provide feedback

        # Static definition part
        output_filename_static = "saved_data/{0} contition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8}.xlsx".format(
                file, fmin, n_size, w, wh, we, wm, opt, definition
            )
        with pd.ExcelWriter(output_filename_static) as writer:
            df_copy.to_excel(writer, index=False) # Use df_copy here
        # writer.save() # Deprecated

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
                                details_filename_static = f"saved_data/{file} {ngram}.xlsx"
                                with pd.ExcelWriter(details_filename_static) as writer_details:
                                    df1 = pd.DataFrame()
                                    df1["w"] = [*model[ngram].fa.keys()]
                                    df1['∆F'] = [*model[ngram].fa.values()] # Original used ∆F here
                                    df1['fit=a*w^b'] = model[ngram].temp_fa
                                    df1.to_excel(writer_details, index=False)
                                # writer_details.save() # Deprecated
                            else:
                                print(f"Warning: Ngram '{ngram}' selected but not found in model for detail saving.")
                     else:
                        print(f"Warning: Selected index {selected_original_index} out of bounds for original DataFrame (static).")
                else:
                     print(f"Warning: Calculated row index {row_index} is invalid for derived indices (static).")

            except Exception as e:
                print(f"Error saving detailed ngram file (static): {e}")
                # Potentially add a Dash alert

    # Use the modified df_copy for saving, keep global df potentially unchanged if needed elsewhere
    # Or update global df if necessary: df = df_copy
    # For now, just provide feedback
    return [html.Div("Saved data.")] # Generic feedback if filename isn't always generated


# import webbrowser # Commented out as it might cause issues if run non-interactively

if __name__ == "__main__":
    # webbrowser.open_new("http://127.0.0.1:8050/") # Commented out
    app.run(debug=False)
