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
import dash_table as dt
from os import listdir
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


def remove_punctuation(data):
    temp = []
    for i in range(len(data)):
        if data[i] in punctuation:
            continue
        else:
            temp.append(data[i].lower())
    return "".join(temp)


class Ngram(dict):
    def __init__(self, iterable=None):  # Ініціалізували наш розподіл як новий об'єкт класу, додаємо наявні елементи
        super(Ngram, self).__init__()
        # self.F_i = 0  # число унікальних ключів в розподілі
        self.fa = {}
        self.counts = {}
        self.sums = {}
        if iterable:
            self.update(iterable)

    def update(self, iterable):  # Оновлюємо розподіл елементами з наявного итерируемого набору даних
        for item in iterable:
            if item in self:
                self[item] += 1
            else:
                self[item] = 1
                # self.F_i += 1

    def hist(self):
        plt.bar(self.keys(), self.values())
        plt.show()


def make_dataframe(model, fmin=0):
    filtered_data = list(filter(lambda x: sum(value for value in model[x].values() if isinstance(value, int)) >= fmin, model))
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

    return pd.DataFrame(data=data)


def make_markov_chain(data, order=1):
    global model, L, V
    model = dict()

    L = len(data) - order
    model['new_ngram'] = Ngram()
    model['new_ngram'].bool = np.zeros(L, dtype=np.uint8)
    model['new_ngram'].pos = []
    if order > 1:
        for i in range(L):
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
                model[data[i]].bool[i] = 1
            else:
                model[data[i]] = Ngram([data[i + order]])
                model[data[i]].pos = []
                model[data[i]].pos.append(i + order)
                model[data[i]].bool = np.zeros(L, dtype=np.uint8)
                model[data[i]].bool[i] = 1

                model['new_ngram'].bool[i] = 1
                model['new_ngram'].pos.append(i + order)
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
            for i in data:
                temp.append(i)
            L = len(temp) - n
            return temp
        if split == 'letter':
            data = remove_punctuation(data)
            for i in data:
                for j in i:
                    if j == " ":
                        continue
                    temp_data.append(j)
            L = len(temp_data) - n
            return temp_data
        if split == 'symbol':
            for i in data:
                for j in i:
                    if j == " ":
                        temp_data.append("space")
                        continue
                    temp_data.append(j)
            L = len(temp_data) - n
            return temp_data
    if n > 1:
        if split == "word":
            data = data.split()
            L = len(data) - n
            for i in range(L):
                window = tuple(data[i: i + n])
                temp_data.append(window)
            return temp_data
        if split == "letter":
            data = remove_punctuation(data.split())
            for i in data:
                for j in i:
                    temp_data.append(j)
            L = len(temp_data) - n
            data = temp_data
            temp_data = []
            for i in range(L):
                window = tuple(data[i: i + n])
                temp_data.append(window)
            return temp_data
        if split == 'symbol':
            temp_data = []
            for i in data:
                for j in i:
                    if j == " ":
                        temp_data.append("space")
                        continue
                    temp_data.append(j)
            data = temp_data
            temp_data = []
            L = len(data) - n
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

# old layout was fun but not what i wanted
#
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
                                                dbc.InputGroupAddon("Size of ngram", addon_type="prepend"),
                                                dbc.Input(id="n_size", type="number", value=1),
                                            ], size="md", className="config"
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupAddon("Split by", addon_type="prepend"),
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
                                                # dbc.InputGroupAddon("Boundary Condition:", addon_type="append"),
                                                dbc.Select(
                                                    id="condition",
                                                    options=[
                                                        {"label": "no", "value": "no"},
                                                        {"label": "periodic", "value": "periodic"},
                                                        {"label": "ordinary", "value": "ordinary"}
                                                    ],
                                                    value="no"
                                                ),
                                                dbc.InputGroupAddon("Boundary Condition:", addon_type="append"),
                                            ], size="md", className="config"
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupAddon("filter", addon_type="prepend"),
                                                dbc.Input(id="f_min", type="number", value=0)
                                            ]
                                        ),
                                        html.Label("Sliding window"),

                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupAddon("Min window", addon_type="prepend"),
                                                dbc.Input(id="w", type="number"),
                                            ], size="md", className="window"
                                        ),

                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupAddon("Window shift", addon_type="prepend"),
                                                dbc.Input(id="wh", type="number"),
                                            ], size="md", className="window"
                                        ),

                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupAddon("Window exspansion", addon_type="prepend"),
                                                dbc.Input(id="we", type="number"),
                                            ], size="md", className="window"
                                        ),

                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupAddon("Max window", addon_type="prepend"),
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
                                                dbc.InputGroupAddon("Definition", addon_type="append")
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
                                        dbc.Button("Analyze", id="chain_button", color="primary", block=True),

                                        dbc.Button("Save data", id="save", color="danger", block=True),
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
                                    card=True,
                                    active_tab="data_table"
                                )

                            ),
                            dbc.CardBody(
                                [
                                    html.Div(id="box_tab",
                                             style={"display": "none", "height": "400px"},
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
                                                              'overflowY': 'auto', "overflowX": "none"}
                                             ))]),
                                    html.Div(id="box_chain",
                                             style={"display": "none"},
                                             children=[dbc.Spinner(dcc.Graph(id="chain", style={"height": "400px"}))]),

                                    dbc.CardHeader("Characteristics"),
                                    dbc.CardBody(
                                        [
                                            html.Div(["Length: "], id="l"),
                                            html.Div(["Vocabulary"], id="v"),
                                            html.Div(["Time: "], id="t")
                                        ]
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
                                card=True,
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
                                card=True,
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
    )

])
from dash.dependencies import Input, Output, State

app.layout = layout1
df = None
g = None
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx


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
    global L, data

    with open("corpus/" + corpus, encoding='utf-8') as f:
        file = f.read()
    if definition == "dynamic":
        data = prepere_data(file, n, split)
        wm = int(L / 10)
        w = int(wm / 10)
    else:
        temp = []
        if split == "letter":
            data = remove_punctuation(file)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            data = temp
        if split == "symbol":
            data = file
            for i in data:
                if i == " ":
                    temp.append("space")
                else:
                    temp.append(i)

            data = temp

        if split == "word":
            data = remove_punctuation(file)
            data = data.split()

        L = len(data) - n
        wm = int(L / 20)
        w = int(wm / 20)
    return [w, w, w, wm, ["Lenght: " + str(L)]]


new_ngram = None


@app.callback([Output("table", "data"), Output("chain", "figure"),
               Output("box_tab", "style"),
               Output("box_chain", "style"),
               Output("alert", "children"),
               Output("v", "children"),
               Output("t", "children")],
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
    if n is None:
        return dash.no_update, dash.no_update, {"display": 'inline'}, {
            "display": "none"}, dash.no_update, dash.no_update, dash.no_update

    # add alert corpus if not selected
    if corpus is None:
        return dash.no_updata, dash.no_update, {"display": "inline"}, {"display": "none"}, dbc.Alert(
            "Please choose corpus", color="danger", duration=2000,
            dismissable=False), dash.no_update, dash.no_update

    global data, L, V, model, ngram, df, g, new_ngram

    # string_counts = {}
    # for element in data:
    #     string_counts[element] = string_counts.get(element, 0) + 1
    #
    # data = [element for element in data if string_counts[element] >= f_min]

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

        return dash.no_update, fig, {"display": "none"}, {
            "display": 'inline'}, dash.no_update, dash.no_update, dash.no_update
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
            new_ngram.dt = calculate_distance(np.array(temp_pos, dtype=np.uint8), L, condition)
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

            windows = list(range(w, wm, we))

            def func(wind):
                model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=wh)
                model[ngram].fa[wind] = mse(model[ngram].counts[wind])

            temp_b = []
            temp_R = []
            temp_error = []
            temp_ngram = []
            temp_a = []

            for i, ngram in enumerate(df["ngram"]):
                for wind in windows:
                    func(wind)

                model[ngram].temp_fa = []
                ff = [*model[ngram].fa.values()]

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
                temp_ngram.append("new_ngram")
                df["ngram"] = temp_ngram
            df['R'] = temp_R
            df['b'] = temp_b
            df['a'] = temp_a
            df['goodness'] = temp_error

            df = df.sort_values(by="ƒ", ascending=False)
            df['rank'] = range(1, len(temp_R) + 1)
            df = df.set_index(pd.Index(np.arange(len(df))))

        return [df.to_dict("record"), dash.no_update, {"display": "inline"}, {"display": "none"}, dash.no_update,
                ["Vocabulary: " + str(V)], ["Time:" + str(round(time() - start, 4))]]


clikced_ngram = None


@app.callback([Output("graphs", "figure"), Output("fa", "figure"), ],
              [Input("dataframe", "active_tab"),
               Input("card-tabs", "active_tab"),
               Input("table", "active_cell"),
               Input("table", "derived_virtual_selected_rows"),
               Input("table", "derived_virtual_indices"),
               Input("chain", "clickData"),
               Input("scale", "value"),
               Input("fa", "clickData"),
               Input("graphs", "clickData"),
               Input("wh", "value")],
              [State("n_size", "value"),
               State("def", "value"), ])
def tab_content(active_tab2, active_tab1, active_cell, row_ids, ids, clicked_data, scale, fa_click, graph_click, wh, n,
                definition):
    global model, df, L, g, new_ngram
    if df is None:
        return dash.no_update, dash.no_update

    if ids is None:
        return dash.no_update, dash.no_update

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

                return fig, fig1
            else:
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
                    hover_data.append("".join(data))
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool, name="positions"))
                if fa_click:
                    ww = fa_click['points'][0]["x"]
                    fig.add_trace(go.Bar(x=np.arange(ww, L, wh), y=model[ngram].counts[ww], name="∑∆w"))

                fa_click = None
                if graph_click:
                    print(model[ngram].sums.keys())

                graph_click = None

                fig1.add_trace(go.Scatter(x=df["R"], y=df["b"], mode="markers", text=hover_data))
                fig1.add_trace(go.Scatter(x=[df['R'][active_cell['row']]],
                                          y=[df["b"][active_cell['row']]],
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
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool))
                for data in df["ngram"]:
                    hover_data.append("".join(data))
                fig1.add_trace(go.Scatter(x=df["R"], y=df["b"], mode="markers", text=hover_data))
                fig1.update_yaxes(type=scale)
                fig1.update_xaxes(type=scale)
                fig1.update_layout(hovermode="x unified")

            return fig, fig1

        return dash.no_update, dash.no_update


@app.callback([Output("temp_seve", "children")],
              [Input("save", "n_clicks"),
               Input("table", "active_cell"),
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
def save(n, active_cell, ids, file, n_size, w, wh, we, wm, fmin, opt, definition):
    if n is None:
        return dash.no_update
    else:
        global df, model, new_ngram

        #   2023
        #   Зміни в save
        #   - вивід без new_ngram
        #   - додаткові параметри

        df = df[df.ngram != 'new_ngram']
        df['rank'] = df['rank'] - 1

        df['w'] = (df['ƒ']) / (df['ƒ'].sum())

        df['R_avg'] = df['R'].mean()
        R_avg = df.iloc[0, df.columns.get_loc('R_avg')]
        del df['R_avg']

        df['dR'] = df['R'].std()
        dR = df.iloc[0, df.columns.get_loc('dR')]
        del df['dR']

        df['Rw'] = (df['R']) * (df['w'])

        df['Rw_avg'] = df['Rw'].sum()
        Rw_avg = df.iloc[0, df.columns.get_loc('Rw_avg')]
        del df['Rw_avg']

        df['dRw'] = np.sqrt((((df['R'] - Rw_avg) ** 2) * (df['w'])).sum())
        dRw = df.iloc[0, df.columns.get_loc('dRw')]
        del df['dRw']
        del df['Rw']


        df['b_avg'] = df['b'].mean()
        b_avg = df.iloc[0, df.columns.get_loc('b_avg')]
        del df['b_avg']

        df['db'] = df['b'].std()
        db = df.iloc[0, df.columns.get_loc('db')]
        del df['db']

        df['bw'] = (df['b']) * (df['w'])

        df['bw_avg'] = df['bw'].sum()
        bw_avg = df.iloc[0, df.columns.get_loc('bw_avg')]
        del df['bw_avg']

        df['dbw'] = np.sqrt((((df['b'] - bw_avg) ** 2) * (df['w'])).sum())
        dbw = df.iloc[0, df.columns.get_loc('dbw')]
        del df['dbw']
        del df['bw']


        df['R_avg'] = R_avg
        df['R_avg'].iloc[1:] = None
        df['dR'] = dR
        df['dR'].iloc[1:] = None
        df['Rw_avg'] = Rw_avg
        df['Rw_avg'].iloc[1:] = None
        df['dRw'] = dRw
        df['dRw'].iloc[1:] = None

        df['b_avg'] = b_avg
        df['b_avg'].iloc[1:] = None
        df['db'] = db
        df['db'].iloc[1:] = None
        df['bw_avg'] = bw_avg
        df['bw_avg'].iloc[1:] = None
        df['dbw'] = dbw
        df['dbw'].iloc[1:] = None



        if definition == "dynamic":
            writer = pd.ExcelWriter(
                "saved_data/{0} contition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8}.xlsx".format(file, fmin,
                                                                                                             n_size, w,
                                                                                                             wh, we, wm,
                                                                                                             opt,
                                                                                                             definition))
            df.to_excel(writer)
            writer.save()
            if active_cell:
                writer = pd.ExcelWriter("saved_data/" + file + " new_ngram.xlsx")
                df1 = pd.DataFrame()
                df1["w"] = [*new_ngram.dfa.keys()]
                df1['‚àÜF'] = [*new_ngram.dfa.values()]
                df1['fit=a*w^b'] = new_ngram.temp_dfa
                df1.to_excel(writer)
                writer.save()
            return dash.no_update

        writer = pd.ExcelWriter(
            "saved_data/{0} contition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8}.xlsx".format(file, fmin,
                                                                                                         n_size, w, wh,
                                                                                                         we, wm, opt,
                                                                                                         definition))
        df.to_excel(writer)
        writer.save()
        if active_cell:
            ngram = df['ngram'][ids[active_cell['row']]]
            writer = pd.ExcelWriter("saved_data/" + file + " " + ngram + ".xlsx")
            df1 = pd.DataFrame()
            df1["w"] = [*model[ngram].fa.keys()]
            df1['∆F'] = [*model[ngram].fa.values()]
            df1['fit=a*w^b'] = model[ngram].temp_fa
            df1.to_excel(writer)
            writer.save()
    return dash.no_update


import webbrowser

if __name__ == "__main__":
    webbrowser.open_new("http://127.0.0.1:8051/")
    app.server.run(host='0.0.0.0', port=8051, debug=False)
