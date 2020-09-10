import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime as dt
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import json
import re

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "mahjan score"

# Player's colors --------------------------------------------------
colors = ['lightblue', 'lightgreen', 'plum', 'lightsalmon']

dselection = ['半荘別ポイント', '月別ポイント']

# Main Dash part
app.layout = html.Div([
    # pick date range
    html.Div([
        html.H2("麻雀成績アナライザー"),
        dcc.DatePickerRange(
            id='my-date-picker-range',
            min_date_allowed=dt(2020, 3, 1),
            max_date_allowed=dt.today(),
            end_date=dt.today()
        ),
        ], className="mytablestyle"),

    # pick the data type
    html.Div([
        dcc.RadioItems(
        id='datatype',
        options=[{'label': i, 'value': i} for i in dselection],
        value=dselection[0],
        labelStyle={'display': 'inline-block'}
        ),
    ], className="mytablestyle"),

    # intermediate dataframe does not display in the website
    html.Div(id='intermediate-value', style={'display': 'none'}),

    # Other graph and table that are all callbacks
    dcc.Graph(id='mygraph'),

    # Total score
    html.Div([
            html.Div(html.P('現在の総合ポイント')),
            html.Div(id='totalscore'),
        ], className="mytablestyle"),

    # Monthly score obtained.
    html.Div([
            html.Div(html.P('月別獲得ポイント（その月に何ポイント獲ったか）')),
            html.Div(id='monthlyscore'),
        ], className="mytablestyle"),

    # Distribution plot of score
    dcc.Graph(id='mydistplot'),

    # Bar graph of standings.
    dcc.Graph(id='bargraph'),

    html.Div([
        html.Div(html.P('各順位の合計獲得数まとめ')),
        html.Div(id='table'),
        ], className="mytablestyle"),

    # Monthly review
    html.Div([
        html.Div(html.P('月別１位獲得数')),
        html.Div(id='firs'),
        ], className="mytablestyle"),
    html.Div([
        html.Div(html.P('月別2位獲得数')),
        html.Div(id='seco'),
        ], className="mytablestyle"),
    html.Div([
        html.Div(html.P('月別3位獲得数')),
        html.Div(id='thir'),
        ], className="mytablestyle"),
    html.Div([
        html.Div(html.P('月別4位獲得数')),
        html.Div(id='four', style={'padding-bottom': '100px'}),
        ], className="mytablestyle"),
])

# Callback for data from players_score.pkl
@app.callback(Output("intermediate-value", "children"),
    [Input("my-date-picker-range", "start_date"),
    Input("my-date-picker-range", "end_date")])
def update_output(start_date, end_date):
    players = pd.read_pickle('players_score.pkl')
    if start_date is not None:
        start_date = dt.strptime(re.split('T| ', start_date)[0], '%Y-%m-%d')
        players = players.loc[(players['date'] >= start_date)]
    if end_date is not None:
        end_date = dt.strptime(re.split('T| ', end_date)[0], '%Y-%m-%d')
        players = players.loc[(players['date'] <= end_date)]
    return players.to_json(date_format='iso', orient='split')

# Callback for total score graph and table
@app.callback([Output('mygraph', 'figure'),
    Output('totalscore', 'children'),
    Output('monthlyscore', 'children')],
    [Input('intermediate-value', 'children'),
    Input('datatype', 'value')])
def update_fig(jsonified_df, data_type):
    players = pd.read_json(jsonified_df, orient='split')
    # Convert players dataframe if data type is selected as monthly type
    if data_type == dselection[1]:
        #Monthly sum of points
        players['date'] = players['date'].dt.to_period('M')
        players = players.groupby('date').sum().reset_index()
        players['date'] = players['date'].dt.strftime('%Y-%m')

    # Figure
    fig = go.Figure()

    for i, name in enumerate(players.columns[1:]):
        fig.add_trace(go.Scatter(x=players.date, y=np.array(players[name]).cumsum(),
                            mode='lines',
                            name=name,
                            line=dict(color=colors[i], width=4)))

    fig.update_layout(plot_bgcolor='whitesmoke',
        title='合計ポイントの推移',
        )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(
            size=18,
            color="black"
        ),),
        font=dict(
            size=18),
    )

    # Reinstall the original dataframe for data table
    players = pd.read_json(jsonified_df, orient='split')

    # Total
    summed = players.sum()

    #Monthly sum of points
    players = pd.read_json(jsonified_df, orient='split')
    players['date'] = players['date'].dt.to_period('M')
    month_sum = players.groupby('date').sum().reset_index()
    month_sum['date'] = month_sum['date'].dt.strftime('%Y-%m')

    return fig, html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in summed.index])
            ),
        html.Tbody(
            html.Tr([html.Td(val) for val in summed])
            ),
        ]), html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in month_sum.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(month_sum.iloc[i][col]) for col in month_sum.columns
            ]) for i in range(len(month_sum['date']))
        ])
    ])

# Callback for distribution plot for all hanchan score
@app.callback(Output('mydistplot', 'figure'),
    [Input('intermediate-value', 'children')])
def update_fig(jsonified_df):
    players = pd.read_json(jsonified_df, orient='split')

    # Distplots of monthly points
    hist_data = [players[name] for name in players.columns[1:]]
    group_labels = [name for name in players.columns[1:]]
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
    fig.update_layout(plot_bgcolor='whitesmoke',
        title='半荘ごとの獲得ポイントの分布',
        )
    fig.update_layout(legend=dict(
        traceorder="normal",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(
            size=18,
            color="black"
        ),),
        font=dict(
            size=18),
    )

    return fig

# Callback for standing winning number graph and tables
@app.callback([Output('bargraph', 'figure'),
    Output('table', 'children'),
    Output('firs', 'children'),
    Output('seco', 'children'),
    Output('thir', 'children'),
    Output('four', 'children'),],
    [Input('intermediate-value', 'children')])
def update_standings(jsonified_df):
    players = pd.read_json(jsonified_df, orient='split')

    # Get summary stats for standings
    onlyplayers = players[[name for name in players.columns[1:]]]
    arr = np.argsort(-onlyplayers.values, axis=1)
    standings = pd.DataFrame(onlyplayers.columns[arr], index=onlyplayers.index)
    standings = standings.rename(columns={0:'1st', 1:'2nd', 2:'3rd', 3:'4th'})
    summary = standings.stack().groupby(level=[1]).value_counts().unstack(0, fill_value=0).reset_index()

    #Subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("１位", "２位", "３位", "４位"))

    fig.add_trace(go.Bar(x=summary['index'], y=summary['1st'],
        marker=dict(color=colors, coloraxis="coloraxis1")), row=1, col=1)
    fig.add_trace(go.Bar(x=summary['index'], y=summary['2nd'],
        marker=dict(color=colors, coloraxis="coloraxis1")), row=1, col=2)
    fig.add_trace(go.Bar(x=summary['index'], y=summary['3rd'],
        marker=dict(color=colors, coloraxis="coloraxis1")), row=2, col=1)
    fig.add_trace(go.Bar(x=summary['index'], y=summary['4th'],
        marker=dict(color=colors, coloraxis="coloraxis1")), row=2, col=2)

    # Update yaxis properties
    # fig.update_yaxes(title_text="合計", row=1, col=1)
    # fig.update_yaxes(title_text="合計", row=1, col=2)
    # fig.update_yaxes(title_text="合計", row=2, col=1)
    # fig.update_yaxes(title_text="合計", row=2, col=2)

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=20,color='#708090')

    fig.update_layout(
        plot_bgcolor='whitesmoke',
        title_text="順位獲得数の比較",
        showlegend=False,
        font=dict(
            size=18),
        )

    # Get summary stats by month
    standings['date'] = players['date'].dt.to_period('M')
    g = standings.groupby('date')
    bm_firs = g['1st'].apply(pd.value_counts).unstack(-1).fillna(0).reset_index()
    bm_seco = g['2nd'].apply(pd.value_counts).unstack(-1).fillna(0).reset_index()
    bm_thir = g['3rd'].apply(pd.value_counts).unstack(-1).fillna(0).reset_index()
    bm_four = g['4th'].apply(pd.value_counts).unstack(-1).fillna(0).reset_index()
    bm_firs['date'] = bm_firs['date'].dt.strftime('%Y-%m')
    bm_seco['date'] = bm_seco['date'].dt.strftime('%Y-%m')
    bm_thir['date'] = bm_thir['date'].dt.strftime('%Y-%m')
    bm_four['date'] = bm_four['date'].dt.strftime('%Y-%m')

    return fig, html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in summary.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(summary.iloc[i][col]) for col in summary.columns
            ]) for i in range(len(summary['1st']))
        ])
    ]), html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in bm_firs.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(bm_firs.iloc[i][col]) for col in bm_firs.columns
            ]) for i in range(len(bm_firs['date']))
        ])
    ]), html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in bm_seco.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(bm_seco.iloc[i][col]) for col in bm_seco.columns
            ]) for i in range(len(bm_seco['date']))
        ])
    ]), html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in bm_thir.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(bm_thir.iloc[i][col]) for col in bm_thir.columns
            ]) for i in range(len(bm_thir['date']))
        ])
    ]), html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in bm_four.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(bm_four.iloc[i][col]) for col in bm_four.columns
            ]) for i in range(len(bm_four['date']))
        ])
    ])

if __name__ == '__main__':
    app.run_server()
