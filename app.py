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

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

# Main Dash part
app.layout = html.Div([
    html.Div([
        html.H2("遠隔マージャン対戦成績表"),
        ], className="topnav"),

    dcc.DatePickerRange(
        id='my-date-picker-range',
        min_date_allowed=dt(2020, 3, 1),
        max_date_allowed=dt.today(),
        end_date=dt.today()
    ),

    html.Div(id='intermediate-value', style={'display': 'none'}),

    html.Div(dcc.Loading([
    dcc.Graph(id='mygraph'),
    html.Div(html.P('合計点数', style={'padding-left': '20px', })),
    html.Div(id='totalscore', style={'padding-left': '20px', }),
    html.Div(html.P('')),
    dcc.Graph(id='bargraph'),
    html.Div(html.P('順位合計獲得数', style={'padding-left': '20px', })),
    html.Div(id='table', style={'padding-left': '20px', }),
    ])),
])

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

@app.callback([Output('mygraph', 'figure'),
    Output('totalscore', 'children')],
    [Input('intermediate-value', 'children')])
def update_fig(jsonified_df):
    players = pd.read_json(jsonified_df, orient='split')

    summed = players.sum()

    fig = go.Figure()
    colors = ['lightblue', 'lightgreen', 'plum', 'lightsalmon']

    fig.add_trace(go.Scatter(x=players.date, y=np.array(players['Mirataro']).cumsum(),
                        mode='lines',
                        name='Mirataro',
                        line=dict(color=colors[0], width=4)))
    fig.add_trace(go.Scatter(x=players.date, y=np.array(players['Shinwan']).cumsum(),
                        mode='lines',
                        name='Shinwan',
                        line=dict(color=colors[1], width=4)))
    fig.add_trace(go.Scatter(x=players.date, y=np.array(players['ToShiroh']).cumsum(),
                        mode='lines',
                        name='ToShiroh',
                        line=dict(color=colors[2], width=4)))
    fig.add_trace(go.Scatter(x=players.date, y=np.array(players['yukoron']).cumsum(),
                        mode='lines',
                        name='yukoron',
                        line=dict(color=colors[3], width=4)))

    fig.update_layout(plot_bgcolor='whitesmoke',
        title='合計点数の推移',
        xaxis_title='日付',
        yaxis_title='合計点数',
        )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(
            size=16,
            color="black"
        ),
    ))

    return fig, html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in summed.index])
            ),
        html.Tbody(
            html.Tr([html.Td(val) for val in summed])
            ),
        ])

@app.callback([Output('bargraph', 'figure'),
    Output('table', 'children')],
    [Input('intermediate-value', 'children')])
def update_standings(jsonified_df):
    players = pd.read_json(jsonified_df, orient='split')

    # Get summary stats for standings
    onlyplayers = players[['ToShiroh', 'Shinwan', 'yukoron', 'Mirataro']]
    arr = np.argsort(-onlyplayers.values, axis=1)
    standings = pd.DataFrame(onlyplayers.columns[arr], index=onlyplayers.index)
    standings = standings.rename(columns={0:'1st', 1:'2nd', 2:'3rd', 3:'4th'})
    summary = standings.stack().groupby(level=[1]).value_counts().unstack(0, fill_value=0).reset_index()

    #Subplots
    colors = ['lightblue', 'lightgreen', 'plum', 'lightsalmon']
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
    fig.update_yaxes(title_text="合計", row=1, col=1)
    fig.update_yaxes(title_text="合計", row=1, col=2)
    fig.update_yaxes(title_text="合計", row=2, col=1)
    fig.update_yaxes(title_text="合計", row=2, col=2)

    fig.update_layout(
        title_text="順位獲得数の比較",
        showlegend=False)

    return fig, html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in summary.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(summary.iloc[i][col]) for col in summary.columns
            ]) for i in range(len(summary['1st']))
        ])
    ])

if __name__ == '__main__':
    app.run_server()
