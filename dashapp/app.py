# Import required libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import sqlite3


# Initialize app with desired style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = 'ContinU'

##############
# Data Loading
##############

cnx = sqlite3.connect('continu_dash.db')

df_user_stats = pd.read_sql_query("SELECT * FROM users", cnx)
df_trials = pd.read_sql_query("SELECT * FROM messages", cnx)


##############
# Defaults
##############

default_user = 22
n_trials = 5
trials_max = 50

##############
# App layout
##############

app.layout = html.Div([
    html.Div([

        html.H1('ContinU Authentication Dashboard',
                style={'width': '100%', 'textAlign': 'center', 'display': 'inline-block'}),
    
    ]),

    html.Div([

        html.Div([
            html.Label('Select a user:',
                       style={'fontSize': 24, 'textAlign': 'center'}),
            dcc.Graph(id='bar_rank',
                    style={'width': '100%', 'float': 'left', 'display': 'inline-block'}),
            ],
            style={'width': '30%', 'float': 'left',
                   'display': 'inline-block'}),

        html.Div([
            html.Label('',
                       id='timeseries_title',
                       style={'fontSize': 24, 'textAlign': 'center'}),
            dcc.Graph(id='time_series_auth',
                      config={'displayModeBar': False}),
            html.Hr(),
            html.Label('',
                       id='fingerprint_title',
                       style={'fontSize': 24, 'textAlign': 'center'}),
            dcc.Graph(id='scatter_fingerprint')
            ],
            style={'width': '70%', 'float': 'right',
                   'display': 'inline-block'})

    ])
])


##############
# Bar chart callback
##############

@app.callback(
    dash.dependencies.Output('bar_rank', 'figure'),
    [dash.dependencies.Input('bar_rank', 'clickData')])
def update_bar_rank(clickData):

    # Select only recent data
    df_plt = df_trials.groupby('user').tail(n_trials)

    # Compute max confidence for last N sessions
    df_plt = df_plt.groupby('user')['adj_ypred'].mean().reset_index()

    # Sort users by most risk
    df_plt = df_plt.sort_values(by='adj_ypred', ascending=False)

    # Make user name
    df_plt['user_name'] = ['User {:d} '.format(u) for u in df_plt['user']]

    # Determine selected user
    if clickData is None:
        user_selected = default_user
        selected_points = []
    else:
        user_selected = clickData['points'][0]['id']
        if user_selected not in df_plt['user'].values:
            user_selected = df_plt['user'].values[0]
        selected_points = [list(df_plt['user']).index(user_selected)]

    return {
        'data': [{'x': df_plt['adj_ypred'],
                  'y': df_plt['user_name'],
                  'ids': df_plt['user'],
                  'type': 'bar',
                  'orientation': 'h',
                  'hoverinfo': 'text',
                  'text': ['{:s}: {:.0f}% average confidence'.format(user_id, avg_pred) for user_id, avg_pred in zip(df_plt['user_name'], df_plt['adj_ypred'])],
                  'selectedpoints': selected_points,
                  'unselected': {'marker': {'opacity': 0.5, 'color': 'black'}}}],
        'layout': go.Layout(
            yaxis={'title': '',
                   'tickfont': {'size': 14}},
            xaxis={'title': 'Recent authentication confidence (%)',
                   'titlefont': {'size': 18},
                   'tickfont': {'size': 14}},
            margin={'l': 80, 'b': 40, 't': 30, 'r': 0},
            hovermode='closest',
            height=600
        )
    }


@app.callback(
    dash.dependencies.Output('time_series_auth', 'figure'),
    [dash.dependencies.Input('bar_rank', 'clickData'),
    dash.dependencies.Input('time_series_auth', 'clickData')])
def make_scatter(clickData, clickData_ts):

    # Determine selected user
    if clickData is None:
        return {'layout': go.Layout(height=250)}
    else:
        user_selected = clickData['points'][0]['id']
    
    # Determine time to plot
    df_plt = df_trials[df_trials['user'] == user_selected]
    df_plt = df_plt.tail(trials_max)
    df_plt.reset_index(drop=True, inplace=True)

    if clickData_ts is None:
        trial_selected = None
    else:
        trial_selected = clickData_ts['points'][0]['id']

    return {
        'data': [go.Scatter(
            x=df_plt.index,
            y=df_plt['adj_ypred'],
            ids=df_plt.index,
            hoverinfo='text',
            text=['Confidence = {:.0f}%<br>Message:<br>{:s}'.format(ypred, msg[:50]+'<br>'+msg[50:]) for ypred, msg in zip(df_plt['adj_ypred'], df_plt['msg'])],
            mode='lines+markers',
            marker={'size': 12, 
                    'color': 'rgba(0, 116, 217, 0.7)'},
            line={'width': 2, 'color': 'black'},
            unselected={'marker': {'opacity': 0.5, 'color': 'black'}},
            selectedpoints=[trial_selected])],

        'layout': go.Layout(
            yaxis={'title': 'Authentication confidence (%)', 'autorange': False, 'range': [-5, 105]},
            xaxis={'title': 'Message ID'},
            margin={'l': 150, 'b': 40, 't': 20, 'r': 20},
            hovermode='closest',
            height=250,
            width=900
        )
    }


@app.callback(
    dash.dependencies.Output('scatter_fingerprint', 'figure'),
    [dash.dependencies.Input('bar_rank', 'clickData'),
     dash.dependencies.Input('time_series_auth', 'clickData')])
def make_fingerprint(clickData_user, clickData_trial):
    # Determine user of interest
    if clickData_user is None:
        return {'layout': go.Layout(height=250)}
    else:
        user_selected = clickData_user['points'][0]['id']

    # Get feats of trials
    df_temp = df_trials.loc[(df_trials['user'] == user_selected)]
    df_temp = df_temp.tail(trials_max)
    df_temp.reset_index(inplace=True, drop=True)

    if clickData_trial is None:
        return {'layout': go.Layout(height=250)}
    else:
        trial_selected = clickData_trial['points'][0]['id']

    # Get features of current trial
    cols_drop = ['index', 'user', 'sess', 'task', 'trial', 'msg', 'adj_ypred']
    feat_trial_plt = df_temp.drop(cols_drop, axis=1)
    feat_trial_plt = feat_trial_plt.loc[trial_selected]
    feat_trial_plt = feat_trial_plt.dropna().reset_index().rename(columns={'index': 'feature', trial_selected: 'trial_time'})

    # Merge features of current trial with user stats
    df_temp = df_user_stats[df_user_stats['user'] == user_selected]
    df_plt = df_temp.merge(feat_trial_plt, on='feature')
    df_plt['pcrange'] = df_plt['pc90'] - df_plt['pc10']
    # df_plt.sort_values(by='pcrange', inplace=True)
    df_plt.sort_values(by='pc90', inplace=True)

    # # Determine if point in bounds
    df_plt['in_range'] = (df_plt['trial_time'] <= df_plt['pc90']) & (df_plt['trial_time'] >= df_plt['pc10'])
    df_plt1 = df_plt[df_plt['in_range']]
    df_plt2 = df_plt[~df_plt['in_range']]

    return {
        'data': [
        go.Scatter(
            x=df_plt['feature'],
            y=df_plt['trial_time'],
            hoverinfo='text',
            name='Message #{:d} statistics: typical'.format(trial_selected),
            text=['{:s} delay = {:.0f}ms<br>User mean = {:.0f}ms'.format(dg, tt, meant) for dg, tt, meant in zip(df_plt['feature'], df_plt['trial_time'], df_plt['mean'])],
            mode='markers',
            marker={'size': 6, 'color': "green"}),
        go.Scatter(
            x=df_plt2['feature'],
            y=df_plt2['trial_time'],
            hoverinfo='text',
            name='Message #{:d} statistics: atypical'.format(trial_selected),
            text=['{:s} delay = {:.0f}ms<br>User mean = {:.0f}ms'.format(dg, tt, meant) for dg, tt, meant in zip(df_plt2['feature'], df_plt2['trial_time'], df_plt2['mean'])],
            mode='markers',
            marker={'size': 6, 'color': "red"}),
        go.Scatter(
            x=df_plt['feature'],
            y=df_plt['pc10'],
            hoverinfo='none',
            line = {"color": "grey"},
            fill=None,
            showlegend=False,
            mode='lines'),
        go.Scatter(
            x=df_plt['feature'],
            y=df_plt['pc90'],
            hoverinfo='none',
            line = {"color": "grey"},
            fill='tonexty',
            name='User {:d} typical profile'.format(user_selected),
            mode='lines')
        ],

        'layout': go.Layout(
            yaxis={'title': 'Keystroke time (ms)', 'autorange': False, 'range': [-100, 1000]},
            xaxis={'title': 'Keystroke features',
                   'showticklabels': False,
                   'showgrid': False},
            margin={'l': 150, 'b': 40, 't': 20, 'r': 20},
            legend=dict(orientation="v", x=.05, y=1.1, font={'size': 15}),
            hovermode='closest',
            height=250,
            width=900
        )
    }


@app.callback(
    dash.dependencies.Output('fingerprint_title', 'children'),
    [dash.dependencies.Input('bar_rank', 'clickData'),
     dash.dependencies.Input('time_series_auth', 'clickData')])
def update_fingerprint_title(clickData, clickData_ts):

    # Determine selected user
    if clickData is None:
        return "Select a user to see their keystroke biometrics"
    else:
        user_selected = clickData['points'][0]['id']

    if clickData_ts is None:
        return "Select a message to compare with user {:d}'s fingerprint".format(user_selected)
    else:
        return 'User {:d} keystroke fingerprint'.format(user_selected)


@app.callback(
    dash.dependencies.Output('timeseries_title', 'children'),
    [dash.dependencies.Input('bar_rank', 'clickData')])
def update_timeseries_title(clickData):

    # Determine selected user
    if clickData is None:
        return "Select a user to see their keystroke biometrics"
    else:
        user_selected = clickData['points'][0]['id']

    return "Authentication confidence for user {:d}'s recent messages".format(user_selected)


# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True)
