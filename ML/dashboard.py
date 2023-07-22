from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
server=app.server

# Part 1d Setup
part1d_df = pd.read_csv('./assets/part1d.csv')
part1d_income_histogram = px.histogram(part1d_df, x='Income', nbins=20, title='Income Distribution')
part1d_histogram = dcc.Graph(id='part1d-histogram', figure={})
part1d_boxplot = dcc.Graph(id='part1d-boxplot', figure={})
part1d_dropdown = dcc.Dropdown(id='part1d-dropdown',
                            options=part1d_df.columns.values[1:],
                               value='Year_Birth',
                               clearable=False)

#Part 4a Setup
part4a_data = pd.read_csv('./assets/part4a_scores.csv')
part4a_data = part4a_data.set_index('Model')
part4a_scores = html.Div([
    html.Br(),
    html.P('Accuracy Score: ', 
           style={'display': 'inline-block', 'font-weight': 'bold'}),
    html.Span(children='', id='accuracy-score', 
              style={'display': 'inline-block', 'margin-left': '5px'}),
    html.Br(),
    html.P('Recall Score: ', 
           style={'display': 'inline-block', 'font-weight': 'bold'}),
    html.Span(children='', id='recall-score', 
              style={'display': 'inline-block', 'margin-left': '5px'}),
    html.Br(),
    html.P('Precision Score: ', 
           style={'display': 'inline-block', 'font-weight': 'bold'}),
    html.Span(children='', id='precision-score', 
              style={'display': 'inline-block', 'margin-left': '5px'}),
    html.Br(),
    html.P('AUC Score: ', 
           style={'display': 'inline-block', 'font-weight': 'bold'}),
    html.Span(children='', id='auc-score', 
              style={'display': 'inline-block', 'margin-left': '5px'}),
    html.Br(),
    html.P('F1 Score: ', 
           style={'display': 'inline-block', 'font-weight': 'bold'}),
    html.Span(children='', id='f1-score', 
              style={'display': 'inline-block', 'margin-left': '5px'}),
])
part4a_cm = html.Img(src='', id='part4a-cm', style={'width': '100%'})
part4a_roc = html.Img(src='', id='part4a-roc', style={'width': '100%'})
part4a_dropdown = dcc.Dropdown(id='part4a-dropdown',
                               options=['Random Forest Classifier','Support Vector Machine','Gradient Boosting Classifier'],
                               value='Random Forest Classifier',
                               clearable=False)

#Part_6 Setup
part6_10important = html.Img(src='./assets/part6_10important.png', id='part6-10important') 
part6_learningcurve = html.Img(src='./assets/part6_learningcurve.png', id='part6-learningcurve', style={'width': '100%'})
part6_partialdependence = html.Img(src='./assets/part6_partialdependence.png', id='part6-partialdependence', style={'width': '100%'})

# Layout
app.layout = dbc.Container([
    #title centered and large
    dbc.Row([
        dbc.Col(html.H1("ML DASHBOARD", className="text-center"), className="mb-1 mt-2")
    ]),
    dcc.Tabs(id="tabs", value='tab-1', children=[   # Tabs
        dcc.Tab(label='Part 1d - Exploratory Data Analysis', value='tab-1'),
        dcc.Tab(label='Part 4a - Model\'s Evaluation', value='tab-2'),
        dcc.Tab(label='Part 6 - Model\'s Insight Generation', value='tab-3'),
    ]),
    html.Div(id='tab-content')
])

# Callback for Tabs
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return dbc.Container([
            dbc.Row([ # Dropdown
                dbc.Col([part1d_dropdown], width=3, className='mt-2'),
            ], justify='center'),
            dbc.Row([ # Histogram
                dbc.Col([part1d_histogram])
            ], justify='center'),
            dbc.Row([ # Boxplot
                dbc.Col([part1d_boxplot])
            ], justify='center'),
        ])
    elif tab == 'tab-2':
        return dbc.Container([
            dbc.Row([ # Dropdown
                dbc.Col([part4a_dropdown], width=3, className='mt-2'),
            ], justify='center'),
            dbc.Row([ # Scores
                dbc.Col([part4a_scores])
            ]),
            dbc.Row([ 
                # Confusion Matrix
                dbc.Col([part4a_cm]),
                # ROC Curve
                dbc.Col([part4a_roc])
            ], style={'textAlign': 'center'}),
        ])
    elif tab == 'tab-3':
        return html.Div([
            dbc.Row([
                # 10 Most Important Features
                dbc.Col([part6_10important]),
            ], style={'textAlign': 'center'}),
            dbc.Row([
                # Learning Curve
                dbc.Col([part6_learningcurve]),
                # Partial Dependence Plot
                dbc.Col([part6_partialdependence])
            ], justify='center')
        ])

# Callback for Part_1d
@app.callback(
    Output('part1d-histogram', 'figure'),
    Output('part1d-boxplot', 'figure'),
    Input('part1d-dropdown', 'value')
)
def update_graph(column):
    histogram = px.histogram(data_frame=part1d_df, x=column)
    boxplot = px.box(data_frame=part1d_df, x=column)
    return histogram, boxplot

# Callback for Part_4a
@app.callback(
    Output('accuracy-score', 'children'),
    Output('recall-score', 'children'),
    Output('precision-score', 'children'),
    Output('auc-score', 'children'),
    Output('f1-score', 'children'),
    Output('part4a-cm', 'src'),
    Output('part4a-roc', 'src'),
    Input('part4a-dropdown', 'value')
)
def update_graph(model):
    # Scores
    accuracy_score = part4a_data.loc[model, 'Accuracy']
    recall_score = part4a_data.loc[model, 'Recall']
    precision_score = part4a_data.loc[model, 'Precision']
    auc_score = part4a_data.loc[model, 'AUC']
    f1_score = part4a_data.loc[model, 'F1']

    if model == 'Random Forest Classifier':
        # RFC Confusion Matrix
        cm = './assets/part4a_rfc_cm.png'
        # RFC ROC Curve
        roc = './assets/part4a_rfc_roc.png'
    elif model == 'Support Vector Machine':
        # SVM Confusion Matrix
        cm = './assets/part4a_svc_cm.png'
        # SVM ROC Curve
        roc = './assets/part4a_svc_roc.png'
    elif model == 'Gradient Boosting Classifier':
        # GBC Confusion Matrix
        cm = './assets/part4a_gbc_cm.png'
        # GBC ROC Curve
        roc = './assets/part4a_gbc_roc.png'

    return accuracy_score, recall_score, precision_score, auc_score, f1_score, cm, roc

if __name__ == "__main__":
    app.run_server(debug=True)
