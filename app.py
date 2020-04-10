import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly import graph_objects as go

import pandas as pd
from openpyxl import load_workbook
import re, json

raw_data_file = 'raw datamt2hhx 2.xlsx'
datamap_file = 'datamap.json'

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app = dash.Dash(__name__)
server = app.server

def load_data(data_file):
    wb = load_workbook(raw_data_file)
    ws = wb['A1']
    data = ws.values
    cols = next(data)[0:]
    df = pd.DataFrame(data, columns=cols)
    return df[~pd.isna(df['LM8'])]

def remap_df(df, datamap_file):
    with open(datamap_file) as fp:
        datamap = json.load(fp)

    df['D2'].fillna(104, inplace=True)

    for col in df.columns:
        try:
            if type(datamap[col]) == type(dict()):
                df[col] = df[col].map(lambda x: datamap[col][str(int(x))])
            elif type(datamap[col]) == type(str()):
                question = re.findall(r'(.*)r\d+', col)[0]
                df.rename(columns={col: question + ' --- ' + datamap[col]}, inplace=True)
                try:
                    col = question + ' --- ' + datamap[col]
                    re.findall('LM6 --- .*', col)[0]
                    d = {1: 'Strongly Agree',
                        2: 'Agree',
                        3: 'Neither Agree nor Disagree',
                        4: 'Disagree',
                        5: 'Strongly Disagree'}
                    df[col] = df[col].map(lambda x: d[int(x)])
                except Exception as e:
                    pass
        except Exception as e:
            pass

# Load and format data
df = load_data(raw_data_file)
remap_df(df, datamap_file)

app.layout = html.Div([
    html.Div([

        html.Div([
            html.Label('Gender'),
            dcc.Dropdown(
                id='gender',
                options=[{'label': i, 'value': i} for i in ['All', "Female", "Male"]],
                value='All'
            ),
            html.Label('Ethnicity'),
            dcc.Dropdown( # 'D9 --- {value}' == 1
                id='ethnicity',
                options=[{'label': i, 'value': i} for i in ["All", "Caucasian", "Black", "Hispanic or Latino", "Asian", "American Indian", "Other", "Prefer not to answer"]],
                value='All'
            ),
            html.Label('Age'),
            dcc.Dropdown(
                id='age',
                options=[{'label': i, 'value': i} for i in ['All', "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]],
                value='All'
            ),
            html.Label('Marital Status'),
            dcc.Dropdown(
                id='marital-status',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted(df['D3'].unique())],
                value='All'
            ),
            html.Label('Children Under 18'),
            dcc.Dropdown(
                id='children',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted(df['D4'].unique())],
                value='All'
            ),
            html.Label('Employment Status'),
            dcc.Dropdown(
                id='employment',
                options=[{'label': i, 'value': i} for i in ['All', "I work full time", "I work part time", "I am a freelancer/consultant", "I am unemployed", "I am a student",
                "I am retired",  "I am a stay at home mom/dad"]],
                value='All'
            ),
            html.Label('Industry'),
            dcc.Dropdown(
                id='industry',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted(df['D2'].unique())],
                value='All'
            ),
            html.Label('Household Income'),
            dcc.Dropdown(
                id='hhi',
                options=[{'label': i, 'value': i} for i in ['All', "Under $25,000", "$25,000 - $49,999", "$50,000 - $74,999", "$75,000 - $99,999",
                "$100,000 - $124,999", "$125,000 - $149,999", "$150,000 - $249,999", "$250,000 or more", "Prefer not to answer"]],
                value='All'
            ),
            html.Label('Highest Education Completed'),
            dcc.Dropdown(
                id='education',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted(df['D8'].unique())],
                value='All'
            ),
        ],
        style={'width': '25%', 'display': 'inline-block'}),

        dcc.Graph(id='S5-S6-S7')
    ]),

    dcc.Graph(id='LM2-graphic'),
    dcc.Graph(id='LM3-graphic')
])

@app.callback(
    [Output('LM2-graphic', 'figure'),
     Output('S5-S6-S7', 'figure'),
     Output('LM3-graphic', 'figure')],
    [Input('ethnicity', 'value'),
     Input('gender', 'value'),
     Input('age', 'value'),
     Input('marital-status', 'value'),
     Input('children', 'value'),
     Input('employment', 'value'),
     Input('industry', 'value'),
     Input('hhi', 'value'),
     Input('education', 'value')])
def update_graph(ethnicity, *args): # same order as inputs
    cols = ['S2', 'Hid_Age', 'D3', 'D4', 'S4', 'D2', 'D5', 'D8']

    dff = df[df[f'D9 --- {ethnicity}']==1] if ethnicity != 'All' else df

    for col, value in zip(cols, args):
        dff = dff[dff[col] == value] if value != 'All' else dff

    # LM2 - What gave most joy (ranked 1-3, 1 being most)
    LM2 = dff.filter(regex=r'LM2.*')
    cols = list(LM2.columns.sort_values(ascending=False))
    cols.remove('LM2 --- Other')
    LM2 = LM2[['LM2 --- Other'] + cols]

    LM2_fig = go.Figure(data=[
        go.Bar(
            orientation='h',
            y = [re.findall(r'--- (.*)', col)[0] for col in LM2.columns],
            x = [LM2[LM2[col]==1][col].count() for col in LM2.columns],
            name = 'Most Joy'),
        go.Bar(
            orientation='h',
            y = [re.findall(r'--- (.*)', col)[0] for col in LM2.columns],
            x = [LM2[LM2[col]==2][col].count() for col in LM2.columns],
            name = 'Second Most Joy'),
        go.Bar(
            orientation='h',
            y = [re.findall(r'--- (.*)', col)[0] for col in LM2.columns],
            x = [LM2[LM2[col]==3][col].count() for col in LM2.columns],
            name = 'Third Most Joy')
    ])
    LM2_fig.update_layout(
        barmode='stack',
        height=800,
        title_text="Please rank these activities in the order of which ones gave you the most joy. Rank up to 3."
        )

    # LM3 - What will you do first (ranked 1-3, 1 being most)
    LM3 = dff.filter(regex=r'LM3.*')
    cols = list(LM3.columns.sort_values(ascending=False))
    cols.remove('LM3 --- Other')
    LM3 = LM3[['LM3 --- Other'] + cols]

    LM3_fig = go.Figure(data=[
        go.Bar(
            orientation='h',
            y = [re.findall(r'--- (.*)', col)[0] for col in LM3.columns],
            x = [LM3[LM3[col]==1][col].count() for col in LM3.columns],
            name = 'Do first'),
        go.Bar(
            orientation='h',
            y = [re.findall(r'--- (.*)', col)[0] for col in LM3.columns],
            x = [LM3[LM3[col]==2][col].count() for col in LM3.columns],
            name = 'Do Second'),
        go.Bar(
            orientation='h',
            y = [re.findall(r'--- (.*)', col)[0] for col in LM3.columns],
            x = [LM3[LM3[col]==3][col].count() for col in LM3.columns],
            name = 'Do Third')
    ])
    LM3_fig.update_layout(
        barmode='stack',
        height=800,
        title_text="Please rank in order the things you will do first when things get back to normal. Rank up to 3."
        )

    # Outlook over next 12 months
    outlook = pd.DataFrame()
    for col in ['S5','S6','S7']:
        outlook[col] = dff[col].value_counts().sort_index()

    outlook = outlook/outlook[outlook.columns].sum()*100
    outlook.columns = ['Financial Future', 'Physical Wellbeing', 'Mental Wellbeing']

    outlook_fig = go.Figure()
    for col in outlook.columns:
        for value in outlook[col]:
            outlook_fig.add_trace(go.Bar(
                orientation='h',
                x = [value],
                y = [col])
            )

    top_labels = ['Very Pessimistic'] + ['']*9 + ['Very Optimistic']

    annotations = []

    for yd, xd in zip(outlook.columns, outlook.T.values):
        # labeling the y-axis
        annotations.append(dict(xref='paper', yref='y',
                                x=0.14, y=yd,
                                xanchor='right',
                                text=str(yd),
                                font=dict(family='Arial', size=14,
                                        color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        # labeling the first percentage of each bar (x_axis)
        annotations.append(dict(xref='x', yref='y',
                                x=xd[0] / 2, y=yd,
                                text=str(int(xd[0])) + '%',
                                font=dict(family='Arial', size=14,
                                        color='rgb(248, 248, 255)'),
                                showarrow=False))
        # labeling the first Likert scale (on the top)
        if yd == outlook.columns[-1]:
            annotations.append(dict(xref='x', yref='paper',
                                    x=xd[0] / 2, y=1.1,
                                    text=top_labels[0],
                                    font=dict(family='Arial', size=14,
                                            color='rgb(67, 67, 67)'),
                                    showarrow=False))
        space = xd[0]
        for i in range(1, len(xd)):
                # labeling the rest of percentages for each bar (x_axis)
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (xd[i]/2), y=yd,
                                        text=str(int(xd[i])) + '%',
                                        font=dict(family='Arial', size=14,
                                                color='rgb(248, 248, 255)'),
                                        showarrow=False))
                # labeling the Likert scale
                if yd == outlook.columns[-1]:
                    annotations.append(dict(xref='x', yref='paper',
                                            x=space + (xd[i]/2), y=1.1,
                                            text=top_labels[i],
                                            font=dict(family='Arial', size=14,
                                                    color='rgb(67, 67, 67)'),
                                            showarrow=False))
                space += xd[i]

    outlook_fig.update_layout(
        barmode='stack',
        title_text="Outlook over next 12 months",
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        annotations=annotations,
    )

    return LM2_fig, outlook_fig, LM3_fig

if __name__ == '__main__':
    app.run_server(debug=True)
