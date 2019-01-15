import dash
from dash.dependencies import Output, Event,Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import json
import pandas as pd
import csv
X = deque(maxlen=20)
X.append(1)
Y = deque(maxlen=20)
Y.append(1)
#X=[]
#Y=[]
data=pd.read_csv(r'yahoo_1980_2018.csv')
#def get_data():
#    for x,y in zip(data['Date'],data['Close']):
#        yield (x,y)

get_data=((x,y) for x,y in zip(data['Date'],data['Close']))

get_data=iter(get_data)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

stop_n_click=0
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.layout = html.Div(
    [
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*1000*5
        ),
        html.Button('Stop', id='Stop-button'),

        dcc.Dropdown(id='anomly-or-not',
    options=[
        {'label': 'Anomaly', 'value': 'anomaly'},
        {'label': 'Not Anomaly', 'value': 'not_anomaly'},
        
    ],
    value='not_anomaly'
)


                
    ]
)
fields=['Type','X','Y','Anom_point']
with open(r'anomaly1.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
#@app.callback(
#    Output('selected-data', 'children'),
#    [Input('basic-interactions', 'selectedData')])
#def display_selected_data(selectedData):
#    return json.dumps(selectedData, indent=2)
def save_data(anom,X1,Y1,x):
    with open(r'anomaly1.csv', 'a') as f:
        writer = csv.writer(f)
        
        writer.writerow([f"{anom,X1,Y1,x}"])


@app.callback(Output('live-graph', 'figure'),
              [Input('Stop-button','n_clicks'),
               Input('live-graph','clickData'),
               Input('anomly-or-not','value')],
              events=[Event('graph-update', 'interval')])
def update_graph_scatter(n_clicks,clickData,anom):
#    global stop_n_click
    
#    if n_clicks==None:stop_n_click=0
    print('n_clicks',n_clicks)
    if clickData!=None:
#    print(json.dumps(clickData, indent=2),anom)
        x=[i['x'] for i in clickData['points']]
        y=[i['y'] for i in clickData['points']]
        
    else:
        x=[0]
        y=[0]
    if n_clicks!=None:
        if (1+n_clicks)%2==0:
#            X.append(X[-1]+1)
#            Y.append(Y[-1]+Y[-1]*random.uniform(-0.1,0.1))
#   
#    if n_clicks!=1:
            
            X1,Y1=next(get_data)
            save_data(anom,X1,Y1,x)
            X.append(X1)
            Y.append(Y1)
            data = plotly.graph_objs.Scatter(
                    x=list(X),
                    y=list(Y),
                    name='Scatter',
                    mode= 'lines+markers'
                    )
            anom=plotly.graph_objs.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker={'size': 12}
                    )
#            stop_n_click=n_clicks
#            print(data)
#            print(anom)
            
            return {'data': [data,anom],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                        yaxis=dict(range=[min(Y),max(Y)]),)}
    #    else:
    #        return 



if __name__ == '__main__':
    app.run_server(debug=True)
