import dash
from dash.dependencies import Output, Event,Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
from plotly import tools
import random
import plotly.graph_objs as go
from collections import deque
import json
import pandas as pd
import csv
from datetime import datetime
import numpy as np
from datetime import timedelta

#class DataGenerator:
#    def __init__(self, df,cont_names=['BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve']):
#        self.df = df
#        self.cont_names=cont_names
#
#    def __iter__(self):
#        self.n=0
#        return self
#
#    def __next__(self):
#        if self.n > self.df.shape[0]:
#            raise StopIteration
#
#        
#        return self.df.loc[self.n,'@timestamp'],self.df.loc[self.n,self.cont_names].tolist()


def DataGenerator(meric_list):
    for i in data[['@timestamp']+meric_list].iterrows():
        yield i[1]
        


##raw_data=pd.read_feather(r'D:\windstream_official\Anomaly_detection\data\processed\raw_df_feather')
#
#get_data1=((x,y) for x,y in zip(data['@timestamp'],data['BerPreFecMax']))
#get_data2=((x,y) for x,y in zip(data['@timestamp'],data['PhaseCorrectionAve']))
#get_data3=((x,y) for x,y in zip(data['@timestamp'],data['PmdMin']))
#get_data4=((x,y) for x,y in zip(data['@timestamp'],data['Qmin']))
#get_data5=((x,y) for x,y in zip(data['@timestamp'],data['SoPmdAve']))
#
#
#get_data1=iter(get_data1)
#get_data2=iter(get_data2)
#get_data3=iter(get_data3)
#get_data4=iter(get_data4)
#get_data5=iter(get_data5)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

stop_n_click=0
app = dash.Dash(__name__)
app.config['suppress_callback_exceptions']=True
app.layout = html.Div(
    [
        html.Div([html.Div([dcc.Graph(id='m1', animate=True,
                  style={"position": "relative", "width": "800px", "height": "800px"}),
        dcc.Graph(id='m2', animate=True,
                  style={"position": "relative", "width": "800px", "height": "800px"})],className='row'),
        html.Div([dcc.Graph(id='m3', animate=True,
                  style={"position": "relative", "width": "800px", "height": "800px"}),
        dcc.Graph(id='m4', animate=True,
                  style={"position": "relative", "width": "800px", "height": "800px"})],className='row'),
        dcc.Graph(id='m5', animate=True,
                  style={"position": "relative", "width": "800px", "height": "800px"}),],className='column'),
                  
        dcc.Interval(
            id='graph-update',
            interval=1*1000
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

def line_plot(X,Y,name=None,anom=None):
    trace = plotly.graph_objs.Scatter(
                    x=list(X),
                    y=list(Y1),
                    name='Scatter',
                    mode= 'lines+markers'
                    )
    if anom:
        return {'data': [trace,anom],}
    
        
    return {'data': [trace],'layout' : go.Layout(xaxis=dict(range=[min(X)-timedelta(hours=1),max(X)+timedelta(hours=1)]),
                                                        yaxis=dict(range=[range_dict[name]['min'],range_dict[name]['max']]),)}
    
#    {'data': [trace],'layout' : go.Layout(xaxis=dict(range=[min(X)-timedelta(hours=1),max(X)+timedelta(hours=1)]),
#                                                        yaxis=dict(range=[range_dict[name]['min'],range_dict[name]['max']])}
    
#def get_graphs(X1,Y1,Y2,Y3,Y4,Y5,anom):
#    trace1 = go.Scatter(x=X1, y=Y1)
#    trace2 = go.Scatter(x=X1, y=Y2)
#    trace3 = go.Scatter(x=X1, y=Y3)
#    trace4 = go.Scatter(x=X1, y=Y4)
#    trace4 = go.Scatter(x=X1, y=Y5)
#    
#    fig = tools.make_subplots(rows=3, cols=2, subplot_titles=('Plot 1', 'Plot 2',
#                                                              'Plot 3', 'Plot 4'))
#    fig.append_trace([trace1,anom], 1, 1)
#    fig.append_trace(trace2, 1, 2)
#    fig.append_trace(trace3, 2, 1)
#    fig.append_trace(trace4, 2, 2)
#    fig.append_trace(trace4, 3, 1)
#
#    fig['layout']['xaxis1'].update(title='xaxis 1 title')
#    fig['layout']['xaxis2'].update(title='xaxis 2 title', )
#    fig['layout']['xaxis3'].update(title='xaxis 3 title', showgrid=False)
#    fig['layout']['xaxis4'].update(title='xaxis 4 title', )
#    
#    fig['layout']['yaxis1'].update(title='yaxis 1 title')
#    fig['layout']['yaxis2'].update(title='yaxis 2 title', )
#    fig['layout']['yaxis3'].update(title='yaxis 3 title', showgrid=False)
#    fig['layout']['yaxis4'].update(title='yaxis 4 title')
#    
#    fig['layout'].update(title='Customizing Subplot Axes')
#    return fig    
    
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

q_to_list=lambda x:list(x)

@app.callback(Output('m1', 'figure'),
              [Input('Stop-button','n_clicks'),
               Input('m1','clickData'),
               Input('anomly-or-not','value')],
              events=[Event('graph-update', 'interval')])
def update_graph_scatter1(n_clicks,clickData,anom):
#    global stop_n_click
    
#    if n_clicks==None:stop_n_click=0
    print('n_clicks',n_clicks)
    if clickData!=None:
#    print(json.dumps(clickData, indent=2),anom)
        Xa=[i['x'] for i in clickData['points']]
        Ya=[i['y'] for i in clickData['points']]
        
    else:
        Xa=[0]
        Ya=[0]
    if n_clicks!=None:
        if (1+n_clicks)%2==0:
            x,y=next(m1_gen)
            X.append(x)
            Y1.append(y)
            print(x,y)
            anom=plotly.graph_objs.Scatter(
                    x=Xa,
                    y=Ya,
                    mode='markers',
                    marker={'size': 12}
                    )
#            anom={'data': [anom],'layout' : go.Layout(xaxis=dict(range=[min(X)-timedelta(hours=1),max(X)+timedelta(hours=1)]),
#                                                        yaxis=dict(range=[min(Y1)-np.mean(Y1),max(Y1)+np.mean(Y1)]),)}
            return line_plot(X,Y1,'BerPreFecMax',anom)
@app.callback(Output('m2', 'figure'),
              [Input('Stop-button','n_clicks'),
               Input('m2','clickData'),
               Input('anomly-or-not','value')],
              events=[Event('graph-update', 'interval')])
def update_graph_scatter2(n_clicks,clickData,anom):
#    global stop_n_click
    
#    if n_clicks==None:stop_n_click=0
    print('n_clicks',n_clicks)
    if clickData!=None:
#    print(json.dumps(clickData, indent=2),anom)
        Xa=[i['x'] for i in clickData['points']]
        Ya=[i['y'] for i in clickData['points']]
        
    else:
        Xa=[0]
        Ya=[0]
    if n_clicks!=None:
        if (1+n_clicks)%2==0:
            x,y=next(m2_gen)
#            X.append(x)
            Y2.append(y)
            anom=plotly.graph_objs.Scatter(
                    x=Xa,
                    y=Ya,
                    mode='markers',
                    marker={'size': 12}
                    )
#            anom={'data': [anom],'layout' : go.Layout(xaxis=dict(range=[min(X)-timedelta(hours=1),max(X)+timedelta(hours=1)]),
#                                                        yaxis=dict(range=[min(Y1)-np.mean(Y1),max(Y1)+np.mean(Y1)]),)}
            return line_plot(X,Y2,'PhaseCorrectionAve',anom)
@app.callback(Output('m3', 'figure'),
              [Input('Stop-button','n_clicks'),
               Input('m3','clickData'),
               Input('anomly-or-not','value')],
              events=[Event('graph-update', 'interval')])
def update_graph_scatter3(n_clicks,clickData,anom):
#    global stop_n_click
    
#    if n_clicks==None:stop_n_click=0
    print('n_clicks',n_clicks)
    if clickData!=None:
#    print(json.dumps(clickData, indent=2),anom)
        Xa=[i['x'] for i in clickData['points']]
        Ya=[i['y'] for i in clickData['points']]
        
    else:
        Xa=[0]
        Ya=[0]
    if n_clicks!=None:
        if (1+n_clicks)%2==0:
            x,y=next(m3_gen)
#            X.append(x)
            Y3.append(y)
            anom=plotly.graph_objs.Scatter(
                    x=Xa,
                    y=Ya,
                    mode='markers',
                    marker={'size': 12}
                    )
#            anom={'data': [anom],'layout' : go.Layout(xaxis=dict(range=[min(X)-timedelta(hours=1),max(X)+timedelta(hours=1)]),
#                                                        yaxis=dict(range=[min(Y1)-np.mean(Y1),max(Y1)+np.mean(Y1)]),)}
            return line_plot(X,Y3,'PmdMin',anom)
        
@app.callback(Output('m4', 'figure'),
              [Input('Stop-button','n_clicks'),
               Input('m4','clickData'),
               Input('anomly-or-not','value')],
              events=[Event('graph-update', 'interval')])
def update_graph_scatter4(n_clicks,clickData,anom):
#    global stop_n_click
    
#    if n_clicks==None:stop_n_click=0
    print('n_clicks',n_clicks)
    if clickData!=None:
#    print(json.dumps(clickData, indent=2),anom)
        Xa=[i['x'] for i in clickData['points']]
        Ya=[i['y'] for i in clickData['points']]
        
    else:
        Xa=[0]
        Ya=[0]
    if n_clicks!=None:
        if (1+n_clicks)%2==0:
            x,y=next(m4_gen)
#            X.append(x)
            Y4.append(y)
            anom=plotly.graph_objs.Scatter(
                    x=Xa,
                    y=Ya,
                    mode='markers',
                    marker={'size': 12}
                    )
#            anom={'data': [anom],'layout' : go.Layout(xaxis=dict(range=[min(X)-timedelta(hours=1),max(X)+timedelta(hours=1)]),
#                                                        yaxis=dict(range=[min(Y1)-np.mean(Y1),max(Y1)+np.mean(Y1)]),)}
            return line_plot(X,Y4,'Qmin',anom)

@app.callback(Output('m5', 'figure'),
              [Input('Stop-button','n_clicks'),
               Input('m5','clickData'),
               Input('anomly-or-not','value')],
              events=[Event('graph-update', 'interval')])
def update_graph_scatter(n_clicks,clickData,anom):
#    global stop_n_click
    
#    if n_clicks==None:stop_n_click=0
    print('n_clicks',n_clicks)
    if clickData!=None:
#    print(json.dumps(clickData, indent=2),anom)
        Xa=[i['x'] for i in clickData['points']]
        Ya=[i['y'] for i in clickData['points']]
        
    else:
        Xa=[0]
        Ya=[0]
    if n_clicks!=None:
        if (1+n_clicks)%2==0:
            x,y=next(m5_gen)
#            X.append(x)
            Y5.append(y)
            anom=plotly.graph_objs.Scatter(
                    x=Xa,
                    y=Ya,
                    mode='markers',
                    marker={'size': 12}
                    )
#            anom={'data': [anom],'layout' : go.Layout(xaxis=dict(range=[min(X)-timedelta(hours=1),max(X)+timedelta(hours=1)]),
#                                                        yaxis=dict(range=[min(Y1)-np.mean(Y1),max(Y1)+np.mean(Y1)]),)}
            return line_plot(X,Y5,'SoPmdAve',anom)
        
#            x1,y1,y2,y3,y4,y5=next(dg)
#            print(x1,y1,y2,y3,y4,y5)
#            save_data(anom,X,Y1,Xa)
#            X.append(x1)
#            Y1.append(y1)
#            Y2.append(y2)
#            Y3.append(y3)
#            Y4.append(y4)
#            Y5.append(y5)
#            X_=list(X)
#            Y1_=q_to_list(Y1)
#            Y2_=q_to_list(Y2)
#            Y3_=q_to_list(Y3)
#            Y4_=q_to_list(Y4)
#            Y5_=q_to_list(Y5)
#            
#            for i in range(5):
#                m1=line_plot(X,Y1)
##            data = plotly.graph_objs.Scatter(
##                    x=list(X),
##                    y=list(Y1),
##                    name='Scatter',
##                    mode= 'lines+markers'
##                    )
#            anom=plotly.graph_objs.Scatter(
#                    x=Xa,
#                    y=Ya,
#                    mode='markers',
#                    marker={'size': 12}
#                    )
#            anom={'data': [anom],'layout' : go.Layout(xaxis=dict(range=[min(X)-timedelta(hours=1),max(X)+timedelta(hours=1)]),
#                                                        yaxis=dict(range=[min(Y1)-np.mean(Y1),max(Y1)+np.mean(Y1)]),)}
##            
#            fig=get_graphs(X_,Y1_,Y2_,Y3_,Y4_,Y5_,anom)
#            return fig
##            stop_n_click=n_clicks
#            print(data)
#            print(anom)
            
#            return {'data': [data,anom],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
#                                                        yaxis=dict(range=[min(Y),max(Y)]),)}



app.css.append_css({
    'external_url': ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                     'https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css',]
})
if __name__ == '__main__':
    data=pd.read_feather(r'D:\windstream_official\Anomaly_detection\data\processed\13L110')
    data=data.sort_values(by='@timestamp')
    
    
    
    X = deque(maxlen=20)
    X.append(data['@timestamp'][0])
    Y1 = deque(maxlen=20)
    Y1.append(data['BerPreFecMax'][0])
    Y2 = deque(maxlen=20)
    Y2.append(data['PhaseCorrectionAve'][0])
    Y3 = deque(maxlen=20)
    Y3.append(data['PmdMin'][0])
    Y4 = deque(maxlen=20)
    Y4.append(data['Qmin'][0])
    Y5 = deque(maxlen=20)
    Y5.append(data['SoPmdAve'][0])
    stats=data.describe()
    range_dict={}
    for col in ['BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve']:
        range_dict[col]={'min':stats.loc['min',col],'max':stats.loc['max',col]}
    dg1=DataGenerator(['BerPreFecMax'])
    m1_gen=iter(dg1)
    
    dg2=DataGenerator(['PhaseCorrectionAve'])
    m2_gen=iter(dg2)
    dg3=DataGenerator(['PmdMin'])
    m3_gen=iter(dg3)
    dg4=DataGenerator(['Qmin'])
    m4_gen=iter(dg4)
    dg5=DataGenerator(['SoPmdAve'])
    m5_gen=iter(dg5)
    
    app.run_server(debug=True)





















