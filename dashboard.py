import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from pandas_datareader import data as web
from datetime import datetime as dt

# Set up global variables
stockpricedf = 0
financialreportingdf =0
discountrate=0.2
margin = 0.15


# Set up the app
app = dash.Dash(__name__)
server = app.server

app.css.append_css({
    "external_url":"https://codepen.io/chriddyp/pen/bWLwgP.css"
})

def portfolio_construction():
    dictlist=[]
    dictlist.append({'value':'Risk Parity', 'label':'RP'})
    dictlist.append({'value':'Max Sharpe Ratio', 'label':'MSR'})
    dictlist.append({'value':'Mean-Variance Optimization', 'label':'MVO'})
    return dictlist

def test(selected_dropdown_value1,selected_dropdown_value2,selected_dropdown_value3,selected_dropdown_value_mix,risk_limit):
    return "Strategy:" + selected_dropdown_value1 +' & ' +selected_dropdown_value2+' & '+selected_dropdown_value3+' ** '+risk_limit + '     Total:' + selected_dropdown_value_mix

app.layout = html.Div([
    html.Div([
        html.H2("Robo Advisor"),
        html.Img(src="/assets/UofT_logo.png")
    ], className="banner"),

    html.Div([
        html.H2("Risk Tolerance (please enter a volatility restraint)"),
        dcc.Input(id="risk-limit-input", value="0.5", type="text")
        # html.Button(id="submit-button", n_clicks=0, children="Submit")
    ]),

    html.Div([

        html.H2('Sub Class Optimization'),
        # First let users choose stocks
        html.H3('Choose an algorithm for equity class'),
        dcc.Dropdown(
            id='my-dropdown1',
            #options=save_sp500_stocks_info()+save_self_stocks_info(),
            options=portfolio_construction(),
            value='Risk Parity'),
        html.H3('Choose an algorithm for credit class'),
        dcc.Dropdown(
            id='my-dropdown2',
            #options=save_sp500_stocks_info()+save_self_stocks_info(),
            options=portfolio_construction(),
            value='Risk Parity'),
        html.H3('Choose an algorithm for PE class'),
        dcc.Dropdown(
            id='my-dropdown3',
            #options=save_sp500_stocks_info()+save_self_stocks_info(),
            options=portfolio_construction(),
            value='Risk Parity'),
        html.H4('Test Output'),
        html.Table(id = 'my-table'),
        html.P('')],style={'width': '40%', 'display': 'inline-block'}),
        html.Div([
        html.H2('Choose an algorithm for total portfolio'),
        dcc.Dropdown(
            id='mix-dropdown',
            #options=save_sp500_stocks_info()+save_self_stocks_info(),
            options=portfolio_construction(),
            value='Risk Parity'),
        html.P(''),
        html.H2("Performance Analysis"),
        html.Img(src="/assets/stock-icon.png",style={'height': '10%','width': '10%', 'display': 'inline-block'})
        ], style={'width': '55%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([
        html.H2('Risk Model'),

        html.H3("Stress Testing"),
        html.Img(src="/assets/stock-icon.png",style={'height': '10%','width': '10%', 'display': 'inline-block'})
        ], style={'width': '40%', 'display': 'inline-block'})

    ])



# For the stocks graph
# @app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown_value):
#     global stockpricedf # Needed to modify global copy of stockpricedf
#     stockpricedf = web.DataReader(
#         selected_dropdown_value.strip(), data_source='yahoo',
#         start=dt(2013, 1, 1), end=dt.now())
#     return {
#         'data': [{
#             'x': stockpricedf.index,
#             'y': stockpricedf.Close
#         }]
#     }


# for the table
@app.callback(Output('my-table', 'children'),
            [Input('my-dropdown1', 'value'),
             Input('my-dropdown2','value'),
             Input('my-dropdown3','value'),
             Input('mix-dropdown','value'),
             Input("risk-limit-input",'value')])
def generate_table(selected_dropdown_value1,selected_dropdown_value2,selected_dropdown_value3,selected_dropdown_value_mix,risk_limit):
    # global financialreportingdf # Needed to modify global copy of financialreportingdf
    test_v = test(selected_dropdown_value1,selected_dropdown_value2,selected_dropdown_value3,selected_dropdown_value_mix,risk_limit)

    # Header
    # return [html.Tr([html.Th(col) for col in financialreportingwritten.columns])] + [html.Tr([
    #     html.Td(financialreportingwritten.iloc[i][col]) for col in financialreportingwritten.columns
    # ]) for i in range(min(len(financialreportingwritten), max_rows))]
    return [html.Tr(html.Th(test_v))]




if __name__ == '__main__':
    app.run_server(debug=False)
