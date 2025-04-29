import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, dcc, no_update

def init_dashboard(server=None):
    # Load data
    df = pd.read_excel("uploads/predicted_churn_results.xlsx")

    # Map binary-encoded columns to labels
    df['Gender'] = df['Gender'].map({0: 'Male', 1: 'Female'})
    df['Partner'] = df['Partner'].map({0: 'No', 1: 'Yes'})
    df['Online Security'] = df['Online Security'].map({0: 'No', 1: 'Yes', 2: 'No internet service'})
    df['Paperless Billing'] = df['Paperless Billing'].map({0:'No', 1:'Yes'})
    df['Dependents'] = df['Dependents'].map({0:'No', 1:'Yes'})

    numeric_cols = df.select_dtypes(include='number').columns

    app = Dash(__name__, server=server, routes_pathname_prefix='/dashapp/')
    app.title = "Enhanced Customer Churn Dashboard"

    app.layout = html.Div([
        html.H1("Enhanced Customer Churn Dashboard", style={'textAlign': 'center'}),

        html.Div([
            html.Div([
                html.Label("Zip Code"),
                dcc.Dropdown(
                    options=[{"label": str(z), "value": z} for z in sorted(df["Zip Code"].unique())],
                    id="zip-dropdown", multi=True, placeholder="Select Zip Codes"
                )
            ], style={'width': '24%', 'display': 'inline-block'}),

            html.Div([
                html.Label("Gender"),
                dcc.Dropdown(
                    options=[{"label": g, "value": g} for g in df["Gender"].unique()],
                    id="gender-dropdown", multi=True, placeholder="Select Genders"
                )
            ], style={'width': '24%', 'display': 'inline-block'}),

            html.Div([
                html.Label("Partner"),
                dcc.Dropdown(
                    options=[{"label": p, "value": p} for p in df["Partner"].unique()],
                    id="partner-dropdown", multi=True, placeholder="Select Partner Status"
                )
            ], style={'width': '24%', 'display': 'inline-block'}),

            html.Div([
                html.Label("Online Security"),
                dcc.Dropdown(
                    options=[{"label": s, "value": s} for s in df["Online Security"].unique()],
                    id="security-dropdown", multi=True, placeholder="Select Online Security"
                )
            ], style={'width': '24%', 'display': 'inline-block'}),
        ], style={'padding': '10px'}),

        dcc.Tabs([
            dcc.Tab(label='Overview', children=[
                html.Div(id='summary-stats', style={'padding': '20px'}),
                dcc.Graph(id='churn-pie'),
                dcc.Graph(id='satisfaction-hist'),
            ]),
            dcc.Tab(label='Revenue Analysis', children=[
                dcc.Graph(id='revenue-line'),
                dcc.Graph(id='charges-vs-age')
            ]),
            dcc.Tab(label='Customer Insights', children=[
                dcc.Graph(id='correlation-heatmap')
            ]),
            dcc.Tab(label='Data Export', children=[
                html.Button("Download Filtered Data", id="btn-download"),
                dcc.Download(id="download-dataframe-csv")
            ])
        ])
    ])

    def filter_data(zip_vals, gender_vals, partner_vals, security_vals):
        temp = df.copy()
        if zip_vals:
            temp = temp[temp['Zip Code'].isin(zip_vals)]
        if gender_vals:
            temp = temp[temp['Gender'].isin(gender_vals)]
        if partner_vals:
            temp = temp[temp['Partner'].isin(partner_vals)]
        if security_vals:
            temp = temp[temp['Online Security'].isin(security_vals)]
        return temp

    @app.callback(
        Output('summary-stats', 'children'),
        Output('churn-pie', 'figure'),
        Output('satisfaction-hist', 'figure'),
        Output('revenue-line', 'figure'),
        Output('charges-vs-age', 'figure'),
        Output('correlation-heatmap', 'figure'),
        Input('zip-dropdown', 'value'),
        Input('gender-dropdown', 'value'),
        Input('partner-dropdown', 'value'),
        Input('security-dropdown', 'value')
    )
    def update_graphs(zip_vals, gender_vals, partner_vals, security_vals):
        filtered = filter_data(zip_vals, gender_vals, partner_vals, security_vals)

        avg_tenure = filtered["Tenure in Months"].mean()
        avg_charges = filtered["Monthly Charges"].mean()
        revenue = filtered["Total Revenue"].sum()
        summary = html.Div([
            html.H4(f"Average Tenure: {avg_tenure:.1f} months"),
            html.H4(f"Average Monthly Charges: ${avg_charges:.2f}"),
            html.H4(f"Total Revenue: ${revenue:,.2f}")
        ])

        churn_fig = px.pie(filtered, names='Predicted Churn', title='Churn Breakdown')
        sat_fig = px.histogram(filtered, x='Satisfaction Score', nbins=10, title='Satisfaction Scores')

        rev_line = px.line(filtered.sort_values('Tenure in Months'),
                           x='Tenure in Months', y='Total Revenue',
                           title='Revenue Over Customer Tenure')

        charges_age = px.scatter(filtered, x='Age', y='Monthly Charges',
                                 color='Predicted Churn', title='Monthly Charges vs Age')

        corr = filtered[numeric_cols].corr()
        heatmap = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Viridis'
        ))
        heatmap.update_layout(title="Correlation Matrix")

        return summary, churn_fig, sat_fig, rev_line, charges_age, heatmap

    @app.callback(
        Output("download-dataframe-csv", "data"),
        Input("btn-download", "n_clicks"),
        State('zip-dropdown', 'value'),
        State('gender-dropdown', 'value'),
        State('partner-dropdown', 'value'),
        State('security-dropdown', 'value'),
        prevent_initial_call=True
    )
    def download_filtered_data(n_clicks, zip_vals, gender_vals, partner_vals, security_vals):
        filtered = filter_data(zip_vals, gender_vals, partner_vals, security_vals)
        return dcc.send_data_frame(filtered.to_csv, "filtered_data.csv", index=False)

    return app