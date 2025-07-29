import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from flask_login import current_user
from flask import request
import numpy as np

def create_dash_app(server):
    """Create and configure the Dash application."""
    
    # Initialize Dash app with Flask server
    dash_app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname='/dashboard/',
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
        ]
    )
    
    # Sample data for demonstration
    def generate_sample_data():
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        return pd.DataFrame({
            'date': dates,
            'sales': np.random.normal(1000, 200, 30),
            'visitors': np.random.normal(500, 100, 30),
            'conversion_rate': np.random.normal(0.05, 0.01, 30)
        })
    
    # Define the layout function to access current user dynamically
    def serve_layout():
        # Get user info safely with fallbacks
        username = current_user.username if current_user and current_user.is_authenticated else "User"
        
        return html.Div([
            # Navigation bar (matching Flask base template)
            html.Nav([
                html.Div([
                    html.A('Flask App', href='/', className='navbar-brand'),
                    html.Div([
                        html.Span(f'Hello, {username}!', className='navbar-text me-3'),
                        html.A('Logout', href='/auth/logout', className='nav-link')
                    ], className='navbar-nav ms-auto')
                ], className='container')
            ], className='navbar navbar-expand-lg navbar-dark bg-dark'),
            
            # Main dashboard content
            html.Div([
                html.H1('Interactive Dashboard', className='mb-4'),
                html.P(f'Welcome to your personalized dashboard, {username}!', className='lead'),
                
                # Controls section
                html.Div([
                    html.Div([
                        html.Label('Select Metric:', className='form-label'),
                        dcc.Dropdown(
                            id='metric-dropdown',
                            options=[
                                {'label': 'Sales', 'value': 'sales'},
                                {'label': 'Visitors', 'value': 'visitors'},
                                {'label': 'Conversion Rate', 'value': 'conversion_rate'}
                            ],
                            value='sales',
                            className='mb-3'
                        )
                    ], className='col-md-4'),
                    
                    html.Div([
                        html.Label('Chart Type:', className='form-label'),
                        dcc.RadioItems(
                            id='chart-type',
                            options=[
                                {'label': 'Line Chart', 'value': 'line'},
                                {'label': 'Bar Chart', 'value': 'bar'}
                            ],
                            value='line',
                            className='mb-3'
                        )
                    ], className='col-md-4'),
                    
                    html.Div([
                        html.Button('Refresh Data', id='refresh-btn', 
                                   className='btn btn-primary mt-4')
                    ], className='col-md-4')
                ], className='row mb-4'),
                
                # Charts section
                html.Div([
                    html.Div([
                        dcc.Graph(id='main-chart')
                    ], className='col-lg-8'),
                    
                    html.Div([
                        html.Div([
                            html.H5('Key Metrics', className='card-header'),
                            html.Div(id='metrics-summary', className='card-body')
                        ], className='card mb-3'),
                        
                        html.Div([
                            html.H5('User Info', className='card-header'),
                            html.Div([
                                html.P(f'Username: {current_user.username if current_user and current_user.is_authenticated else "N/A"}'),
                                html.P(f'Email: {current_user.email if current_user and current_user.is_authenticated else "N/A"}'),
                                html.P(f'User ID: {current_user.id if current_user and current_user.is_authenticated else "N/A"}')
                            ], className='card-body')
                        ], className='card')
                    ], className='col-lg-4')
                ], className='row'),
                
                # Additional chart
                html.Div([
                    html.H3('Correlation Analysis', className='mt-4 mb-3'),
                    dcc.Graph(id='correlation-chart')
                ], className='row')
                
            ], className='container mt-4')
        ])
    
    # Set the layout to use the function for dynamic user data
    dash_app.layout = serve_layout
    
    # Callbacks
    @dash_app.callback(
        [Output('main-chart', 'figure'),
         Output('metrics-summary', 'children'),
         Output('correlation-chart', 'figure')],
        [Input('metric-dropdown', 'value'),
         Input('chart-type', 'value'),
         Input('refresh-btn', 'n_clicks')]
    )
    def update_dashboard(selected_metric, chart_type, n_clicks):
        # Generate fresh data
        df = generate_sample_data()
        
        # Main chart
        if chart_type == 'line':
            main_fig = px.line(df, x='date', y=selected_metric, 
                              title=f'{selected_metric.replace("_", " ").title()} Over Time')
        else:
            main_fig = px.bar(df, x='date', y=selected_metric,
                             title=f'{selected_metric.replace("_", " ").title()} Over Time')
        
        main_fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        # Metrics summary
        metrics_summary = [
            html.P(f'Avg Sales: ${df["sales"].mean():.2f}'),
            html.P(f'Avg Visitors: {df["visitors"].mean():.0f}'),
            html.P(f'Avg Conversion: {df["conversion_rate"].mean():.2%}'),
            html.Hr(),
            html.P(f'Max Sales: ${df["sales"].max():.2f}'),
            html.P(f'Min Sales: ${df["sales"].min():.2f}')
        ]
        
        # Correlation chart
        correlation_fig = px.scatter(df, x='visitors', y='sales', 
                                   size='conversion_rate',
                                   title='Sales vs Visitors (size = conversion rate)',
                                   trendline='ols')
        correlation_fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return main_fig, metrics_summary, correlation_fig
    
    return dash_app