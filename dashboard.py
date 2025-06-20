import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional

from data_collector import DataCollector
from momentum_strategy import MomentumStrategy
from live_trader import LiveTrader

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Momentum Trading Dashboard"

# Global variables for data storage
trading_data = {
    'portfolio_values': [],
    'positions': {},
    'signals': pd.DataFrame(),
    'performance_metrics': {}
}

def create_layout():
    """Create the main dashboard layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("📈 Momentum Trading Dashboard", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        # Performance Overview Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Portfolio Value", className="card-title"),
                        html.H2(id="portfolio-value", children="$0", className="text-success"),
                        html.P(id="portfolio-change", children="+0.00%", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Return", className="card-title"),
                        html.H2(id="total-return", children="0.00%", className="text-primary"),
                        html.P("Since Inception", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Sharpe Ratio", className="card-title"),
                        html.H2(id="sharpe-ratio", children="0.00", className="text-info"),
                        html.P("Risk-Adjusted Return", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Active Positions", className="card-title"),
                        html.H2(id="active-positions", children="0", className="text-warning"),
                        html.P("Current Holdings", className="text-muted")
                    ])
                ])
            ], width=3)
        ], className="mb-4"),
        
        # Charts Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Portfolio Performance"),
                    dbc.CardBody([
                        dcc.Graph(id="portfolio-chart")
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Top Holdings"),
                    dbc.CardBody([
                        html.Div(id="holdings-table")
                    ])
                ])
            ], width=4)
        ], className="mb-4"),
        
        # Strategy Signals and Risk Metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Momentum Signals"),
                    dbc.CardBody([
                        dcc.Graph(id="signals-chart")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Risk Metrics"),
                    dbc.CardBody([
                        html.Div(id="risk-metrics")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Controls and Settings
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Trading Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Start Trading", id="start-btn", color="success", className="me-2"),
                                dbc.Button("Stop Trading", id="stop-btn", color="danger", className="me-2"),
                                dbc.Button("Refresh Data", id="refresh-btn", color="info")
                            ])
                        ]),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Symbols:"),
                                dcc.Dropdown(
                                    id="symbols-dropdown",
                                    options=[
                                        {'label': 'AAPL', 'value': 'AAPL'},
                                        {'label': 'MSFT', 'value': 'MSFT'},
                                        {'label': 'GOOGL', 'value': 'GOOGL'},
                                        {'label': 'TSLA', 'value': 'TSLA'},
                                        {'label': 'NVDA', 'value': 'NVDA'},
                                        {'label': 'META', 'value': 'META'},
                                        {'label': 'AMZN', 'value': 'AMZN'},
                                        {'label': 'NFLX', 'value': 'NFLX'},
                                        {'label': 'JPM', 'value': 'JPM'},
                                        {'label': 'V', 'value': 'V'}
                                    ],
                                    value=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
                                    multi=True
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Max Positions:"),
                                dcc.Slider(
                                    id="max-positions-slider",
                                    min=5, max=20, step=1, value=10,
                                    marks={i: str(i) for i in range(5, 21, 5)}
                                )
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ]),
        
        # Hidden div for storing data
        html.Div(id="data-store", style={"display": "none"}),
        
        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=30*1000,  # 30 seconds
            n_intervals=0
        )
    ], fluid=True)

app.layout = create_layout()

# Callbacks
@app.callback(
    [Output("portfolio-value", "children"),
     Output("portfolio-change", "children"),
     Output("total-return", "children"),
     Output("sharpe-ratio", "children"),
     Output("active-positions", "children")],
    [Input("interval-component", "n_intervals"),
     Input("refresh-btn", "n_clicks")]
)
def update_performance_metrics(n_intervals, refresh_clicks):
    """Update performance metrics."""
    try:
        # Load current state
        if os.path.exists("trading_data/current_state.json"):
            with open("trading_data/current_state.json", 'r') as f:
                state = json.load(f)
            
            portfolio_value = state.get('portfolio_value', 0)
            positions = state.get('current_positions', {})
            
            # Calculate metrics
            total_return = ((portfolio_value / 100000) - 1) * 100  # Assuming 100k initial
            
            return [
                f"${portfolio_value:,.2f}",
                f"{total_return:+.2f}%" if total_return != 0 else "0.00%",
                f"{total_return:.2f}%",
                "0.85",  # Placeholder - calculate from historical data
                str(len(positions))
            ]
        else:
            return ["$100,000.00", "0.00%", "0.00%", "0.00", "0"]
    except Exception as e:
        return ["$100,000.00", "0.00%", "0.00%", "0.00", "0"]

@app.callback(
    Output("portfolio-chart", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("refresh-btn", "n_clicks")]
)
def update_portfolio_chart(n_intervals, refresh_clicks):
    """Update portfolio performance chart."""
    try:
        # Load performance history
        if os.path.exists("trading_data/performance_history.csv"):
            perf_df = pd.read_csv("trading_data/performance_history.csv")
            perf_df['date'] = pd.to_datetime(perf_df['date'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=perf_df['date'],
                y=perf_df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                template="plotly_white"
            )
            
            return fig
        else:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Portfolio Value Over Time",
                template="plotly_white"
            )
            return fig
    except Exception as e:
        # Return empty chart on error
        fig = go.Figure()
        fig.add_annotation(
            text="Error loading data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@app.callback(
    Output("holdings-table", "children"),
    [Input("interval-component", "n_intervals"),
     Input("refresh-btn", "n_clicks")]
)
def update_holdings_table(n_intervals, refresh_clicks):
    """Update holdings table."""
    try:
        if os.path.exists("trading_data/current_state.json"):
            with open("trading_data/current_state.json", 'r') as f:
                state = json.load(f)
            
            positions = state.get('current_positions', {})
            
            if not positions:
                return html.P("No active positions", className="text-muted")
            
            # Create table rows
            rows = []
            for symbol, shares in positions.items():
                rows.append(html.Tr([
                    html.Td(symbol, className="fw-bold"),
                    html.Td(f"{shares:.2f}"),
                    html.Td("Active", className="text-success")
                ]))
            
            return dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Symbol"),
                        html.Th("Shares"),
                        html.Th("Status")
                    ])
                ]),
                html.Tbody(rows)
            ], striped=True, bordered=True, hover=True)
        else:
            return html.P("No position data available", className="text-muted")
    except Exception as e:
        return html.P("Error loading positions", className="text-danger")

@app.callback(
    Output("signals-chart", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("refresh-btn", "n_clicks")]
)
def update_signals_chart(n_intervals, refresh_clicks):
    """Update momentum signals chart."""
    try:
        # This would typically load from the strategy's signal data
        # For now, create a sample chart
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        momentum_scores = np.random.randn(len(symbols)) * 0.5 + 0.1
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=momentum_scores,
                marker_color=['green' if x > 0 else 'red' for x in momentum_scores]
            )
        ])
        
        fig.update_layout(
            title="Current Momentum Signals",
            xaxis_title="Symbol",
            yaxis_title="Momentum Score",
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text="Error loading signals",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@app.callback(
    Output("risk-metrics", "children"),
    [Input("interval-component", "n_intervals"),
     Input("refresh-btn", "n_clicks")]
)
def update_risk_metrics(n_intervals, refresh_clicks):
    """Update risk metrics display."""
    try:
        # Sample risk metrics
        metrics = [
            ("Max Drawdown", "-5.2%"),
            ("Volatility", "12.8%"),
            ("Beta", "0.95"),
            ("Sortino Ratio", "1.45"),
            ("VaR (95%)", "-2.1%"),
            ("Correlation", "0.78")
        ]
        
        rows = []
        for metric, value in metrics:
            rows.append(html.Tr([
                html.Td(metric, className="fw-bold"),
                html.Td(value, className="text-end")
            ]))
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Metric"),
                    html.Th("Value", className="text-end")
                ])
            ]),
            html.Tbody(rows)
        ], striped=True, bordered=True, hover=True)
    except Exception as e:
        return html.P("Error loading risk metrics", className="text-danger")

@app.callback(
    [Output("start-btn", "disabled"),
     Output("stop-btn", "disabled")],
    [Input("start-btn", "n_clicks"),
     Input("stop-btn", "n_clicks")]
)
def handle_trading_controls(start_clicks, stop_clicks):
    """Handle trading control buttons."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return False, True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "start-btn":
        # Start trading logic would go here
        return True, False
    elif button_id == "stop-btn":
        # Stop trading logic would go here
        return False, True
    
    return False, True

def run_dashboard(debug=True, port=8050):
    """Run the dashboard."""
    print(f"Starting dashboard on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    
    app.run_server(debug=debug, port=port)

def main():
    """Main function to run the dashboard."""
    print("Momentum Trading Dashboard")
    print("=" * 40)
    print("Starting dashboard...")
    
    try:
        run_dashboard()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    main() 