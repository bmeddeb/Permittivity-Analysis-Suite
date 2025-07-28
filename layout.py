# layout.py
from dash import html, dcc
import dash_bootstrap_components as dbc

layout = html.Div([
    # Header
    html.Div([
        html.H1([
            html.I(className="fas fa-wave-square me-3", style={'color': '#007bff'}),
            "Permittivity Analysis Suite"
        ], className="text-center mb-4 text-primary"),
        html.P(
            "Advanced Causality Check & Hybrid Debye-Lorentz Modeling",
            className="text-center text-muted lead"
        )
    ], className="container-fluid bg-light py-4 mb-4"),

    # Upload Section (full width now)
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H3([
                        html.I(className="fas fa-upload me-2"),
                        "Data Upload"
                    ], className="card-title text-primary"),
                    # Configuration button
                    html.Button([
                        html.I(className="fas fa-cogs me-2"),
                        "Analysis Configuration"
                    ], 
                    className="btn btn-outline-primary",
                    id="config-button",
                    **{"data-bs-toggle": "offcanvas", "data-bs-target": "#configOffcanvas"}
                    )
                ], className="d-flex justify-content-between align-items-center mb-3"),
                
                html.P(
                    "Upload a CSV file with [Frequency_GHz, Dk, Df]",
                    className="card-text text-muted"
                ),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt fa-2x mb-3 text-primary"),
                        html.Br(),
                        html.Div(
                            "Drag and drop or click to select a CSV file",
                            className="text-muted fw-bold"
                        )
                    ], className="d-flex flex-column align-items-center justify-content-center h-100"),
                    style={
                        "width": "100%", "height": "140px",
                        "borderWidth": "2px", "borderStyle": "dashed",
                        "borderRadius": "10px", "borderColor": "#007bff",
                        "backgroundColor": "#f8f9fa",
                        "cursor": "pointer"
                    },
                    multiple=False
                ),
                html.Small(
                    "Expected format: [Frequency_GHz, Dk, Df]",
                    className="form-text text-muted mt-2"
                )
            ], className="card-body")
        ], className="card shadow-sm mb-4")
    ], className="row"),

    # Hidden data store for model selection
    dcc.Store(id="model-selection-internal", data=["debye"]),

    # Off-canvas Configuration Panel
    html.Div([
        html.Div([
            html.Div([
                html.H4([
                    html.I(className="fas fa-cogs me-2"),
                    "Analysis Configuration"
                ], className="offcanvas-title text-primary"),
                html.Button([
                    html.I(className="fas fa-times")
                ], className="btn-close", **{"data-bs-dismiss": "offcanvas"})
            ], className="offcanvas-header"),
            
            html.Div([
                # Auto-Selection Mode Toggle
                html.Div([
                    html.Label("Analysis Mode:", className="form-label fw-bold"),
                    dcc.RadioItems(
                        id="analysis-mode",
                        options=[
                            {"label": " Auto-Select Best Model", "value": "auto"},
                            {"label": " Auto + Compare All Models", "value": "auto_compare"},
                            {"label": " Manual Model Selection", "value": "manual"}
                        ],
                        value="auto",
                        className="mb-3",
                        style={"display": "flex", "flexDirection": "column", "gap": "8px"}
                    )
                ], className="mb-3"),

                # Selection Method (for auto modes)
                html.Div([
                    html.Label("Selection Criteria:", className="form-label fw-bold"),
                    dcc.Dropdown(
                        id="selection-method",
                        options=[
                            {"label": "🎯 Balanced (AIC + RMSE)", "value": "balanced"},
                            {"label": "📊 Statistical Rigor (AIC-focused)", "value": "aic_focused"},
                            {"label": "🎯 Best Accuracy (RMSE-focused)", "value": "rmse_focused"}
                        ],
                        value="balanced",
                        clearable=False,
                        className="mb-3"
                    )
                ], id="selection-method-div", className="mb-3"),

                # Manual Model Selection with Switches (hidden by default)
                html.Div([
                    html.Label("Select Analysis Methods:", className="form-label fw-bold mb-3"),
                    html.Div([
                        dbc.Switch(id="switch-debye", label="Simple Debye Model", value=True),
                        dbc.Switch(id="switch-multipole", label="Multi-Pole Debye Model", value=False),
                        dbc.Switch(id="switch-cole-cole", label="Cole-Cole Model", value=False),
                        dbc.Switch(id="switch-cole-davidson", label="Cole-Davidson Model", value=False),
                        dbc.Switch(id="switch-havriliak", label="Havriliak-Negami Model", value=False),
                        dbc.Switch(id="switch-lorentz", label="Lorentz Oscillator Model", value=False),
                        dbc.Switch(id="switch-sarkar", label="D. Sarkar Model", value=False),
                        dbc.Switch(id="switch-hybrid", label="Hybrid Debye-Lorentz Model", value=False),
                        dbc.Switch(id="switch-kk", label="KK Causality Check", value=False)
                    ], className="d-flex flex-column gap-2")
                ], id="manual-selection-div", style={"display": "none"}, className="mb-3"),
                
                html.Div([
                    html.Label("Number of Terms for Hybrid Model:", className="form-label fw-bold"),
                    dcc.Slider(
                        id="n-terms-slider",
                        min=1, max=5, step=1, value=2,
                        marks={i: str(i) for i in range(1, 6)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="mb-3"),

                html.Div([
                    html.Label("Number of Poles for Multipole Debye:", className="form-label fw-bold"),
                    dcc.Slider(
                        id="multipole-terms-slider",
                        min=1, max=5, step=1, value=3,
                        marks={i: str(i) for i in range(1, 6)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="mb-3"),

                html.Div([
                    html.Label("Number of Oscillators for Lorentz Model:", className="form-label fw-bold"),
                    dcc.Slider(
                        id="lorentz-terms-slider",
                        min=1, max=4, step=1, value=2,
                        marks={i: str(i) for i in range(1, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="mb-3"),

                # Info note about parameter behavior
                html.Div([
                    html.Small([
                        html.I(className="fas fa-info-circle me-1"),
                        html.Strong("Note: "),
                        "Auto modes optimize N automatically (testing N=1-5 for each variable model). ",
                        "Sliders only apply in Manual mode."
                    ], className="text-info")
                ], className="mb-2")
            ], className="offcanvas-body")
        ], className="offcanvas offcanvas-end", id="configOffcanvas")
    ]),

    # Results Section
    html.Div([
        html.Div([
            html.H3([
                html.I(className="fas fa-chart-area me-2"),
                "Analysis Results"
            ], className="card-title text-primary"),

            html.Div(id="results-summary", className="mb-3"),

            dcc.Graph(id="permittivity-plots", style={"height": "700px"}),

            html.Div(id="download-section", className="mt-3")
        ], className="card-body")
    ], className="card shadow-sm mb-4", id="results-card")
], className="container", style={"maxWidth": "1200px"})
