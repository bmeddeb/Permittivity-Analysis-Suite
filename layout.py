# layout.py
from dash import html, dcc
import dash_bootstrap_components as dbc

layout = html.Div([
    # Header - Compact version
    html.Div([
        html.H3([
            html.I(className="fas fa-wave-square me-2", style={'color': '#007bff', 'fontSize': '1.2rem'}),
            "Permittivity Analysis Suite"
        ], className="text-center mb-1 text-primary", style={'fontSize': '1.5rem', 'fontWeight': '600'}),
        html.P(
            "Advanced Causality Check & Hybrid Debye-Lorentz Modeling",
            className="text-center text-muted",
            style={'fontSize': '0.9rem', 'marginBottom': '0.5rem'}
        )
    ], className="container-fluid bg-light py-2 mb-3"),

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
                        html.I(className="fas fa-cloud-upload-alt fa-lg mb-2 text-primary"),
                        html.Br(),
                        html.Div(
                            "Drag and drop or click to select a CSV file",
                            className="text-muted fw-bold",
                            style={"fontSize": "0.9rem"}
                        )
                    ], className="d-flex flex-column align-items-center justify-content-center h-100"),
                    style={
                        "width": "100%", "height": "100px",
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
                ),
                
                # Preprocessing Controls Section
                html.Hr(className="my-3"),
                html.Div([
                    html.H6([
                        html.I(className="fas fa-magic me-2"),
                        "Data Preprocessing"
                    ], className="text-primary mb-2"),
                    
                    html.P("Automatically analyze and smooth noisy data for better model fitting.", 
                           className="text-muted small mb-2"),
                    
                    # Preprocessing Mode Selection
                    dcc.RadioItems(
                        id="preprocessing-mode",
                        options=[
                            {"label": " Auto (Recommended)", "value": "auto"},
                            {"label": " Manual Selection", "value": "manual"},
                            {"label": " No Smoothing", "value": "none"}
                        ],
                        value="auto",
                        className="mb-2",
                        style={"fontSize": "0.9rem"}
                    ),
                    
                    # Auto Mode Settings (visible by default)
                    html.Div([
                        html.Label("Selection Method:", className="form-label small fw-bold"),
                        dcc.Dropdown(
                            id="preprocessing-selection-method",
                            options=[
                                {"label": "🎯 Hybrid (Rule + Quality)", "value": "hybrid"},
                                {"label": "⚡ Rule-Based (Fast)", "value": "rule_based"},
                                {"label": "🎪 Quality-Based (Thorough)", "value": "quality_based"}
                            ],
                            value="hybrid",
                            clearable=False,
                            style={"fontSize": "0.85rem"}
                        )
                    ], id="auto-preprocessing-div", className="mb-2"),
                    
                    # Manual Mode Settings (hidden by default)
                    html.Div([
                        html.Label("Smoothing Algorithm:", className="form-label small fw-bold"),
                        dcc.Dropdown(
                            id="smoothing-algorithm",
                            options=[
                                {"label": "🔬 Interpolating Spline (Clean Data)", "value": "interpolating_spline"},
                                {"label": "🌊 Smoothing Spline (Adaptive)", "value": "smoothing_spline"},
                                {"label": "📈 PCHIP (Shape Preserving)", "value": "pchip"},
                                {"label": "🛡️ LOWESS (Robust)", "value": "lowess"},
                                {"label": "📊 Savitzky-Golay", "value": "savitzky_golay"},
                                {"label": "🔲 Gaussian Filter", "value": "gaussian"},
                                {"label": "🔧 Median Filter", "value": "median"}
                            ],
                            value="smoothing_spline",
                            clearable=False,
                            style={"fontSize": "0.85rem"}
                        ),
                        
                        # Algorithm Information Panel
                        html.Div(id="algorithm-info", className="mt-2 p-2 border rounded", 
                                style={"backgroundColor": "#f8f9fa", "fontSize": "0.8rem"}),
                        
                        # Algorithm-specific Parameters
                        html.Div([
                            # Smoothing Spline Parameters
                            html.Div([
                                html.Label("Smoothing Factor (s):", className="form-label small"),
                                dcc.RadioItems(
                                    id="spline-s-mode",
                                    options=[
                                        {"label": " Auto (s = m×σ²)", "value": "auto"},
                                        {"label": " Manual", "value": "manual"}
                                    ],
                                    value="auto",
                                    className="mb-1",
                                    style={"fontSize": "0.75rem"}
                                ),
                                dcc.Input(
                                    id="spline-s-value",
                                    type="number",
                                    placeholder="0.001",
                                    min=0,
                                    step=0.001,
                                    className="form-control form-control-sm",
                                    style={"display": "none"}
                                )
                            ], id="spline-params", style={"display": "none"}, className="mt-2"),
                            
                            # LOWESS Parameters
                            html.Div([
                                html.Label("Fraction of data to use:", className="form-label small"),
                                dcc.Slider(
                                    id="lowess-frac",
                                    min=0.1,
                                    max=0.8,
                                    step=0.05,
                                    value=0.3,
                                    marks={0.1: "0.1", 0.3: "0.3", 0.5: "0.5", 0.8: "0.8"},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], id="lowess-params", style={"display": "none"}, className="mt-2"),
                            
                            # Savitzky-Golay Parameters
                            html.Div([
                                html.Label("Window Size:", className="form-label small"),
                                dcc.Slider(
                                    id="savgol-window",
                                    min=3,
                                    max=15,
                                    step=2,
                                    value=5,
                                    marks={3: "3", 5: "5", 7: "7", 9: "9", 11: "11", 13: "13", 15: "15"},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Label("Polynomial Order:", className="form-label small mt-2"),
                                dcc.Slider(
                                    id="savgol-polyorder",
                                    min=1,
                                    max=4,
                                    step=1,
                                    value=2,
                                    marks={1: "1", 2: "2", 3: "3", 4: "4"},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], id="savgol-params", style={"display": "none"}, className="mt-2"),
                            
                            # Gaussian Filter Parameters
                            html.Div([
                                html.Label("Sigma (smoothing strength):", className="form-label small"),
                                dcc.Slider(
                                    id="gaussian-sigma",
                                    min=0.5,
                                    max=3.0,
                                    step=0.1,
                                    value=1.0,
                                    marks={0.5: "0.5", 1.0: "1.0", 2.0: "2.0", 3.0: "3.0"},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], id="gaussian-params", style={"display": "none"}, className="mt-2"),
                            
                            # Median Filter Parameters
                            html.Div([
                                html.Label("Window Size:", className="form-label small"),
                                dcc.Slider(
                                    id="median-window",
                                    min=3,
                                    max=11,
                                    step=2,
                                    value=3,
                                    marks={3: "3", 5: "5", 7: "7", 9: "9", 11: "11"},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], id="median-params", style={"display": "none"}, className="mt-2")
                        ], id="algorithm-params-container")
                        
                    ], id="manual-preprocessing-div", style={"display": "none"}, className="mb-2"),
                    
                    # Preprocessing Preview Section (for uploaded data)
                    html.Div([
                        html.Hr(className="my-2"),
                        html.Button([
                            html.I(className="fas fa-eye me-1"),
                            "Preview Preprocessing"
                        ], id="preview-preprocessing-btn", className="btn btn-outline-info btn-sm",
                           disabled=True, style={"fontSize": "0.8rem"}),
                        html.Div(id="preprocessing-preview", className="mt-2"),
                        # Visualization control buttons (initially hidden)
                        html.Div([
                            html.Hr(className="my-2"),
                            html.Button([
                                html.I(className="fas fa-chart-line me-1"),
                                "Show Before/After Comparison"
                            ], id="show-comparison-btn", className="btn btn-outline-primary btn-sm me-2",
                               style={"fontSize": "0.8rem", "display": "none"}),
                            html.Button([
                                html.I(className="fas fa-wave-square me-1"),
                                "Show Noise Analysis"
                            ], id="show-noise-analysis-btn", className="btn btn-outline-info btn-sm",
                               style={"fontSize": "0.8rem", "display": "none"})
                        ], id="visualization-buttons", className="mt-2"),
                        # Containers for visualizations
                        html.Div(id="preprocessing-comparison-plot", className="mt-3"),
                        html.Div(id="noise-analysis-plot", className="mt-3")
                    ], id="preprocessing-preview-section", style={"display": "none"})
                ], className="mt-2")
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
            html.Div([
                html.H3([
                    html.I(className="fas fa-chart-area me-2"),
                    "Analysis Results"
                ], className="card-title text-primary mb-0"),
                html.Div([
                    html.Button([
                        html.I(className="fas fa-download me-2"),
                        "Download CSV"
                    ], id="download-csv-btn", className="btn btn-success btn-sm", disabled=True),
                    dcc.Download(id="download-csv")
                ])
            ], className="d-flex justify-content-between align-items-center mb-3"),

            # Preprocessing Results
            html.Div(id="preprocessing-summary", className="mb-3"),
            
            # Model Results
            html.Div(id="results-summary", className="mb-3"),

            dcc.Loading(
                id="loading-analysis",
                type="dot",
                color="#007bff",
                children=[
                    dcc.Graph(id="permittivity-plots", style={"height": "700px"})
                ],
                style={"height": "700px"}
            )
        ], className="card-body", style={"position": "relative"})
    ], className="card shadow-sm mb-4", id="results-card"),

], className="container", style={"maxWidth": "1200px"})
