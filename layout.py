# layout.py
from dash import html, dcc

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

    # Row with Upload & Config Side by Side
    html.Div([
        # Upload (6-width)
        html.Div([
            html.Div([
                html.H3([
                    html.I(className="fas fa-upload me-2"),
                    "Data Upload"
                ], className="card-title text-primary"),
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
        ], className="card shadow-sm mb-4 col-md-6"),

        # Analysis Config (6-width)
        html.Div([
            html.Div([
                html.H3([
                    html.I(className="fas fa-cogs me-2"),
                    "Analysis Configuration"
                ], className="card-title text-primary"),
                html.Div([
                    html.Label("Select Analysis Methods:", className="form-label fw-bold"),
                    dcc.Checklist(
                        id="model-selection",
                        options=[
                            {"label": "Simple Debye Model", "value": "debye"},
                            {"label": "Multi-Pole Debye Model", "value": "multipole_debye"},
                            {"label": "Cole-Cole Model", "value": "cole_cole"},
                            {"label": "Cole-Davidson Model", "value": "cole_davidson"},
                            {"label": "Havriliak-Negami Model", "value": "havriliak_negami"},
                            {"label": "Lorentz Oscillator Model", "value": "lorentz"},
                            {"label": "D. Sarkar Model", "value": "sarkar"},
                            {"label": "Hybrid Debye-Lorentz Model", "value": "hybrid"},
                            {"label": "KK Causality Check", "value": "kk"}
                        ],
                        value=["debye"]
                    )

                ]),
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
                ])
            ], className="card-body")
        ], className="card shadow-sm mb-4 col-md-6")
    ], className="row"),

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
