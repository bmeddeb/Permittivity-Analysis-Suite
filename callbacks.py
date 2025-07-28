# callbacks.py - Updated for multiple sliders
from dash import Input, Output, State, html, dcc
import pandas as pd
import io
from utils.helpers import parse_contents
from analysis import run_analysis
from plots import create_permittivity_plot


def create_preprocessing_summary(preprocessing_info):
    """Create HTML summary of preprocessing results"""
    if not preprocessing_info or preprocessing_info.get('status') == 'insufficient_data':
        return html.Div()
    
    # Data quality assessment
    noise_metrics = preprocessing_info.get('noise_metrics', {})
    noise_score = noise_metrics.get('overall_noise_score', 0)
    
    # Quality badge
    if noise_score < 0.2:
        quality_badge = html.Span("Excellent", className="badge bg-success")
        quality_icon = "fas fa-check-circle text-success"
    elif noise_score < 0.4:
        quality_badge = html.Span("Good", className="badge bg-primary")
        quality_icon = "fas fa-thumbs-up text-primary"
    elif noise_score < 0.7:
        quality_badge = html.Span("Fair", className="badge bg-warning")
        quality_icon = "fas fa-exclamation-triangle text-warning"
    else:
        quality_badge = html.Span("Poor", className="badge bg-danger")
        quality_icon = "fas fa-times-circle text-danger"
    
    # Processing status
    preprocessing_mode = preprocessing_info.get('preprocessing_mode', 'none')
    smoothing_applied = preprocessing_info.get('smoothing_applied', False)
    
    if preprocessing_mode == 'none':
        processing_text = "No preprocessing applied"
        processing_icon = "fas fa-ban text-muted"
    elif smoothing_applied:
        dk_alg = preprocessing_info.get('dk_algorithm', 'Unknown')
        processing_text = f"{dk_alg.replace('_', ' ').title()} applied"
        processing_icon = "fas fa-magic text-success"
    else:
        processing_text = "No smoothing needed"
        processing_icon = "fas fa-check text-success"
    
    # Build summary
    summary_items = [
        # Data Quality Row
        html.Div([
            html.I(className=quality_icon + " me-2"),
            html.Strong("Data Quality: "),
            quality_badge,
            html.Small(f" (Noise Score: {noise_score:.2f})", className="text-muted ms-2")
        ], className="mb-2"),
        
        # Processing Status Row
        html.Div([
            html.I(className=processing_icon + " me-2"),
            html.Strong("Preprocessing: "),
            html.Span(processing_text, className="text-dark")
        ], className="mb-2")
    ]
    
    # Add recommendations if available
    recommendations = preprocessing_info.get('recommendations', [])
    if recommendations:
        rec_text = "; ".join(recommendations)
        summary_items.append(
            html.Div([
                html.I(className="fas fa-lightbulb text-info me-2"),
                html.Small(rec_text, className="text-info")
            ], className="mb-1")
        )
    
    # Add detailed metrics for transparency (collapsible)
    if noise_metrics:
        details_items = []
        
        # SNR metrics
        dk_snr = noise_metrics.get('dk_spectral_snr', 0)
        df_snr = noise_metrics.get('df_spectral_snr', 0)
        if dk_snr > 0:
            details_items.append(f"Dk SNR: {dk_snr:.1f} dB")
        if df_snr > 0:
            details_items.append(f"Df SNR: {df_snr:.1f} dB")
        
        # Roughness metrics
        dk_rough = noise_metrics.get('dk_roughness', 0)
        df_rough = noise_metrics.get('df_roughness', 0)
        if dk_rough > 0:
            details_items.append(f"Dk Roughness: {dk_rough:.3f}")
        if df_rough > 0:
            details_items.append(f"Df Roughness: {df_rough:.3f}")
        
        # Additional metrics
        dk_var = noise_metrics.get('dk_derivative_var', 0)
        df_var = noise_metrics.get('df_derivative_var', 0)
        if dk_var > 0:
            details_items.append(f"Dk Derivative Variance: {dk_var:.4f}")
        if df_var > 0:
            details_items.append(f"Df Derivative Variance: {df_var:.4f}")
        
        if details_items:
            summary_items.append(
                html.Details([
                    html.Summary("Show Detailed Metrics", className="text-muted small"),
                    html.Div([
                        html.Small(item, className="text-muted d-block") 
                        for item in details_items
                    ], className="ms-3 mt-1")
                ], className="mt-2")
            )
    
    # Add preprocessing impact visualization if data was smoothed
    if preprocessing_info.get('smoothing_applied', False):
        summary_items.append(
            html.Div([
                html.Hr(className="my-2"),
                html.Div([
                    html.I(className="fas fa-chart-bar text-success me-2"),
                    html.Strong("Noise Reduction Impact:", className="text-success")
                ], className="mb-2"),
                html.Div([
                    # Create simple progress bar for impact visualization
                    html.Div([
                        html.Small("Data Quality Improvement:", className="text-muted d-block"),
                        html.Div([
                            html.Div(
                                style={
                                    "width": f"{min(100, max(10, (1 - noise_score) * 100))}%",
                                    "backgroundColor": "#28a745" if noise_score < 0.3 else "#ffc107" if noise_score < 0.6 else "#dc3545",
                                    "height": "8px",
                                    "borderRadius": "4px"
                                }
                            )
                        ], className="bg-light", style={"height": "8px", "borderRadius": "4px", "width": "150px"}),
                        html.Small(f"Quality Score: {(1-noise_score)*100:.0f}%", className="text-success small")
                    ], className="mb-2")
                ])
            ], className="mt-2")
        )
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-magic me-2"),
            html.Strong("Data Preprocessing", className="text-primary")
        ], className="mb-2"),
        html.Div(summary_items)
    ], className="border rounded p-3 mb-3 bg-light")


def register_callbacks(app):
    # Callback to show/hide manual model selection based on analysis mode
    @app.callback(
        [
            Output("manual-selection-div", "style"),
            Output("selection-method-div", "style"),
        ],
        [Input("analysis-mode", "value")]
    )
    def toggle_selection_ui(analysis_mode):
        if analysis_mode == "manual":
            # Show manual selection, hide selection method
            return {"display": "block"}, {"display": "none"}
        else:
            # Hide manual selection, show selection method
            return {"display": "none"}, {"display": "block"}
    
    # Callback to show/hide preprocessing UI elements
    @app.callback(
        [
            Output("auto-preprocessing-div", "style"),
            Output("manual-preprocessing-div", "style"),
        ],
        [Input("preprocessing-mode", "value")]
    )
    def toggle_preprocessing_ui(preprocessing_mode):
        if preprocessing_mode == "manual":
            # Show manual algorithm selection, hide auto settings
            return {"display": "none"}, {"display": "block"}
        elif preprocessing_mode == "auto":
            # Show auto settings, hide manual selection
            return {"display": "block"}, {"display": "none"}
        else:  # preprocessing_mode == "none"
            # Hide both
            return {"display": "none"}, {"display": "none"}
    
    # Callback to show/hide algorithm-specific parameters based on selected algorithm
    @app.callback(
        [
            Output("spline-params", "style"),
            Output("lowess-params", "style"),
            Output("savgol-params", "style"),
            Output("gaussian-params", "style"),
            Output("median-params", "style"),
        ],
        [Input("smoothing-algorithm", "value")]
    )
    def toggle_algorithm_params(algorithm):
        # Default: hide all
        styles = [{"display": "none"} for _ in range(5)]
        
        if algorithm == "smoothing_spline":
            styles[0] = {"display": "block"}  # spline-params
        elif algorithm == "lowess":
            styles[1] = {"display": "block"}  # lowess-params
        elif algorithm == "savitzky_golay":
            styles[2] = {"display": "block"}  # savgol-params
        elif algorithm == "gaussian":
            styles[3] = {"display": "block"}  # gaussian-params
        elif algorithm == "median":
            styles[4] = {"display": "block"}  # median-params
        
        return styles
    
    # Callback to enable/disable preview button when data is uploaded
    @app.callback(
        [
            Output("preprocessing-preview-section", "style"),
            Output("preview-preprocessing-btn", "disabled"),
        ],
        [Input("upload-data", "contents")]
    )
    def toggle_preview_section(contents):
        if contents:
            return {"display": "block"}, False
        else:
            return {"display": "none"}, True
    
    # Callback to show/hide manual spline parameter input
    @app.callback(
        Output("spline-s-value", "style"),
        [Input("spline-s-mode", "value")]
    )
    def toggle_spline_manual_input(s_mode):
        if s_mode == "manual":
            return {"display": "block"}
        else:
            return {"display": "none"}
    
    # Callback to display algorithm information
    @app.callback(
        Output("algorithm-info", "children"),
        [Input("smoothing-algorithm", "value")]
    )
    def update_algorithm_info(algorithm):
        algorithm_descriptions = {
            "interpolating_spline": {
                "title": "🔬 Interpolating Spline",
                "description": "Cubic spline that passes exactly through all data points. Best for high-quality, clean data where preserving exact values is critical.",
                "use_case": "Ideal for: Clean experimental data, calibration curves",
                "note": "No smoothing - preserves all noise"
            },
            "smoothing_spline": {
                "title": "🌊 Smoothing Spline", 
                "description": "Adaptive cubic spline with automatic noise handling using the scientific s = m×σ² rule. Balances smoothness with data fidelity.",
                "use_case": "Ideal for: Most dielectric measurements, moderate noise",
                "note": "Recommended for typical dielectric data"
            },
            "pchip": {
                "title": "📈 PCHIP (Piecewise Cubic Hermite)",
                "description": "Shape-preserving interpolation that maintains monotonicity and prevents overshooting. Excellent for preserving physical trends.",
                "use_case": "Ideal for: Monotonic data, frequency sweeps",
                "note": "Preserves shape but minimal smoothing"
            },
            "lowess": {
                "title": "🛡️ LOWESS (Locally Weighted Scatterplot Smoothing)",
                "description": "Robust local regression that's resistant to outliers. Uses local weighted least squares with configurable neighborhood size.",
                "use_case": "Ideal for: Data with outliers, non-uniform noise",
                "note": "Most robust to measurement artifacts"
            },
            "savitzky_golay": {
                "title": "📊 Savitzky-Golay Filter",
                "description": "Polynomial smoothing filter that preserves peaks and features better than simple averaging. Maintains higher-order moments.",
                "use_case": "Ideal for: Spectroscopic data, peak preservation",
                "note": "Good balance of smoothing and feature retention"
            },
            "gaussian": {
                "title": "🔲 Gaussian Filter",
                "description": "Simple convolution with Gaussian kernel. Provides uniform smoothing across the frequency range with adjustable width.",
                "use_case": "Ideal for: Uniform noise, general smoothing",
                "note": "May blur important features at high sigma"
            },
            "median": {
                "title": "🔧 Median Filter",
                "description": "Non-linear filter that replaces each point with the median of its neighborhood. Excellent for removing spikes and impulse noise.",
                "use_case": "Ideal for: Spike removal, impulse noise",
                "note": "Best for outlier removal, minimal smoothing"
            }
        }
        
        if algorithm not in algorithm_descriptions:
            return html.Div("Select an algorithm to see information", className="text-muted")
        
        info = algorithm_descriptions[algorithm]
        
        return html.Div([
            html.Div([
                html.Strong(info["title"], className="text-primary"),
            ], className="mb-2"),
            html.Div(info["description"], className="mb-2"),
            html.Div([
                html.Strong("💡 ", className="text-info"),
                html.Span(info["use_case"], className="text-info small")
            ], className="mb-1"),
            html.Div([
                html.Strong("⚠️ ", className="text-warning"),
                html.Span(info["note"], className="text-warning small")
            ])
        ])
    
    # Callback for preprocessing preview
    @app.callback(
        [
            Output("preprocessing-preview", "children"),
            Output("show-comparison-btn", "style"),
            Output("show-noise-analysis-btn", "style"),
        ],
        [
            Input("preview-preprocessing-btn", "n_clicks"),
        ],
        [
            State("upload-data", "contents"),
            State("upload-data", "filename"),
            State("preprocessing-mode", "value"),
            State("preprocessing-selection-method", "value"),
            State("smoothing-algorithm", "value"),
            State("spline-s-mode", "value"),
            State("spline-s-value", "value"),
            State("lowess-frac", "value"),
            State("savgol-window", "value"),
            State("savgol-polyorder", "value"),
            State("gaussian-sigma", "value"),
            State("median-window", "value"),
        ],
        prevent_initial_call=True
    )
    def preview_preprocessing(n_clicks, contents, filename, preprocessing_mode, 
                            preprocessing_selection_method, smoothing_algorithm,
                            spline_s_mode, spline_s_value, lowess_frac, 
                            savgol_window, savgol_polyorder, gaussian_sigma, median_window):
        if not n_clicks or not contents:
            return html.Div(), {"fontSize": "0.8rem", "display": "none"}, {"fontSize": "0.8rem", "display": "none"}
        
        try:
            # Parse uploaded data
            df = parse_contents(contents, filename)
            
            # Set up preprocessing parameters
            from utils.enhanced_preprocessing import EnhancedDielectricPreprocessor
            preprocessor = EnhancedDielectricPreprocessor()
            
            manual_params = {
                'spline_s_mode': spline_s_mode,
                'spline_s_value': spline_s_value,
                'lowess_frac': lowess_frac,
                'savgol_window': savgol_window,
                'savgol_polyorder': savgol_polyorder,
                'gaussian_sigma': gaussian_sigma,
                'median_window': median_window
            }
            
            # Run preprocessing preview
            if preprocessing_mode == 'auto':
                _, preprocessing_info = preprocessor.preprocess(
                    df, apply_smoothing=True, selection_method=preprocessing_selection_method
                )
                preview_title = f"Auto Preprocessing Preview ({preprocessing_selection_method})"
            elif preprocessing_mode == 'manual':
                _, preprocessing_info = preprocessor.preprocess_manual(
                    df, smoothing_algorithm, manual_params
                )
                preview_title = f"Manual Preprocessing Preview ({smoothing_algorithm})"
            else:
                return (html.Div([
                    html.I(className="fas fa-info-circle text-info me-2"),
                    html.Span("No preprocessing will be applied.", className="text-muted")
                ], className="alert alert-info"), 
                {"fontSize": "0.8rem", "display": "none"}, 
                {"fontSize": "0.8rem", "display": "none"})
            
            # Create preview summary
            if preprocessing_info.get('status') == 'insufficient_data':
                return (html.Div([
                    html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                    html.Span("Insufficient data for preprocessing analysis.", className="text-warning")
                ], className="alert alert-warning"),
                {"fontSize": "0.8rem", "display": "none"}, 
                {"fontSize": "0.8rem", "display": "none"})
            
            noise_metrics = preprocessing_info.get('noise_metrics', {})
            noise_score = noise_metrics.get('overall_noise_score', 0)
            smoothing_applied = preprocessing_info.get('smoothing_applied', False)
            
            # Quality assessment
            if noise_score < 0.2:
                quality_color = "success"
                quality_text = "Excellent"
                quality_icon = "fas fa-check-circle"
            elif noise_score < 0.4:
                quality_color = "primary"
                quality_text = "Good"
                quality_icon = "fas fa-thumbs-up"
            elif noise_score < 0.7:
                quality_color = "warning"
                quality_text = "Fair"
                quality_icon = "fas fa-exclamation-triangle"
            else:
                quality_color = "danger"
                quality_text = "Poor"
                quality_icon = "fas fa-times-circle"
            
            preview_items = [
                html.H6(preview_title, className="text-primary mb-2"),
                html.Div([
                    html.I(className=f"{quality_icon} text-{quality_color} me-2"),
                    html.Strong("Data Quality: "),
                    html.Span(quality_text, className=f"badge bg-{quality_color}"),
                    html.Small(f" (Score: {noise_score:.3f})", className="text-muted ms-2")
                ], className="mb-2"),
            ]
            
            if smoothing_applied:
                algorithm = preprocessing_info.get('dk_algorithm', 'Unknown')
                preview_items.append(
                    html.Div([
                        html.I(className="fas fa-magic text-success me-2"),
                        html.Strong("Recommended Action: "),
                        html.Span(f"Apply {algorithm.replace('_', ' ').title()}", className="text-success")
                    ], className="mb-2")
                )
            else:
                preview_items.append(
                    html.Div([
                        html.I(className="fas fa-check text-success me-2"),
                        html.Strong("Recommended Action: "),
                        html.Span("No smoothing needed", className="text-success")
                    ], className="mb-2")
                )
            
            # Add recommendations
            recommendations = preprocessing_info.get('recommendations', [])
            if recommendations:
                preview_items.append(
                    html.Div([
                        html.I(className="fas fa-lightbulb text-info me-2"),
                        html.Strong("Recommendations: "),
                        html.Ul([
                            html.Li(rec, className="small") for rec in recommendations
                        ], className="mb-0 mt-1")
                    ], className="mb-2")
                )
            
            # Show visualization buttons
            button_style = {"fontSize": "0.8rem", "display": "inline-block"}
            
            return (html.Div(preview_items, className="border rounded p-3 bg-light"),
                   button_style, button_style)
            
        except Exception as e:
            return (html.Div([
                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                html.Span(f"Preview failed: {str(e)}", className="text-danger")
            ], className="alert alert-danger"),
            {"fontSize": "0.8rem", "display": "none"}, 
            {"fontSize": "0.8rem", "display": "none"})
    
    # Callback for before/after comparison visualization
    @app.callback(
        Output("preprocessing-comparison-plot", "children"),
        [Input("show-comparison-btn", "n_clicks")],
        [
            State("upload-data", "contents"),
            State("upload-data", "filename"),
            State("preprocessing-mode", "value"),
            State("preprocessing-selection-method", "value"),
            State("smoothing-algorithm", "value"),
            State("spline-s-mode", "value"),
            State("spline-s-value", "value"),
            State("lowess-frac", "value"),
            State("savgol-window", "value"),
            State("savgol-polyorder", "value"),
            State("gaussian-sigma", "value"),
            State("median-window", "value"),
        ],
        prevent_initial_call=True
    )
    def show_preprocessing_comparison(n_clicks, contents, filename, preprocessing_mode,
                                    preprocessing_selection_method, smoothing_algorithm,
                                    spline_s_mode, spline_s_value, lowess_frac,
                                    savgol_window, savgol_polyorder, gaussian_sigma, median_window):
        if not n_clicks or not contents:
            return html.Div()
        
        try:
            # Parse data and run preprocessing
            original_df = parse_contents(contents, filename)
            
            from utils.enhanced_preprocessing import EnhancedDielectricPreprocessor
            from plots import create_preprocessing_comparison_plot
            
            preprocessor = EnhancedDielectricPreprocessor()
            
            manual_params = {
                'spline_s_mode': spline_s_mode,
                'spline_s_value': spline_s_value,
                'lowess_frac': lowess_frac,
                'savgol_window': savgol_window,
                'savgol_polyorder': savgol_polyorder,
                'gaussian_sigma': gaussian_sigma,
                'median_window': median_window
            }
            
            # Run preprocessing
            if preprocessing_mode == 'auto':
                processed_df, preprocessing_info = preprocessor.preprocess(
                    original_df, apply_smoothing=True, selection_method=preprocessing_selection_method
                )
            elif preprocessing_mode == 'manual':
                processed_df, preprocessing_info = preprocessor.preprocess_manual(
                    original_df, smoothing_algorithm, manual_params
                )
            else:
                processed_df = original_df.copy()
                preprocessing_info = {'smoothing_applied': False}
            
            # Create comparison plot
            fig = create_preprocessing_comparison_plot(original_df, processed_df, preprocessing_info)
            
            return html.Div([
                dcc.Graph(figure=fig, style={"height": "500px"})
            ])
            
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                    html.Span(f"Visualization failed: {str(e)}", className="text-warning")
                ], className="alert alert-warning")
            ])
    
    # Callback for noise analysis visualization
    @app.callback(
        Output("noise-analysis-plot", "children"),
        [Input("show-noise-analysis-btn", "n_clicks")],
        [
            State("upload-data", "contents"),
            State("upload-data", "filename"),
        ],
        prevent_initial_call=True
    )
    def show_noise_analysis(n_clicks, contents, filename):
        if not n_clicks or not contents:
            return html.Div()
        
        try:
            # Parse data and analyze noise
            df = parse_contents(contents, filename)
            
            from utils.enhanced_preprocessing import EnhancedNoiseAnalyzer
            from plots import create_noise_analysis_plot
            
            # Extract data arrays
            frequency = df.iloc[:, 0].values
            dk = df.iloc[:, 1].values
            df_loss = df.iloc[:, 2].values
            
            # Analyze noise
            analyzer = EnhancedNoiseAnalyzer()
            noise_metrics = analyzer.comprehensive_analysis(dk, df_loss, frequency)
            
            # Create noise analysis plot
            fig = create_noise_analysis_plot(noise_metrics, frequency, dk, df_loss)
            
            return html.Div([
                dcc.Graph(figure=fig, style={"height": "400px"})
            ])
            
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                    html.Span(f"Noise analysis failed: {str(e)}", className="text-warning")
                ], className="alert alert-warning")
            ])

    # Callback to convert switch states to model selection list
    @app.callback(
        Output("model-selection-internal", "data"),
        [
            Input("switch-debye", "value"),
            Input("switch-multipole", "value"),
            Input("switch-cole-cole", "value"),
            Input("switch-cole-davidson", "value"),
            Input("switch-havriliak", "value"),
            Input("switch-lorentz", "value"),
            Input("switch-sarkar", "value"),
            Input("switch-hybrid", "value"),
            Input("switch-kk", "value"),
        ]
    )
    def convert_switches_to_selection(debye, multipole, cole_cole, cole_davidson, havriliak, lorentz, sarkar, hybrid, kk):
        selected_models = []
        if debye: selected_models.append("debye")
        if multipole: selected_models.append("multipole_debye")
        if cole_cole: selected_models.append("cole_cole")
        if cole_davidson: selected_models.append("cole_davidson")
        if havriliak: selected_models.append("havriliak_negami")
        if lorentz: selected_models.append("lorentz")
        if sarkar: selected_models.append("sarkar")
        if hybrid: selected_models.append("hybrid")
        if kk: selected_models.append("kk")
        return selected_models


    @app.callback(
        [
            Output("preprocessing-summary", "children"),
            Output("results-summary", "children"),
            Output("permittivity-plots", "figure"),
            Output("download-csv-btn", "disabled"),
        ],
        [
            Input("upload-data", "contents"),
            Input("analysis-mode", "value"),
            Input("selection-method", "value"),
            Input("model-selection-internal", "data"),
            Input("n-terms-slider", "value"),  # Hybrid model terms
            Input("multipole-terms-slider", "value"),  # Multipole Debye terms
            Input("lorentz-terms-slider", "value"),  # Lorentz oscillators
            Input("preprocessing-mode", "value"),  # Preprocessing mode
            Input("preprocessing-selection-method", "value"),  # Auto preprocessing method
            Input("smoothing-algorithm", "value"),  # Manual preprocessing algorithm
            Input("spline-s-mode", "value"),  # Spline s parameter mode
            Input("spline-s-value", "value"),  # Manual spline s value
            Input("lowess-frac", "value"),  # LOWESS fraction
            Input("savgol-window", "value"),  # Savitzky-Golay window
            Input("savgol-polyorder", "value"),  # Savitzky-Golay polynomial order
            Input("gaussian-sigma", "value"),  # Gaussian sigma
            Input("median-window", "value"),  # Median window size
        ],
        [State("upload-data", "filename")],
    )
    def update_output(contents, analysis_mode, selection_method, selected_models, hybrid_terms, multipole_terms, lorentz_terms, preprocessing_mode, preprocessing_selection_method, smoothing_algorithm, spline_s_mode, spline_s_value, lowess_frac, savgol_window, savgol_polyorder, gaussian_sigma, median_window, filename):
        print(f"=== CALLBACK DEBUG ===")
        print(f"Analysis mode: {analysis_mode}")
        print(f"Selection method: {selection_method}")
        print(f"Selected models: {selected_models}")
        print(f"Hybrid terms: {hybrid_terms}")
        print(f"Multipole terms: {multipole_terms}")
        print(f"Lorentz terms: {lorentz_terms}")
        print(f"Filename: {filename}")

        if not contents:
            return html.Div(), html.Div(), {}, True

        print("Parsing contents...")
        df = parse_contents(contents, filename)
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame head:\n{df.head()}")

        print("Running analysis...")
        # Create a dictionary of model parameters including preprocessing
        model_params = {
            'hybrid_terms': hybrid_terms,
            'multipole_terms': multipole_terms,
            'lorentz_terms': lorentz_terms,
            'selection_method': selection_method,
            'preprocessing_mode': preprocessing_mode,
            'preprocessing_method': smoothing_algorithm,
            'preprocessing_selection_method': preprocessing_selection_method,
            # Algorithm-specific parameters for manual mode
            'preprocessing_params': {
                'spline_s_mode': spline_s_mode,
                'spline_s_value': spline_s_value,
                'lowess_frac': lowess_frac,
                'savgol_window': savgol_window,
                'savgol_polyorder': savgol_polyorder,
                'gaussian_sigma': gaussian_sigma,
                'median_window': median_window
            }
        }

        # Run analysis with appropriate mode
        print(f"Starting {analysis_mode} mode analysis...")
        results = run_analysis(df, selected_models or [], model_params, analysis_mode)
        print(f"Analysis complete. Results type: {type(results)}")

        # Extract preprocessing info if available
        preprocessing_info = results.get('preprocessing_info', {})
        preprocessing_summary = create_preprocessing_summary(preprocessing_info)
        
        # Handle different result types based on analysis mode
        if analysis_mode == "auto":
            # Auto mode returns best model selection result
            if "error" in results:
                summary = html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        html.Strong(f"Error: {results['error']}")
                    ], className="alert alert-danger")
                ])
                return preprocessing_summary, summary, {}, True
            
            # Extract auto-selection results
            best_model_name = results["best_model_name"]
            best_model_result = results["best_model_result"]
            rationale = results["selection_rationale"]
            
            # Create summary for auto mode
            summary_items = [
                html.Div([
                    html.I(className="fas fa-trophy me-2 text-warning"),
                    html.Strong("Auto-Selected Best Model", className="text-primary")
                ], className="mb-2"),
                html.Div([
                    html.Strong(f"🏆 {best_model_name.replace('_', ' ').title()}", className="fs-5 text-success"),
                    html.Span("BEST FIT", className="badge bg-success ms-2")
                ], className="mb-2"),
                html.Div([
                    html.Small(f"💡 {rationale}", className="text-muted")
                ], className="mb-2"),
                html.Div([
                    html.Small(f"📊 AIC: {best_model_result.get('aic', 'N/A'):.1f} | "
                              f"🎯 RMSE: {best_model_result.get('rmse', 'N/A'):.4f} | "
                              f"✅ Success: {best_model_result.get('success', False)}", 
                              className="text-info")
                ])
            ]
            
            # Add optimal N information if available
            if 'optimal_n' in best_model_result:
                summary_items.append(html.Div([
                    html.Small(f"🔧 Optimal N: {best_model_result['optimal_n']} "
                              f"(tested N={best_model_result['optimization_summary']['tested_n_values']})", 
                              className="text-success")
                ], className="mb-1"))
            
            # Prepare results for plotting (convert to compatible format)
            plot_results = {best_model_name: best_model_result}
            
        elif analysis_mode == "auto_compare":
            # Auto-compare mode returns detailed comparison
            if "error" in results:
                summary = html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        html.Strong(f"Error: {results['error']}")
                    ], className="alert alert-danger")
                ])
                return html.Div(), summary, {}, True
            
            enhanced_comparison = results["enhanced_comparison"]
            valid_results = results["valid_results"]
            
            # Create enhanced summary
            summary_items = [
                html.Div([
                    html.I(className="fas fa-chart-bar me-2 text-primary"),
                    html.Strong("Enhanced Model Comparison", className="text-primary")
                ], className="mb-2"),
                html.Div([
                    html.Strong(f"🏆 Best: {enhanced_comparison['best_model'].replace('_', ' ').title()}", className="fs-6 text-success"),
                    html.Span("OPTIMAL", className="badge bg-success ms-2")
                ], className="mb-2"),
                html.Div([
                    html.Small(f"💡 {enhanced_comparison['selection_rationale']}", className="text-muted")
                ], className="mb-2")
            ]
            
            # Add top 3 models summary
            comparison_data = enhanced_comparison.get("comparison_data", [])[:3]
            for i, model_data in enumerate(comparison_data):
                rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                badge_class = "bg-success" if i == 0 else "bg-info" if i == 1 else "bg-secondary"
                
                model_name = model_data['model']
                model_result = valid_results.get(model_name, {})
                
                # Build display text
                display_text = f"{rank_icon} {model_name.replace('_', ' ').title()}: AIC={model_data['aic']:.1f}, RMSE={model_data['rmse']:.4f}"
                
                # Add optimal N if available
                if 'optimal_n' in model_result:
                    display_text += f" (N={model_result['optimal_n']})"
                
                summary_items.append(html.Div([
                    html.Small(display_text),
                    html.Span(f"#{i+1}", className=f"badge {badge_class} ms-2")
                ], className="mb-1"))
            
            # Use all valid results for plotting
            plot_results = valid_results
            
        else:  # manual mode
            # Manual mode returns results with preprocessing info
            print("Manual mode - processing results with preprocessing info")
            
            # Extract model results from the new structure
            model_results = results.get('model_results', results)
            
            # Determine best fit (lowest RMSE) for manual mode
            best_model = None
            best_rmse = float("inf")
            for key, r in model_results.items():
                if r is not None and "rmse" in r and r["rmse"] < best_rmse:
                    best_rmse = r["rmse"]
                    best_model = key

            summary_items = []
            for key, r in model_results.items():
                if r is None:
                    continue
                    
                label = key.replace("_", " ").title()
                if "rmse" in r:
                    label += f": RMSE={r['rmse']:.4f}"

                if key == best_model:
                    badge_class = "badge bg-success ms-2"
                    badge_text = "BEST FIT"
                elif "rmse" in r:
                    badge_class = "badge bg-secondary ms-2"
                    badge_text = "PASS"
                else:
                    badge_class = "badge bg-warning ms-2"
                    badge_text = "N/A"

                badge = html.Span(badge_text, className=badge_class)
                summary_items.append(html.Div([html.Strong(label), badge]))
            
            plot_results = model_results

        # Create summary
        summary = html.Div(summary_items, className="p-2")
        
        print("Creating plot...")
        # Determine best model for highlighting
        plot_best_model = None
        if analysis_mode == "auto_compare":
            plot_best_model = enhanced_comparison.get('best_model')
        elif analysis_mode == "auto":
            plot_best_model = best_model_name
        else:  # manual mode
            plot_best_model = best_model
            
        fig = create_permittivity_plot(plot_results, df, best_model=plot_best_model)
        print("Plot created successfully")

        # Determine if download button should be enabled (only if exactly one model)
        valid_models = [k for k, v in plot_results.items() if v is not None and k != "kk"]
        download_disabled = len(valid_models) != 1

        print("=== CALLBACK COMPLETE ===")
        return preprocessing_summary, summary, fig, download_disabled

    # Store data for download
    @app.callback(
        Output("download-csv", "data"),
        [Input("download-csv-btn", "n_clicks")],
        [
            State("upload-data", "contents"),
            State("upload-data", "filename"),
            State("analysis-mode", "value"),
            State("selection-method", "value"),
            State("model-selection-internal", "data"),
            State("n-terms-slider", "value"),
            State("multipole-terms-slider", "value"),
            State("lorentz-terms-slider", "value"),
        ],
        prevent_initial_call=True
    )
    def download_csv(n_clicks, contents, filename, analysis_mode, selection_method, 
                     selected_models, hybrid_terms, multipole_terms, lorentz_terms):
        if not n_clicks or not contents:
            return None
            
        # Parse data and run analysis again to get current results
        df = parse_contents(contents, filename)
        model_params = {
            'hybrid_terms': hybrid_terms,
            'multipole_terms': multipole_terms,
            'lorentz_terms': lorentz_terms,
            'selection_method': selection_method
        }
        results = run_analysis(df, selected_models or [], model_params, analysis_mode)
        
        # Determine the single model to export
        single_model_name = None
        single_model_result = None
        
        if analysis_mode == "auto":
            if "error" not in results:
                single_model_name = results["best_model_name"]
                single_model_result = results["best_model_result"]
        elif analysis_mode == "auto_compare":
            if "error" not in results:
                valid_results = results["valid_results"]
                valid_models = [k for k, v in valid_results.items() if v is not None and k != "kk"]
                if len(valid_models) == 1:
                    single_model_name = valid_models[0]
                    single_model_result = valid_results[single_model_name]
        else:  # manual mode
            model_results = results.get('model_results', results)
            valid_models = [k for k, v in model_results.items() if v is not None and k != "kk"]
            if len(valid_models) == 1:
                single_model_name = valid_models[0]
                single_model_result = model_results[single_model_name]
        
        if not single_model_name or not single_model_result:
            return None
            
        # Extract data for CSV
        freq_vals = None
        dk_fit = None
        df_fit = None
        
        # Get frequency values
        if "freq" in single_model_result:
            freq_vals = single_model_result["freq"]
        elif "freq_ghz" in single_model_result:
            freq_vals = single_model_result["freq_ghz"]
        else:
            return None
            
        # Get fitted values
        if "dk_fit" in single_model_result and "df_fit" in single_model_result:
            dk_fit = single_model_result["dk_fit"]
            df_fit = single_model_result["df_fit"]
        elif "eps_fit" in single_model_result:
            eps_fit = single_model_result["eps_fit"]
            if hasattr(eps_fit, 'real') and hasattr(eps_fit, 'imag'):
                dk_fit = eps_fit.real
                df_fit = eps_fit.imag
            else:
                dk_fit = eps_fit
                df_fit = [0] * len(dk_fit)
        else:
            return None
            
        # Create DataFrame for export
        export_df = pd.DataFrame({
            'Frequency_GHz': freq_vals,
            'Dk': dk_fit,
            'Df': df_fit
        })
        
        # Generate filename
        base_name = filename.rsplit('.', 1)[0] if filename else "data"
        model_name = single_model_name.replace('_', '-')
        export_filename = f"{base_name}_{model_name}.csv"
        
        return dcc.send_data_frame(export_df.to_csv, export_filename, index=False)

