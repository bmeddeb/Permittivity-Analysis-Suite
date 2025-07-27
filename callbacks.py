# callbacks.py - Updated for multiple sliders
from dash import Input, Output, State, html
from utils.helpers import parse_contents
from analysis import run_analysis
from plots import create_permittivity_plot


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

    @app.callback(
        [
            Output("results-summary", "children"),
            Output("permittivity-plots", "figure"),
            Output("download-section", "children"),
        ],
        [
            Input("upload-data", "contents"),
            Input("analysis-mode", "value"),
            Input("selection-method", "value"),
            Input("model-selection", "value"),
            Input("n-terms-slider", "value"),  # Hybrid model terms
            Input("multipole-terms-slider", "value"),  # Multipole Debye terms
            Input("lorentz-terms-slider", "value"),  # Lorentz oscillators
        ],
        [State("upload-data", "filename")],
    )
    def update_output(contents, analysis_mode, selection_method, selected_models, hybrid_terms, multipole_terms, lorentz_terms, filename):
        print(f"=== CALLBACK DEBUG ===")
        print(f"Analysis mode: {analysis_mode}")
        print(f"Selection method: {selection_method}")
        print(f"Selected models: {selected_models}")
        print(f"Hybrid terms: {hybrid_terms}")
        print(f"Multipole terms: {multipole_terms}")
        print(f"Lorentz terms: {lorentz_terms}")
        print(f"Filename: {filename}")

        if not contents:
            return html.Div(), {}, html.Div()

        print("Parsing contents...")
        df = parse_contents(contents, filename)
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame head:\n{df.head()}")

        print("Running analysis...")
        # Create a dictionary of model parameters
        model_params = {
            'hybrid_terms': hybrid_terms,
            'multipole_terms': multipole_terms,
            'lorentz_terms': lorentz_terms,
            'selection_method': selection_method
        }

        # Run analysis with appropriate mode
        results = run_analysis(df, selected_models or [], model_params, analysis_mode)
        print(f"Analysis complete. Results type: {type(results)}")

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
                return summary, {}, html.Div()
            
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
                return summary, {}, html.Div()
            
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
            # Manual mode returns traditional results dict
            print("Manual mode - processing traditional results")
            
            # Determine best fit (lowest RMSE) for manual mode
            best_model = None
            best_rmse = float("inf")
            for key, r in results.items():
                if r is not None and "rmse" in r and r["rmse"] < best_rmse:
                    best_rmse = r["rmse"]
                    best_model = key

            summary_items = []
            for key, r in results.items():
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
            
            plot_results = results

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

        print("=== CALLBACK COMPLETE ===")
        return summary, fig, html.Div("Download section")