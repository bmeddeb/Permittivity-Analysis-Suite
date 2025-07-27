# callbacks.py - Updated for multiple sliders
from dash import Input, Output, State, html
from utils.helpers import parse_contents
from analysis import run_analysis
from plots import create_permittivity_plot


def register_callbacks(app):
    @app.callback(
        [
            Output("results-summary", "children"),
            Output("permittivity-plots", "figure"),
            Output("download-section", "children"),
        ],
        [
            Input("upload-data", "contents"),
            Input("model-selection", "value"),
            Input("n-terms-slider", "value"),  # Hybrid model terms
            Input("multipole-terms-slider", "value"),  # Multipole Debye terms
            Input("lorentz-terms-slider", "value"),  # Lorentz oscillators
        ],
        [State("upload-data", "filename")],
    )
    def update_output(contents, selected_models, hybrid_terms, multipole_terms, lorentz_terms, filename):
        print(f"=== CALLBACK DEBUG ===")
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
            'lorentz_terms': lorentz_terms
        }

        results = run_analysis(df, selected_models, model_params)
        print(f"Analysis complete. Results keys: {list(results.keys())}")

        # Debug each result
        for key, result in results.items():
            print(f"\nResult '{key}' structure:")
            for rkey, rval in result.items():
                if hasattr(rval, '__len__') and not isinstance(rval, str):
                    print(f"  {rkey}: length={len(rval)}, type={type(rval)}")
                else:
                    print(f"  {rkey}: {rval}")

        print("Creating plot...")
        fig = create_permittivity_plot(results, df)
        print("Plot created successfully")

        # Determine best fit (lowest RMSE)
        best_model = None
        best_rmse = float("inf")
        for key, r in results.items():
            if "rmse" in r and r["rmse"] < best_rmse:
                best_rmse = r["rmse"]
                best_model = key

        summary_items = []
        for key, r in results.items():
            label = key.replace("_", " ").title() + ":"
            if "rmse" in r:
                label += f" RMSE={r['rmse']:.4f}"

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

        summary = html.Div(summary_items, className="p-2")
        print("=== CALLBACK COMPLETE ===")
        return summary, fig, html.Div("Download section")