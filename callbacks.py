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
            Input("n-terms-slider", "value"),
        ],
        [State("upload-data", "filename")],
    )
    def update_output(contents, selected_models, n_terms, filename):
        if not contents:
            return html.Div(), {}, html.Div()

        df = parse_contents(contents, filename)
        results = run_analysis(df, selected_models, n_terms)
        fig = create_permittivity_plot(results, df)

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
        return summary, fig, html.Div("Download section")
