from dash import Input, Output, State, html
from utils import parse_contents
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

        summary_items = []

        if "debye" in results:
            r = results["debye"]
            rmse_real = ((r["eps_fit"].real - r["dk_exp"]) ** 2).mean() ** 0.5
            pass_fail = rmse_real < 0.1
            badge = html.Span("PASS" if pass_fail else "FAIL",
                              className=f"badge {'bg-success' if pass_fail else 'bg-danger'} ms-2")
            summary_items.append(html.Div([html.Strong("Debye Model:"), badge]))

        if "hybrid" in results:
            r = results["hybrid"]
            rmse_real = ((r["dk_fit"] - r["dk_exp"]) ** 2).mean() ** 0.5
            pass_fail = rmse_real < 0.1 and r["success"]
            badge = html.Span("PASS" if pass_fail else "FAIL",
                              className=f"badge {'bg-success' if pass_fail else 'bg-danger'} ms-2")
            summary_items.append(html.Div([html.Strong("Hybrid Model:"), badge]))

        if "kk" in results:
            r = results["kk"]
            pass_fail = r["mean_err_full"] < 0.05
            badge = html.Span("PASS" if pass_fail else "FAIL",
                              className=f"badge {'bg-success' if pass_fail else 'bg-danger'} ms-2")
            summary_items.append(html.Div([html.Strong("KK Causality Check:"), badge]))

        summary = html.Div(summary_items, className="p-2")
        return summary, fig, html.Div("Download section")
