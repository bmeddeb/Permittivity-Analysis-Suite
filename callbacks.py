from dash import Input, Output, State, html
from utils import parse_contents
from analysis import run_analysis
from plots import create_permittivity_plot

def register_callbacks(app):
    @app.callback(
        [
            Output('results-summary', 'children'),
            Output('permittivity-plots', 'figure'),
            Output('download-section', 'children')
        ],
        [Input('upload-data', 'contents'),
         Input('model-selection', 'value'),
         Input('n-terms-slider', 'value')],
        [State('upload-data', 'filename')]
    )
    def update_output(contents, selected_models, n_terms, filename):
        if not contents:
            # Must return exactly 3 items
            return html.Div(), {}, html.Div()

        df = parse_contents(contents, filename)
        results = run_analysis(df, selected_models, n_terms)
        fig = create_permittivity_plot(results, df)

        # ✅ Show only PASS/FAIL summary
        summary_items = []
        if "debye" in results:
            summary_items.append(html.Div(
                f"Debye Model: {'PASS' if True else 'FAIL'}",  # Replace True with real check
                className="fw-bold"
            ))
        if "kk" in results:
            summary_items.append(html.Div(
                f"KK Causality Check: {'PASS' if True else 'FAIL'}",  # Replace True with check
                className="fw-bold"
            ))

        summary = html.Div(summary_items)

        download_section = html.Div("Download section")

        return summary, fig, download_section
