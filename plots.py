import plotly.graph_objects as go

def create_permittivity_plot(results, df):
    fig = go.Figure()

    freq = df.iloc[:, 0].values
    Dk = df.iloc[:, 1].values
    fig.add_trace(go.Scatter(x=freq, y=Dk, mode="markers+lines", name="Measured Dk"))

    color_map = {
        "debye": "red",
        "multipole_debye": "orange",
        "cole_cole": "green",
        "cole_davidson": "blue",
        "havriliak_negami": "purple",
        "lorentz": "brown",
        "sarkar": "pink",
        "hybrid": "black",
        "kk": "gray",
    }

    for key, result in results.items():
        if key in color_map:
            if "freq" in result:
                x_vals = result["freq"]
            elif "freq_ghz" in result:
                x_vals = result["freq_ghz"]
            else:
                continue

            if "eps_fit" in result:
                y_vals = result["eps_fit"].real
            elif "eps_real_kk_full" in result:
                y_vals = result["eps_real_kk_full"]
            else:
                continue

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=key.replace("_", " ").title(),
                line=dict(color=color_map[key])
            ))

    fig.update_layout(
        title="Permittivity Analysis",
        xaxis_title="Frequency (GHz)",
        yaxis_title="Dk"
    )
    return fig
