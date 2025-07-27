import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_permittivity_plot(results, df):
    freq = df.iloc[:, 0].values
    dk_exp = df.iloc[:, 1].values
    df_exp = df.iloc[:, 2].values

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Dielectric Constant (Dk)", "Dissipation Factor (Df)")
    )

    # Experimental Data
    fig.add_trace(go.Scatter(x=freq, y=dk_exp, mode="markers+lines", name="Measured Dk"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq, y=df_exp, mode="markers+lines", name="Measured Df"), row=2, col=1)

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
        if key not in color_map:
            continue

        # Get frequency values
        if "freq" in result:
            x_vals = result["freq"]
        elif "freq_ghz" in result:
            x_vals = result["freq_ghz"]
        else:
            continue

        # Get fitted values
        if "eps_fit" in result:
            dk_fit = result["eps_fit"].real
            df_fit = result["eps_fit"].imag
        elif "eps_real_kk_full" in result:
            dk_fit = result["eps_real_kk_full"]
            df_fit = None
        else:
            continue

        # Plot Dk
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=dk_fit,
            mode="lines",
            name=f"{key.replace('_', ' ').title()} Dk",
            line=dict(color=color_map[key])
        ), row=1, col=1)

        # Plot Df (if available)
        if df_fit is not None:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=df_fit,
                mode="lines",
                name=f"{key.replace('_', ' ').title()} Df",
                line=dict(color=color_map[key], dash="dot")
            ), row=2, col=1)

    fig.update_layout(
        title="Permittivity Analysis",
        xaxis_title="Frequency (GHz)",
        yaxis_title="Dk",
        height=800
    )

    return fig
