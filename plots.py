# plots.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_permittivity_plot(results, df):
    freq = df.iloc[:, 0].values
    dk_exp = df.iloc[:, 1].values
    df_exp = df.iloc[:, 2].values

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("Dielectric Constant (Dk)", "Dissipation Factor (Df)")
    )

    # Experimental Data
    fig.add_trace(go.Scatter(
        x=freq,
        y=dk_exp,
        mode="markers+lines",
        name="Measured Dk",
        marker=dict(color='blue', size=6),
        line=dict(color='blue', width=3)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=freq,
        y=df_exp,
        mode="markers+lines",
        name="Measured Df",
        marker=dict(color='red', size=6),
        line=dict(color='red', width=3)
    ), row=2, col=1)

    color_map = {
        "debye": "darkred",
        "multipole_debye": "orange",
        "cole_cole": "darkgreen",
        "cole_davidson": "darkblue",
        "havriliak_negami": "purple",
        "lorentz": "brown",
        "sarkar": "hotpink",
        "hybrid": "black",
    }

    for key, result in results.items():
        if key not in color_map:
            continue

        # Get frequency values safely
        x_vals = None
        if "freq" in result:
            x_vals = result["freq"]
        elif "freq_ghz" in result:
            x_vals = result["freq_ghz"]
        else:
            continue

        # Skip KK causality check for plotting - it's a validation tool, not a fitting model
        # The KK results are shown in the summary (PASS/FAIL) which is more meaningful
        if key == "kk":
            continue

        # Normal model handling for all other models
        dk_fit = None
        df_fit = None

        # Priority: use separate dk_fit/df_fit if available
        if "dk_fit" in result and "df_fit" in result:
            dk_fit = result["dk_fit"]
            df_fit = result["df_fit"]
        elif "eps_fit" in result:
            eps_fit = result["eps_fit"]
            if np.iscomplexobj(eps_fit):
                dk_fit = eps_fit.real
                df_fit = eps_fit.imag
            else:
                dk_fit = eps_fit  # Real values only
        elif "eps_real_kk_full" in result:
            dk_fit = result["eps_real_kk_full"]
            if "eps_imag_kk_full" in result:
                df_fit = result["eps_imag_kk_full"]

        # Plot Dk (always plot this for fitting models)
        if dk_fit is not None and len(x_vals) == len(dk_fit):
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=dk_fit,
                mode="lines",
                name=f"{key.replace('_', ' ').title()} Dk",
                line=dict(color=color_map[key], width=2),
                showlegend=True
            ), row=1, col=1)

        # Plot Df ONLY if it has meaningful values (not all zeros)
        if (df_fit is not None and len(x_vals) == len(df_fit) and
                np.max(np.abs(df_fit)) > 1e-10):  # Only plot if max value > threshold

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=df_fit,
                mode="lines",
                name=f"{key.replace('_', ' ').title()} Df",
                line=dict(color=color_map[key], dash="dot", width=2),
                showlegend=True
            ), row=2, col=1)

    # Update layout
    fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=1)
    fig.update_yaxes(title_text="Dk", row=1, col=1)
    fig.update_yaxes(title_text="Df", row=2, col=1)

    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    fig.update_layout(
        title={
            'text': "Permittivity Analysis",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=850,
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Gray",
            borderwidth=1
        ),
        hovermode='closest'
    )

    return fig