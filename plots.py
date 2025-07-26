import plotly.graph_objects as go

def create_permittivity_plot(results, df):
    fig = go.Figure()

    # Always plot measured data
    freq = df.iloc[:,0].values
    Dk = df.iloc[:,1].values
    fig.add_trace(go.Scatter(x=freq, y=Dk, mode='markers+lines', name='Measured Dk'))

    # Debye model
    if "debye" in results:
        r = results["debye"]
        fig.add_trace(go.Scatter(x=r['freq_ghz'], y=r['eps_fit'].real,
                                 mode='lines', name='Debye Fit'))

    # KK Causality Check
    if "kk" in results:
        r = results["kk"]
        fig.add_trace(go.Scatter(x=r['freq_ghz'], y=r['eps_real_kk_full'],
                                 mode='lines', name='KK Real (Full)', line=dict(dash="dash")))

    fig.update_layout(title="Permittivity Analysis", xaxis_title="Frequency (GHz)", yaxis_title="Dk")
    return fig
