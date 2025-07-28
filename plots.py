# plots.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def create_permittivity_plot(results, df, best_model=None):
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

        # Determine if this is the best model for highlighting
        is_best_model = (best_model is not None and key == best_model)
        line_width = 4 if is_best_model else 2
        name_prefix = "🏆 " if is_best_model else ""

        # Plot Dk (always plot this for fitting models)
        if dk_fit is not None and len(x_vals) == len(dk_fit):
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=dk_fit,
                mode="lines",
                name=f"{name_prefix}{key.replace('_', ' ').title()} Dk",
                line=dict(color=color_map[key], width=line_width),
                showlegend=True
            ), row=1, col=1)

        # Plot Df ONLY if it has meaningful values (not all zeros)
        if (df_fit is not None and len(x_vals) == len(df_fit) and
                np.max(np.abs(df_fit)) > 1e-10):  # Only plot if max value > threshold

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=df_fit,
                mode="lines",
                name=f"{name_prefix}{key.replace('_', ' ').title()} Df",
                line=dict(color=color_map[key], dash="dot", width=line_width),
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
        height=700,
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


def create_preprocessing_comparison_plot(original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                                       processing_info: dict) -> go.Figure:
    """
    Create before/after comparison plot for preprocessing visualization
    
    Args:
        original_df: Original DataFrame with [Frequency_GHz, Dk, Df]
        processed_df: Preprocessed DataFrame with same structure
        processing_info: Dictionary with preprocessing information
        
    Returns:
        Plotly figure showing before/after comparison
    """
    freq_orig = original_df.iloc[:, 0].values
    dk_orig = original_df.iloc[:, 1].values
    df_orig = original_df.iloc[:, 2].values
    
    freq_proc = processed_df.iloc[:, 0].values
    dk_proc = processed_df.iloc[:, 1].values
    df_proc = processed_df.iloc[:, 2].values
    
    # Create subplot with before/after comparison
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Original Dk Data", "Preprocessed Dk Data",
            "Original Df Data", "Preprocessed Df Data"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Color scheme for consistency
    original_color = "#1f77b4"  # Blue
    processed_color = "#ff7f0e"  # Orange
    
    # Plot original Dk
    fig.add_trace(go.Scatter(
        x=freq_orig,
        y=dk_orig,
        mode="markers+lines",
        name="Original Dk",
        marker=dict(color=original_color, size=4, opacity=0.7),
        line=dict(color=original_color, width=2),
        showlegend=True
    ), row=1, col=1)
    
    # Plot processed Dk
    fig.add_trace(go.Scatter(
        x=freq_proc,
        y=dk_proc,
        mode="lines",
        name="Smoothed Dk",
        line=dict(color=processed_color, width=3),
        showlegend=True
    ), row=1, col=2)
    
    # Plot original Df
    fig.add_trace(go.Scatter(
        x=freq_orig,
        y=df_orig,
        mode="markers+lines",
        name="Original Df",
        marker=dict(color=original_color, size=4, opacity=0.7),
        line=dict(color=original_color, width=2),
        showlegend=False
    ), row=2, col=1)
    
    # Plot processed Df
    fig.add_trace(go.Scatter(
        x=freq_proc,
        y=df_proc,
        mode="lines",
        name="Smoothed Df",
        line=dict(color=processed_color, width=3),
        showlegend=False
    ), row=2, col=2)
    
    # Add overlay comparison if data is smoothed
    if processing_info.get('smoothing_applied', False):
        # Add faint original data on processed plots for comparison
        fig.add_trace(go.Scatter(
            x=freq_orig,
            y=dk_orig,
            mode="markers",
            name="Original (reference)",
            marker=dict(color="gray", size=2, opacity=0.3),
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=True
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=freq_orig,
            y=df_orig,
            mode="markers",
            name="Original (reference)",
            marker=dict(color="gray", size=2, opacity=0.3),
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False
        ), row=2, col=2)
    
    # Calculate improvement metrics for display
    noise_metrics = processing_info.get('noise_metrics', {})
    noise_score = noise_metrics.get('overall_noise_score', 0)
    algorithm = processing_info.get('dk_algorithm', 'unknown')
    
    # Create title with processing information
    title_text = f"Preprocessing Comparison"
    if processing_info.get('smoothing_applied', False):
        title_text += f" - {algorithm.replace('_', ' ').title()} Applied"
        title_text += f" (Noise Score: {noise_score:.3f})"
    else:
        title_text += " - No Smoothing Applied"
    
    # Update layout
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.15,
            yanchor="top"
        ),
        hovermode='closest'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=2)
    fig.update_yaxes(title_text="Dk", row=1, col=1)
    fig.update_yaxes(title_text="Dk", row=1, col=2)
    fig.update_yaxes(title_text="Df", row=2, col=1)
    fig.update_yaxes(title_text="Df", row=2, col=2)
    
    return fig


def create_noise_analysis_plot(noise_metrics: dict, frequency: np.ndarray, 
                             dk: np.ndarray, df: np.ndarray) -> go.Figure:
    """
    Create detailed noise analysis visualization
    
    Args:
        noise_metrics: Dictionary with noise analysis results
        frequency: Frequency array
        dk: Dk data array  
        df: Df data array
        
    Returns:
        Plotly figure showing noise analysis details
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Spectral Analysis (Dk)", "Spectral Analysis (Df)",
            "Derivative Analysis (Dk)", "Derivative Analysis (Df)"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Spectral analysis using FFT
    def analyze_spectrum(data):
        """Simple spectral analysis"""
        n = len(data)
        if n < 4:
            return np.array([]), np.array([])
        
        # Remove DC and apply window
        data_centered = data - np.mean(data)
        window = np.hanning(n)
        data_windowed = data_centered * window
        
        # FFT
        spectrum = np.fft.rfft(data_windowed)
        freqs = np.fft.rfftfreq(n, d=1.0)  # Normalized frequency
        power = np.abs(spectrum)**2
        
        return freqs[1:], power[1:]  # Skip DC component
    
    # Analyze spectra
    dk_freqs, dk_power = analyze_spectrum(dk)
    df_freqs, df_power = analyze_spectrum(df)
    
    if len(dk_freqs) > 0:
        fig.add_trace(go.Scatter(
            x=dk_freqs,
            y=10*np.log10(dk_power + 1e-10),  # dB scale
            mode="lines",
            name="Dk Power Spectrum",
            line=dict(color="blue", width=2)
        ), row=1, col=1)
    
    if len(df_freqs) > 0:
        fig.add_trace(go.Scatter(
            x=df_freqs,
            y=10*np.log10(df_power + 1e-10),  # dB scale
            mode="lines",
            name="Df Power Spectrum",
            line=dict(color="red", width=2)
        ), row=1, col=2)
    
    # Derivative analysis
    if len(dk) > 1:
        dk_diff = np.diff(dk)
        fig.add_trace(go.Scatter(
            x=frequency[1:],
            y=dk_diff,
            mode="lines+markers",
            name="Dk Derivative",
            line=dict(color="darkblue", width=1),
            marker=dict(size=3)
        ), row=2, col=1)
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     row=2, col=1, opacity=0.5)
    
    if len(df) > 1:
        df_diff = np.diff(df)
        fig.add_trace(go.Scatter(
            x=frequency[1:],
            y=df_diff,
            mode="lines+markers",
            name="Df Derivative",
            line=dict(color="darkred", width=1),
            marker=dict(size=3)
        ), row=2, col=2)
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     row=2, col=2, opacity=0.5)
    
    # Add metrics as annotations
    metrics_text = []
    if 'dk_spectral_snr' in noise_metrics:
        metrics_text.append(f"Dk SNR: {noise_metrics['dk_spectral_snr']:.1f} dB")
    if 'df_spectral_snr' in noise_metrics:
        metrics_text.append(f"Df SNR: {noise_metrics['df_spectral_snr']:.1f} dB")
    if 'dk_roughness' in noise_metrics:
        metrics_text.append(f"Dk Roughness: {noise_metrics['dk_roughness']:.3f}")
    if 'df_roughness' in noise_metrics:
        metrics_text.append(f"Df Roughness: {noise_metrics['df_roughness']:.3f}")
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Noise Analysis (Overall Score: {noise_metrics.get('overall_noise_score', 0):.3f})",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=400,
        showlegend=False,
        annotations=[
            dict(
                text="<br>".join(metrics_text),
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10)
            )
        ]
    )
    
    # Update axes
    fig.update_xaxes(title_text="Normalized Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Normalized Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=2)
    fig.update_yaxes(title_text="Power (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Power (dB)", row=1, col=2)
    fig.update_yaxes(title_text="dDk/dF", row=2, col=1)
    fig.update_yaxes(title_text="dDf/dF", row=2, col=2)
    
    return fig