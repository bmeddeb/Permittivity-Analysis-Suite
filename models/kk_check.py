# models/kk_check.py
import numpy as np
from utils.helpers import get_numeric_data


def kk_real_from_imag(freq, eps_imag, eps_inf):
    """
    Calculate real permittivity from imaginary part using Kramers-Kronig relations

    Args:
        freq: frequency array (Hz)
        eps_imag: imaginary permittivity
        eps_inf: high-frequency limit of real permittivity

    Returns:
        eps_real_kk: real permittivity calculated from KK relations
    """
    N = len(freq)
    eps_real_kk = np.zeros(N)
    omega_all = 2 * np.pi * freq

    # Use more robust gradient calculation
    d_omega = np.gradient(omega_all)

    for i in range(N):
        omega = omega_all[i]

        # Kramers-Kronig relation: avoid singularity at omega = omega_i
        # Use small regularization parameter
        regularization = 1e-12 * np.max(omega_all)
        denominator = omega_all ** 2 - omega ** 2 + regularization

        # Principal value integral
        integrand = (omega_all * eps_imag) / denominator

        # Set integrand to zero at the singular point
        integrand[i] = 0

        # Calculate principal value using trapezoid rule
        principal_value = (2 / np.pi) * np.trapz(integrand, omega_all)

        eps_real_kk[i] = eps_inf + principal_value

    return eps_real_kk


def kk_causality_check(df):
    """
    Perform Kramers-Kronig causality check on experimental data

    Args:
        df: DataFrame with experimental data [Frequency_GHz, Dk, Df]

    Returns:
        Dictionary with KK analysis results
    """
    freq_ghz, dk_exp, df_exp = get_numeric_data(df)
    freq = freq_ghz * 1e9  # Convert to Hz

    print("Kramers-Kronig Causality Check")
    print(f"Frequency range: {np.min(freq_ghz):.1f} - {np.max(freq_ghz):.1f} GHz")
    print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
    print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

    # Real and imaginary permittivity from experimental data
    eps_real_meas = dk_exp

    # CORRECTED: Use df_exp directly, not dk*df_loss
    # The input Df should already be the imaginary part of permittivity
    eps_imag_meas = df_exp

    # Estimate eps_inf from high-frequency behavior
    num_tail = max(3, round(0.1 * len(freq)))
    eps_inf = np.mean(eps_real_meas[-num_tail:])

    print(f"Estimated eps_inf from high-frequency tail: {eps_inf:.3f}")

    # Full frequency range KK analysis
    print("Performing full-range KK analysis...")
    eps_real_kk_full = kk_real_from_imag(freq, eps_imag_meas, eps_inf)

    # Calculate relative errors
    rel_error_full = np.abs(eps_real_meas - eps_real_kk_full) / eps_real_meas
    mean_err_full = np.mean(rel_error_full)
    max_err_full = np.max(rel_error_full)

    print(f"Full-range KK check - Mean error: {mean_err_full * 100:.2f}%, Max error: {max_err_full * 100:.2f}%")

    # Partial frequency range analysis (up to 40 GHz)
    f_cut = 40e9  # 40 GHz cutoff
    idx_partial = freq <= f_cut

    if np.sum(idx_partial) > 3:  # Only if we have enough points
        freq_partial = freq[idx_partial]
        eps_real_partial = eps_real_meas[idx_partial]
        eps_imag_partial = eps_imag_meas[idx_partial]

        print(f"Performing partial-range KK analysis (up to {f_cut / 1e9:.0f} GHz)...")
        eps_real_kk_partial = kk_real_from_imag(freq_partial, eps_imag_partial, eps_inf)

        rel_error_partial = np.abs(eps_real_partial - eps_real_kk_partial) / eps_real_partial
        mean_err_partial = np.mean(rel_error_partial)
        max_err_partial = np.max(rel_error_partial)

        print(
            f"Partial-range KK check - Mean error: {mean_err_partial * 100:.2f}%, Max error: {max_err_partial * 100:.2f}%")
    else:
        print("Insufficient data points for partial-range analysis")
        freq_partial = freq
        eps_real_partial = eps_real_meas
        eps_real_kk_partial = eps_real_kk_full
        mean_err_partial = mean_err_full

    # Causality assessment
    causality_threshold = 0.05  # 5% error threshold

    if mean_err_full < causality_threshold:
        causality_status = "PASS"
        print(f"✓ Causality check PASSED (error {mean_err_full * 100:.2f}% < {causality_threshold * 100:.0f}%)")
    else:
        causality_status = "FAIL"
        print(f"✗ Causality check FAILED (error {mean_err_full * 100:.2f}% >= {causality_threshold * 100:.0f}%)")

    # Suggestions based on results
    if mean_err_full > 0.1:  # > 10% error
        print("Suggestion: Large KK errors may indicate:")
        print("  - Measurement artifacts or noise")
        print("  - Insufficient frequency range")
        print("  - Non-causal behavior (unusual)")
    elif mean_err_full > 0.05:  # 5-10% error
        print("Suggestion: Moderate KK errors may indicate:")
        print("  - Limited frequency range")
        print("  - Small measurement uncertainties")

    return {
        "freq_ghz": freq_ghz,
        "eps_real_meas": eps_real_meas,
        "eps_real_kk_full": eps_real_kk_full,
        "eps_imag_meas": eps_imag_meas,
        "mean_err_full": mean_err_full,
        "max_err_full": max_err_full,
        "freq_partial": freq_partial / 1e9,  # Convert back to GHz
        "eps_real_partial": eps_real_partial,
        "eps_real_kk_partial": eps_real_kk_partial,
        "mean_err_partial": mean_err_partial,
        "eps_inf": eps_inf,
        "causality_status": causality_status,
        "causality_threshold": causality_threshold
    }