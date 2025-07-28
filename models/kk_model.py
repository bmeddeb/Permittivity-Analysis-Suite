# models/kk_model.py
import numpy as np
from .base_model import BaseModel
from .kk_check import kk_causality_check


class KKModel(BaseModel):
    """
    Kramers-Kronig Causality Check Model

    This model doesn't fit parameters but checks if the experimental data
    satisfies the Kramers-Kronig relations (causality requirement).
    """

    def analyze(self, df):
        """
        Perform Kramers-Kronig causality analysis on experimental data

        Args:
            df: DataFrame with experimental data [Frequency_GHz, Dk, Df]

        Returns:
            Dictionary with KK analysis results compatible with other models
        """
        # Perform the KK causality check
        kk_results = kk_causality_check(df)

        # Calculate RMSE between measured and KK-reconstructed real permittivity
        eps_real_meas = kk_results["eps_real_meas"]
        eps_real_kk = kk_results["eps_real_kk_full"]

        rmse_dk = np.sqrt(np.mean((eps_real_meas - eps_real_kk) ** 2))

        # Since this is a causality check, not a fit, we don't have fitted Df
        # We return the measured imaginary part
        eps_imag_meas = kk_results["eps_imag_meas"]

        # Create complex permittivity for consistency with other models
        eps_fit = eps_real_kk + 1j * eps_imag_meas

        print(f"KK Analysis Summary:")
        print(f"RMSE between measured and KK-reconstructed Dk: {rmse_dk:.4f}")
        print(f"Causality status: {kk_results['causality_status']}")

        # Create fitted_params in expected format - KK model only has eps_inf
        fitted_params = {
            'eps_inf': kk_results["eps_inf"]
        }

        # Return results in format compatible with other models
        return {
            "freq": kk_results["freq_ghz"],
            "freq_ghz": kk_results["freq_ghz"],
            "eps_fit": eps_fit,
            "dk_fit": eps_real_kk,  # KK-reconstructed real part
            "df_fit": eps_imag_meas,  # Measured imaginary part
            "eps_real_kk_full": eps_real_kk,  # For plotting
            "eps_imag_kk_full": eps_imag_meas,  # For plotting
            "dk_exp": eps_real_meas,
            "df_exp": eps_imag_meas,
            "success": kk_results["causality_status"] == "PASS",
            "rmse": rmse_dk,
            "rmse_dk": rmse_dk,
            "rmse_df": 0.0,  # Not applicable for KK check
            "cost": kk_results["mean_err_full"],
            # Only eps_inf as "parameter"
            "params_fit": [kk_results["eps_inf"]],
            "fitted_params": fitted_params,  # Add expected fitted_params dictionary

            # Additional KK-specific results
            "mean_error_full": kk_results["mean_err_full"],
            "max_error_full": kk_results["max_err_full"],
            "causality_status": kk_results["causality_status"],
            "causality_threshold": kk_results["causality_threshold"],
            "eps_inf_estimated": kk_results["eps_inf"],

            # Partial range results (if available)
            "freq_partial": kk_results.get("freq_partial", kk_results["freq_ghz"]),
            "eps_real_kk_partial": kk_results.get("eps_real_kk_partial", eps_real_kk),
            "mean_error_partial": kk_results.get("mean_err_partial", kk_results["mean_err_full"])
        }

    def model(self, params, freq):
        """
        KK model doesn't have a parametric form - this is for compatibility
        """
        # For KK check, we don't have a parametric model
        # Return zeros as placeholder
        return np.zeros(len(freq), dtype=complex)

    def objective(self, params, freq, dk_exp, df_exp):
        """
        KK check doesn't optimize parameters - this is for compatibility
        """
        return np.array([0.0])
