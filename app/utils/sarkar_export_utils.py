# app/models/sarkar_export_utils.py
"""
Export utilities for the Djordjevic-Sarkar model.

Provides functionality to export model results to various formats including
SPICE models, Touchstone files, and visualization data.
"""

import numpy as np
import lmfit
from typing import Dict, Any, Optional, Tuple, List
import logging

from app.models import SarkarModel

logger = logging.getLogger(__name__)


class SarkarExportUtils:
    """
    Export utilities for Sarkar model results.
    """

    def __init__(self, model: SarkarModel):
        """
        Initialize with a SarkarModel instance.

        Args:
            model: SarkarModel instance
        """
        self.model = model
        self.eps0 = 8.854187817e-12

    def generate_spice_model(
        self, params: lmfit.Parameters, z0: float = 50.0, format: str = "behavioral"
    ) -> Dict[str, Any]:
        """
        Generate SPICE-compatible model parameters.

        Args:
            params: Model parameters
            z0: Reference impedance (ohms)
            format: 'behavioral' or 'ladder' format

        Returns:
            Dictionary with SPICE model elements
        """
        p = params.valuesdict()

        # Basic parameters
        spice_model = {
            "type": "Djordjevic-Sarkar",
            "format": format,
            "reference_impedance": z0,
            "parameters": {
                "eps_r_inf": p["eps_r_inf"],
                "eps_r_s": p["eps_r_s"],
                "f1_hz": p["f1"] * 1e9 if self.model.use_ghz else p["f1"],
                "f2_hz": p["f2"] * 1e9 if self.model.use_ghz else p["f2"],
                "sigma_dc": p["sigma_dc"],
            },
        }

        if format == "behavioral":
            # Behavioral model using Laplace transform
            spice_model["laplace"] = self._generate_laplace_expression(p)
            spice_model["netlist"] = self._generate_behavioral_netlist(p, z0)

        elif format == "ladder":
            # RC ladder approximation
            ladder_params = self._calculate_rc_ladder(p)
            spice_model["ladder"] = ladder_params
            spice_model["netlist"] = self._generate_ladder_netlist(ladder_params, z0)

        # Add comments
        spice_model["comments"] = [
            f"* Djordjevic-Sarkar Model",
            f"* eps_r: {p['eps_r_inf']:.3f} to {p['eps_r_s']:.3f}",
            f"* Transitions: {p['f1']:.3e} to {p['f2']:.3e} "
            + ("GHz" if self.model.use_ghz else "Hz"),
            f"* DC conductivity: {p['sigma_dc']:.3e} S/m",
        ]

        return spice_model

    def _generate_laplace_expression(self, params: Dict[str, float]) -> str:
        """Generate Laplace domain expression for behavioral model."""
        # Simplified expression for SPICE
        # Note: This is an approximation suitable for time-domain simulation
        eps_inf = params["eps_r_inf"]
        eps_s = params["eps_r_s"]
        omega1 = 2 * np.pi * params["f1"] * (1e9 if self.model.use_ghz else 1)
        omega2 = 2 * np.pi * params["f2"] * (1e9 if self.model.use_ghz else 1)

        # Approximate with modified Debye form
        tau_avg = 1 / np.sqrt(omega1 * omega2)
        delta_eps = eps_s - eps_inf

        return f"({eps_inf} + {delta_eps}/(1 + s*{tau_avg}))"

    def _generate_behavioral_netlist(
        self, params: Dict[str, float], z0: float
    ) -> List[str]:
        """Generate behavioral SPICE netlist."""
        netlist = []
        netlist.append("* Behavioral D-S Model Subcircuit")
        netlist.append(".SUBCKT DS_MODEL IN OUT REF")
        netlist.append("* Voltage-controlled current source")
        netlist.append(
            "G1 IN OUT LAPLACE {V(IN,REF)} = "
            + self._generate_laplace_expression(params)
        )
        netlist.append(".ENDS DS_MODEL")
        return netlist

    def _calculate_rc_ladder(
        self, params: Dict[str, float], n_sections: int = 5
    ) -> Dict[str, Any]:
        """Calculate RC ladder network approximation."""
        eps_inf = params["eps_r_inf"]
        eps_s = params["eps_r_s"]
        f1 = params["f1"] * (1e9 if self.model.use_ghz else 1)
        f2 = params["f2"] * (1e9 if self.model.use_ghz else 1)

        # Distribute time constants logarithmically
        freqs = np.logspace(np.log10(f1), np.log10(f2), n_sections)

        # Calculate R and C values for each section
        sections = []
        delta_eps_per_section = (eps_s - eps_inf) / n_sections

        for i, f in enumerate(freqs):
            tau = 1 / (2 * np.pi * f)
            # Assuming unit length
            c_section = (eps_inf + i * delta_eps_per_section) * self.eps0
            r_section = tau / c_section

            sections.append({"R": r_section, "C": c_section, "f": f, "tau": tau})

        return {
            "n_sections": n_sections,
            "sections": sections,
            "r_dc": 1 / params["sigma_dc"] if params["sigma_dc"] > 0 else 1e12,
        }

    def _generate_ladder_netlist(
        self, ladder_params: Dict[str, Any], z0: float
    ) -> List[str]:
        """Generate RC ladder SPICE netlist."""
        netlist = []
        netlist.append("* RC Ladder Approximation of D-S Model")
        netlist.append(".SUBCKT DS_LADDER IN OUT")

        # DC resistance
        netlist.append(f"RDC IN n1 {ladder_params['r_dc']:.3e}")

        # RC sections
        for i, section in enumerate(ladder_params["sections"]):
            node_in = f"n{i + 1}"
            node_out = f"n{i + 2}" if i < len(ladder_params["sections"]) - 1 else "OUT"
            netlist.append(f"R{i + 1} {node_in} {node_out} {section['R']:.3e}")
            netlist.append(f"C{i + 1} {node_out} 0 {section['C']:.3e}")

        netlist.append(".ENDS DS_LADDER")
        return netlist

    def to_touchstone(
        self,
        params: lmfit.Parameters,
        freq_hz: np.ndarray,
        thickness_m: float = 1e-3,
        z0: float = 50.0,
        eps_r_ref: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Convert to Touchstone S-parameter format for a transmission line.

        Args:
            params: Model parameters
            freq_hz: Frequency array in Hz
            thickness_m: Material thickness in meters
            z0: Reference impedance (ohms)
            eps_r_ref: Reference permittivity (usually 1 for air)

        Returns:
            Dictionary with S-parameters and transmission line properties
        """
        # Convert frequency if needed
        freq_model = freq_hz / 1e9 if self.model.use_ghz else freq_hz

        # Calculate complex permittivity
        eps_r = self.model.predict(freq_model, params)

        # Effective permittivity (for microstrip, this would need geometry)
        eps_eff = eps_r / eps_r_ref

        # Propagation constant
        c0 = 3e8  # Speed of light
        beta0 = 2 * np.pi * freq_hz / c0
        gamma = 1j * beta0 * np.sqrt(eps_eff)

        # Add loss from conductivity
        sigma = params.valuesdict()["sigma_dc"]
        if sigma > 0:
            alpha = sigma / (2 * np.sqrt(eps_eff.real) * self.eps0 * c0)
            gamma = alpha + 1j * np.imag(gamma)

        # Characteristic impedance (simplified TEM assumption)
        z_c = z0 / np.sqrt(eps_eff)

        # Calculate S-parameters for the transmission line section
        gamma_l = gamma * thickness_m
        z_ratio = (z_c - z0) / (z_c + z0)

        # Avoid numerical issues with large gamma_l
        exp_gamma_l = np.exp(-gamma_l)
        exp_gamma_l = np.where(np.abs(gamma_l) > 100, 0, exp_gamma_l)

        denominator = 1 - z_ratio**2 * exp_gamma_l**2

        s11 = z_ratio * (1 - exp_gamma_l**2) / denominator
        s21 = (1 - z_ratio**2) * exp_gamma_l / denominator
        s12 = s21  # Reciprocal network
        s22 = s11  # Symmetric network

        # Calculate additional transmission line parameters
        attenuation_db_m = 8.686 * np.real(gamma)  # Nepers to dB
        phase_constant_rad_m = np.imag(gamma)
        phase_velocity = 2 * np.pi * freq_hz / phase_constant_rad_m

        return {
            "frequency_hz": freq_hz,
            "s11": s11,
            "s21": s21,
            "s12": s12,
            "s22": s22,
            "s_matrix": np.array([[s11, s12], [s21, s22]]).transpose(2, 0, 1),
            "z_characteristic": z_c,
            "propagation_constant": gamma,
            "attenuation_dB_m": attenuation_db_m,
            "phase_constant_rad_m": phase_constant_rad_m,
            "phase_velocity_m_s": phase_velocity,
            "eps_effective": eps_eff,
        }

    def generate_touchstone_file(
        self,
        params: lmfit.Parameters,
        freq_hz: np.ndarray,
        filename: str,
        thickness_m: float = 1e-3,
        z0: float = 50.0,
        format: str = "RI",
    ) -> None:
        """
        Write Touchstone .s2p file.

        Args:
            params: Model parameters
            freq_hz: Frequency array in Hz
            filename: Output filename
            thickness_m: Material thickness
            z0: Reference impedance
            format: 'RI' (real-imaginary) or 'MA' (magnitude-angle)
        """
        # Get S-parameters
        s_data = self.to_touchstone(params, freq_hz, thickness_m, z0)

        # Write file
        with open(filename, "w") as f:
            # Header
            f.write("! Djordjevic-Sarkar Model S-Parameters\n")
            f.write(f"! Thickness: {thickness_m * 1e3:.3f} mm\n")

            p = params.valuesdict()
            f.write(f"! eps_r: {p['eps_r_inf']:.3f} to {p['eps_r_s']:.3f}\n")
            f.write(
                f"! f1: {p['f1']:.3e} " + ("GHz" if self.model.use_ghz else "Hz") + "\n"
            )
            f.write(
                f"! f2: {p['f2']:.3e} " + ("GHz" if self.model.use_ghz else "Hz") + "\n"
            )
            f.write(f"! sigma_dc: {p['sigma_dc']:.3e} S/m\n")

            # Format line
            f.write(f"# Hz S {format} R {z0}\n")

            # Data
            for i, freq in enumerate(freq_hz):
                if format == "RI":
                    f.write(f"{freq:.6e} ")
                    f.write(f"{s_data['s11'][i].real:.6e} {s_data['s11'][i].imag:.6e} ")
                    f.write(f"{s_data['s21'][i].real:.6e} {s_data['s21'][i].imag:.6e} ")
                    f.write(f"{s_data['s12'][i].real:.6e} {s_data['s12'][i].imag:.6e} ")
                    f.write(
                        f"{s_data['s22'][i].real:.6e} {s_data['s22'][i].imag:.6e}\n"
                    )
                else:  # MA format
                    f.write(f"{freq:.6e} ")
                    for s_param in ["s11", "s21", "s12", "s22"]:
                        mag = np.abs(s_data[s_param][i])
                        ang = np.angle(s_data[s_param][i], deg=True)
                        f.write(f"{mag:.6e} {ang:.6f} ")
                    f.write("\n")

    def extrapolate_with_uncertainty(
        self,
        params: lmfit.Parameters,
        freq_extrap: np.ndarray,
        freq_data: np.ndarray,
        confidence: float = 0.95,
        extrapolation_penalty: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """
        Extrapolate model with uncertainty estimates.

        Args:
            params: Model parameters
            freq_extrap: Extrapolation frequencies (same unit as model)
            freq_data: Original data frequencies for reference
            confidence: Confidence level (0.95 or 0.99)
            extrapolation_penalty: Uncertainty increase per decade of extrapolation

        Returns:
            Dictionary with extrapolated values and uncertainties
        """
        # Basic extrapolation
        eps_extrap = self.model.predict(freq_extrap, params)
        dk_extrap = eps_extrap.real
        df_extrap = eps_extrap.imag / eps_extrap.real

        # Calculate distance from data (in log space)
        log_freq = np.log10(freq_extrap)
        log_data_min = np.log10(np.min(freq_data))
        log_data_max = np.log10(np.max(freq_data))

        # Distance from nearest data point (0 if within range)
        distance = np.maximum(
            0, np.maximum(log_data_min - log_freq, log_freq - log_data_max)
        )

        # Base uncertainty from parameter uncertainties
        base_uncertainty = self._estimate_base_uncertainty(params)

        # Uncertainty grows with extrapolation distance
        uncertainty_factor = 1 + distance * extrapolation_penalty

        # Apply to Dk and Df
        dk_uncertainty = dk_extrap * base_uncertainty * uncertainty_factor
        df_uncertainty = df_extrap * base_uncertainty * uncertainty_factor

        # Confidence intervals
        z_score = 1.96 if confidence == 0.95 else 2.58

        result = {
            "frequency": freq_extrap,
            "dk": dk_extrap,
            "df": df_extrap,
            "eps_real": eps_extrap.real,
            "eps_imag": eps_extrap.imag,
            "dk_uncertainty": dk_uncertainty,
            "df_uncertainty": df_uncertainty,
            "dk_lower": dk_extrap - z_score * dk_uncertainty,
            "dk_upper": dk_extrap + z_score * dk_uncertainty,
            "df_lower": np.maximum(0, df_extrap - z_score * df_uncertainty),
            "df_upper": df_extrap + z_score * df_uncertainty,
            "extrapolation_distance": distance,
            "confidence_level": confidence,
        }

        # Add warnings
        max_distance = np.max(distance)
        if max_distance > 2:
            result["warning"] = (
                f"Extreme extrapolation: {max_distance:.1f} decades beyond data"
            )
        elif max_distance > 1:
            result["warning"] = (
                f"Significant extrapolation: {max_distance:.1f} decades beyond data"
            )

        return result

    def _estimate_base_uncertainty(self, params: lmfit.Parameters) -> float:
        """Estimate base uncertainty from parameter uncertainties."""
        # Simple estimate based on parameter relative errors
        rel_errors = []

        for param_name in ["eps_r_inf", "eps_r_s", "f1", "f2"]:
            param = params[param_name]
            if param.stderr is not None and param.value != 0:
                rel_errors.append(param.stderr / abs(param.value))

        if rel_errors:
            # Use RMS of relative errors
            return np.sqrt(np.mean(np.array(rel_errors) ** 2))
        else:
            # Default uncertainty if no error estimates available
            return 0.01  # 1%

    def generate_plot_data(
        self,
        result: lmfit.model.ModelResult,
        n_points: int = 1000,
        freq_range_factor: float = 10,
        include_components: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data for plotting.

        Args:
            result: Fit result
            n_points: Number of points for smooth curves
            freq_range_factor: Factor to extend frequency range for plots
            include_components: Include component breakdown

        Returns:
            Dictionary with plot data
        """
        # Frequency range for plotting
        freq_min = result.freq.min() / freq_range_factor
        freq_max = result.freq.max() * freq_range_factor
        freq_plot = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)

        # Model predictions
        eps_model = self.model.predict(freq_plot, result.params)

        # Basic plot data
        plot_data = {
            "freq": freq_plot,
            "freq_ghz": freq_plot if self.model.use_ghz else freq_plot / 1e9,
            "dk_fit": eps_model.real,
            "df_fit": eps_model.imag / eps_model.real,
            "eps_real": eps_model.real,
            "eps_imag": eps_model.imag,
            "tan_delta": eps_model.imag / eps_model.real,
            "freq_exp": result.freq,
            "freq_exp_ghz": result.freq if self.model.use_ghz else result.freq / 1e9,
            "dk_exp": result.dk_exp,
            "df_exp": result.df_exp,
        }

        # Add residuals
        eps_fit_at_data = self.model.predict(result.freq, result.params)
        plot_data["dk_residual"] = result.dk_exp - eps_fit_at_data.real
        plot_data["df_residual"] = (
            result.df_exp - eps_fit_at_data.imag / eps_fit_at_data.real
        )

        # Add transition markers
        p = result.params.valuesdict()
        plot_data["f1"] = p["f1"] if self.model.use_ghz else p["f1"] / 1e9
        plot_data["f2"] = p["f2"] if self.model.use_ghz else p["f2"] / 1e9
        plot_data["f_center"] = (
            np.sqrt(p["f1"] * p["f2"])
            if self.model.use_ghz
            else np.sqrt(p["f1"] * p["f2"]) / 1e9
        )

        # Component breakdown if requested
        if include_components:
            components = self._calculate_components(freq_plot, result.params)
            plot_data["components"] = components

        # Add uncertainty bands if available
        uncertainty_data = self.extrapolate_with_uncertainty(
            result.params, freq_plot, result.freq, confidence=0.95
        )
        plot_data["dk_uncertainty_band"] = {
            "lower": uncertainty_data["dk_lower"],
            "upper": uncertainty_data["dk_upper"],
        }
        plot_data["df_uncertainty_band"] = {
            "lower": uncertainty_data["df_lower"],
            "upper": uncertainty_data["df_upper"],
        }

        # Mark extrapolation regions
        in_data_range = (freq_plot >= result.freq.min()) & (
            freq_plot <= result.freq.max()
        )
        plot_data["is_interpolation"] = in_data_range
        plot_data["is_extrapolation"] = ~in_data_range

        return plot_data

    def _calculate_components(
        self, freq: np.ndarray, params: lmfit.Parameters
    ) -> Dict[str, np.ndarray]:
        """Calculate individual components of the model."""
        p = params.valuesdict()

        # High-frequency component
        eps_inf_component = np.full_like(freq, p["eps_r_inf"], dtype=complex)

        # Calculate dispersion component only
        params_no_dc = params.copy()
        params_no_dc["sigma_dc"].value = 0
        eps_no_dc = self.model.predict(freq, params_no_dc)
        dispersion_component = eps_no_dc - eps_inf_component

        # DC conductivity component
        if self.model.use_ghz:
            omega = 2 * np.pi * freq * 1e9
        else:
            omega = 2 * np.pi * freq
        dc_component = -1j * p["sigma_dc"] / (omega * self.eps0)

        return {
            "eps_inf": eps_inf_component,
            "dispersion": dispersion_component,
            "dc_conductivity": dc_component,
            "total": eps_inf_component + dispersion_component + dc_component,
        }
