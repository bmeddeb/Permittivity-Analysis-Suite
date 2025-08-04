# utils/helpers.py
import numpy as np
import pandas as pd
from typing import Tuple


def get_numeric_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts and validates numeric data from a DataFrame.

    This function looks for columns containing frequency, Dk (real permittivity),
    and Df (imaginary permittivity) and returns them as NumPy arrays. It is
    designed to be case-insensitive and flexible with column naming.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the dielectric data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the frequency (in GHz), Dk, and Df as NumPy arrays.

    Raises
    ------
    ValueError
        If the required columns cannot be found or contain non-numeric data.
    """
    # Normalize column names to lower case for case-insensitive matching
    df.columns = [str(c).lower() for c in df.columns]

    # Identify columns by likely names
    freq_col = next((c for c in df.columns if "freq" in c), None)
    dk_col = next((c for c in df.columns if "dk" in c or "real" in c), None)
    df_col = next((c for c in df.columns if "df" in c or "imag" in c), None)

    if not all([freq_col, dk_col, df_col]):
        raise ValueError("Could not find required columns for Frequency, Dk, and Df.")

    try:
        # Convert to numeric, coercing errors to NaN, then extract as numpy arrays
        freq_ghz = pd.to_numeric(df[freq_col], errors="coerce").to_numpy()
        dk_exp = pd.to_numeric(df[dk_col], errors="coerce").to_numpy()
        df_exp = pd.to_numeric(df[df_col], errors="coerce").to_numpy()

        if np.isnan(freq_ghz).any() or np.isnan(dk_exp).any() or np.isnan(df_exp).any():
            raise ValueError("One or more columns contain non-numeric values.")

        return freq_ghz, dk_exp, df_exp

    except Exception as e:
        raise ValueError(f"Error processing data columns: {e}") from e
