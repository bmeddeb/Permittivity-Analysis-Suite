"""
Lorentz Oscillator model for dielectric response.

Implements the Lorentz oscillator model for resonant polarization:
ε*(ω) = ε_∞ + Σ(f_i / (ω₀ᵢ² - ω² - jγᵢω))

This model describes resonant effects and bound electron transitions.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel
