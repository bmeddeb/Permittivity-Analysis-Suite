"""
D. Sarkar model for dielectric response with conduction.

Implements the Sarkar model combining Debye relaxation with conductivity:
ε*(ω) = ε_∞ + Δε/(1+jωτ) + σ/(jωε₀)

This model includes explicit conduction losses for broadband materials.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel

# Physical constants
EPSILON_0 = 8.854187817e-12  # F/m, vacuum permittivity
