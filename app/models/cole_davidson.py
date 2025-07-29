"""
Cole-Davidson model for dielectric relaxation.

Implements the Cole-Davidson relaxation model with asymmetric broadening:
ε*(ω) = ε_∞ + Δε / (1 + jωτ)^β

This model describes materials with asymmetric distribution of relaxation times.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel
