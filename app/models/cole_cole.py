"""
Cole-Cole model for dielectric relaxation.

Implements the Cole-Cole relaxation model with symmetric broadening:
ε*(ω) = ε_∞ + Δε / (1 + (jωτ)^(1-α))

This model describes materials with broad symmetric distribution of relaxation times.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel

