"""
Debye model for dielectric relaxation.

Implements the classic single-pole Debye relaxation model:
ε*(ω) = ε_∞ + Δε / (1 + jωτ)

This model describes materials with a single relaxation time.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel

