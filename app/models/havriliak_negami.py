"""
Havriliak-Negami model for dielectric relaxation.

Implements the most general empirical dielectric relaxation model:
ε*(ω) = ε_∞ + Δε / (1 + (jωτ)^α)^β

This model combines both symmetric (α) and asymmetric (β) broadening factors.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel
