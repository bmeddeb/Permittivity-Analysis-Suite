"""
Hybrid Debye-Lorentz model combining relaxation and resonance effects.

Implements a model with both Debye relaxation and Lorentz oscillators:
ε*(ω) = ε_∞ + Σ(Δεᵢ/(1+jωτᵢ)) + Σ(fₖ/(ω₀ₖ²-ω²-jγₖω))

This model describes materials with both relaxation and resonance processes.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel

