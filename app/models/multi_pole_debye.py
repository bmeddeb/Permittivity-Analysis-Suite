"""
Multi-Pole Debye model for dielectric relaxation.

Implements a sum of multiple Debye relaxation processes:
ε*(ω) = ε_∞ + Σ(Δε_i / (1 + jωτ_i))

This model describes materials with multiple distinct relaxation times.
"""
from app.models import BaseModel


class MultiPoleDebyeModel(BaseModel):
    pass