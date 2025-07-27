from .kk_check import kk_causality_check

class KKModel:
    """Wrapper class for KK Causality Check so it behaves like other models."""
    def analyze(self, df):
        return kk_causality_check(df)
