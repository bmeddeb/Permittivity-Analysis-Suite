"""
Enhanced preprocessing module combining spline-based approach with comprehensive analysis
Integrates domain-specific dielectric methods with robust algorithm selection
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
from typing import Dict, Tuple, Optional, Callable, List, Union


class EnhancedNoiseAnalyzer:
    """Enhanced noise analysis combining domain-specific and general metrics"""
    
    @staticmethod
    def roughness_metric(y: np.ndarray) -> float:
        """
        Domain-specific roughness metric for dielectric data
        Uses second differences normalized by first differences
        """
        if len(y) < 3:
            return 0.0
        d1 = np.diff(y)
        d2 = np.diff(d1)
        return np.mean(np.abs(d2)) / (np.mean(np.abs(d1)) + 1e-8)
    
    @staticmethod
    def spectral_snr(y: np.ndarray) -> float:
        """
        Spectral SNR using FFT - perfect for frequency-domain dielectric data
        Ratio of low-frequency to high-frequency power
        """
        if len(y) < 4:
            return np.inf
            
        # Remove DC component and apply window
        y_centered = y - np.mean(y)
        window = np.hanning(len(y))
        y_windowed = y_centered * window
        
        # FFT analysis
        Y = np.fft.rfft(y_windowed)
        freqs = np.fft.rfftfreq(len(y))
        
        # Split at median frequency
        cut = len(Y) // 2
        low_power = np.sum(np.abs(Y[:cut])**2)
        high_power = np.sum(np.abs(Y[cut:])**2) + 1e-8
        
        return low_power / high_power
    
    @staticmethod
    def derivative_variance(y: np.ndarray) -> float:
        """Variance of first derivative - general noise metric"""
        if len(y) < 2:
            return 0.0
        return np.var(np.diff(y))
    
    @staticmethod
    def local_variance_ratio(y: np.ndarray, window_size: int = 5) -> float:
        """
        Ratio of local variance to global variance
        High values indicate non-stationary noise
        """
        if len(y) < window_size * 2:
            return 1.0
            
        global_var = np.var(y)
        if global_var == 0:
            return 0.0
            
        # Calculate local variances
        local_vars = []
        for i in range(window_size, len(y) - window_size):
            local_var = np.var(y[i-window_size:i+window_size+1])
            local_vars.append(local_var)
        
        mean_local_var = np.mean(local_vars)
        return mean_local_var / global_var
    
    @classmethod
    def comprehensive_analysis(cls, dk: np.ndarray, df: np.ndarray, 
                             frequency: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive noise analysis combining all metrics
        """
        metrics = {}
        
        # Domain-specific metrics (your approach)
        metrics['dk_roughness'] = cls.roughness_metric(dk)
        metrics['df_roughness'] = cls.roughness_metric(df)
        metrics['dk_spectral_snr'] = cls.spectral_snr(dk)
        metrics['df_spectral_snr'] = cls.spectral_snr(df)
        
        # General metrics (my approach)
        metrics['dk_derivative_var'] = cls.derivative_variance(dk)
        metrics['df_derivative_var'] = cls.derivative_variance(df)
        metrics['dk_local_var_ratio'] = cls.local_variance_ratio(dk)
        metrics['df_local_var_ratio'] = cls.local_variance_ratio(df)
        
        # Estimate noise variance for spline parameter tuning
        dk_noise_var = np.var(dk - gaussian_filter1d(dk, sigma=1))
        df_noise_var = np.var(df - gaussian_filter1d(df, sigma=1))
        metrics['dk_noise_variance'] = dk_noise_var
        metrics['df_noise_variance'] = df_noise_var
        
        # Combined scores
        # Roughness-based score (0-1, higher = noisier)
        roughness_score = np.tanh((metrics['dk_roughness'] + metrics['df_roughness']) / 2)
        
        # SNR-based score (0-1, higher = noisier)
        avg_snr = (metrics['dk_spectral_snr'] + metrics['df_spectral_snr']) / 2
        snr_score = 1 / (1 + avg_snr / 10)  # Sigmoid transformation
        
        # Stability score (0-1, higher = more unstable)
        stability_score = (metrics['dk_local_var_ratio'] + metrics['df_local_var_ratio']) / 2
        stability_score = np.tanh(stability_score)
        
        # Overall noise score
        metrics['overall_noise_score'] = (0.4 * roughness_score + 
                                        0.4 * snr_score + 
                                        0.2 * stability_score)
        
        return metrics


class SplineSmoothing:
    """Spline-based smoothing algorithms - the gold standard for scientific data"""
    
    @staticmethod
    def interpolating_spline(x: np.ndarray, y: np.ndarray, 
                           x_eval: Optional[np.ndarray] = None) -> np.ndarray:
        """Cubic spline that passes exactly through data points"""
        if x_eval is None:
            x_eval = x
        spline = UnivariateSpline(x, y, k=3, s=0)
        return spline(x_eval)
    
    @staticmethod
    def smoothing_spline(x: np.ndarray, y: np.ndarray, s: Optional[float] = None,
                        noise_var: Optional[float] = None) -> np.ndarray:
        """
        Smoothing spline with automatic parameter selection
        Uses s = m * σ² rule when noise_var is provided
        """
        if s is None and noise_var is not None:
            # Scientific rule: s = m * σ²
            s = len(x) * noise_var
        elif s is None:
            # Estimate from data
            # Quick noise estimate using difference between data and smooth version
            y_rough_smooth = gaussian_filter1d(y, sigma=1)
            estimated_var = np.var(y - y_rough_smooth)
            s = len(x) * estimated_var
        
        # Ensure s is reasonable
        s = max(s, 1e-10)  # Avoid s=0 which becomes interpolating
        s = min(s, np.var(y) * len(x))  # Upper bound
        
        spline = UnivariateSpline(x, y, k=3, s=s)
        return spline(x)
    
    @staticmethod
    def pchip_smoothing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """PCHIP interpolation - shape preserving, good for monotonic data"""
        interpolator = PchipInterpolator(x, y)
        return interpolator(x)
    
    @staticmethod
    def lowess_smoothing(x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> np.ndarray:
        """LOWESS smoothing - robust to outliers"""
        try:
            smoothed = lowess(y, x, frac=frac, return_sorted=False)
            return smoothed
        except:
            # Fallback to simple smoothing if LOWESS fails
            return gaussian_filter1d(y, sigma=1)


class LegacySmoothing:
    """Additional smoothing methods from original implementation"""
    
    @staticmethod
    def savitzky_golay(y: np.ndarray, window_size: int = 5, poly_order: int = 2) -> np.ndarray:
        """Savitzky-Golay filter"""
        if window_size >= len(y):
            window_size = len(y) // 2
            if window_size % 2 == 0:
                window_size -= 1
        if window_size < 3:
            return y.copy()
        if poly_order >= window_size:
            poly_order = window_size - 1
        
        try:
            return signal.savgol_filter(y, window_size, poly_order)
        except:
            return y.copy()
    
    @staticmethod
    def gaussian_filter(y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian smoothing"""
        return gaussian_filter1d(y, sigma)
    
    @staticmethod
    def median_filter(y: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Median filter for outlier removal"""
        if window_size >= len(y):
            window_size = len(y) // 2 + 1
        return signal.medfilt(y, kernel_size=window_size)


class EnhancedAlgorithmSelector:
    """Enhanced algorithm selection combining rule-based and quality-based approaches"""
    
    def __init__(self):
        # Primary algorithms (spline-based, domain-specific)
        self.primary_algorithms = {
            'interpolating_spline': SplineSmoothing.interpolating_spline,
            'smoothing_spline': SplineSmoothing.smoothing_spline,
            'pchip': SplineSmoothing.pchip_smoothing,
            'lowess': SplineSmoothing.lowess_smoothing
        }
        
        # Secondary algorithms (general-purpose)
        self.secondary_algorithms = {
            'savitzky_golay': LegacySmoothing.savitzky_golay,
            'gaussian': LegacySmoothing.gaussian_filter,
            'median': LegacySmoothing.median_filter
        }
        
        self.all_algorithms = {**self.primary_algorithms, **self.secondary_algorithms}
    
    def rule_based_selection(self, noise_score: float, n_points: int, 
                           spectral_snr: float) -> Tuple[str, Dict]:
        """
        Rule-based selection (your approach) - domain-specific logic
        """
        if noise_score < 0.1 and spectral_snr > 20:
            return "interpolating_spline", {}
        
        if n_points < 8:
            return "pchip", {}
        
        if noise_score < 0.2:
            return "savitzky_golay", {'window_size': 5, 'poly_order': 2}
        
        if noise_score < 0.4:
            return "smoothing_spline", {}
        
        if noise_score < 0.7:
            # Adapt LOWESS fraction based on noise
            frac = min(0.75, max(0.1, 0.3 * (noise_score / 0.4)))
            return "lowess", {'frac': frac}
        
        # Very noisy data
        return "median", {'window_size': 5}
    
    def quality_based_evaluation(self, x: np.ndarray, y: np.ndarray, 
                               noise_metrics: Dict[str, float]) -> Tuple[str, Dict, float]:
        """
        Quality-based evaluation (my approach) - test multiple algorithms
        """
        best_algorithm = "smoothing_spline"
        best_params = {}
        best_score = -1
        
        # Test primary algorithms first
        for alg_name, alg_func in self.primary_algorithms.items():
            try:
                params = self._get_algorithm_params(alg_name, noise_metrics, len(x))
                
                if alg_name == 'smoothing_spline':
                    # Use noise variance for spline parameter
                    params['noise_var'] = noise_metrics.get('dk_noise_variance', None)
                
                smoothed = alg_func(x, y, **params)
                score = self._evaluate_smoothing_quality(y, smoothed, noise_metrics)
                
                if score > best_score:
                    best_score = score
                    best_algorithm = alg_name
                    best_params = params
                    
            except Exception as e:
                continue
        
        return best_algorithm, best_params, best_score
    
    def _get_algorithm_params(self, alg_name: str, noise_metrics: Dict[str, float], 
                            n_points: int) -> Dict:
        """Get algorithm-specific parameters based on data characteristics"""
        noise_score = noise_metrics.get('overall_noise_score', 0.5)
        
        if alg_name == 'savitzky_golay':
            window_size = max(5, min(n_points // 3, int(7 + 8 * noise_score)))
            return {'window_size': window_size, 'poly_order': 2}
        elif alg_name == 'lowess':
            frac = min(0.75, max(0.1, 0.3 * (noise_score / 0.4)))
            return {'frac': frac}
        elif alg_name == 'gaussian':
            sigma = 0.5 + 2 * noise_score
            return {'sigma': sigma}
        elif alg_name == 'median':
            window_size = max(3, min(n_points // 4, int(3 + 6 * noise_score)))
            return {'window_size': window_size}
        else:
            return {}
    
    def _evaluate_smoothing_quality(self, original: np.ndarray, smoothed: np.ndarray,
                                  noise_metrics: Dict[str, float]) -> float:
        """Evaluate smoothing quality"""
        # Noise reduction
        orig_roughness = EnhancedNoiseAnalyzer.roughness_metric(original)
        smooth_roughness = EnhancedNoiseAnalyzer.roughness_metric(smoothed)
        noise_reduction = max(0, (orig_roughness - smooth_roughness) / (orig_roughness + 1e-10))
        
        # Feature preservation (correlation)
        correlation = np.corrcoef(original, smoothed)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        # Over-smoothing penalty
        signal_loss = 1 - (np.var(smoothed) / (np.var(original) + 1e-10))
        over_smooth_penalty = max(0, signal_loss - 0.3)
        
        return 0.4 * noise_reduction + 0.5 * correlation - 0.1 * over_smooth_penalty
    
    def select_algorithm(self, x: np.ndarray, y: np.ndarray, 
                        noise_metrics: Dict[str, float], 
                        method: str = 'hybrid') -> Tuple[str, Callable, Dict]:
        """
        Main algorithm selection method
        
        Args:
            method: 'rule_based', 'quality_based', or 'hybrid'
        """
        if method == 'rule_based':
            alg_name, params = self.rule_based_selection(
                noise_metrics['overall_noise_score'],
                len(x),
                (noise_metrics['dk_spectral_snr'] + noise_metrics['df_spectral_snr']) / 2
            )
        elif method == 'quality_based':
            alg_name, params, _ = self.quality_based_evaluation(x, y, noise_metrics)
        else:  # hybrid
            # Start with rule-based, then validate with quality evaluation
            rule_alg, rule_params = self.rule_based_selection(
                noise_metrics['overall_noise_score'],
                len(x),
                (noise_metrics['dk_spectral_snr'] + noise_metrics['df_spectral_snr']) / 2
            )
            
            # If rule-based suggests a primary algorithm, use it
            if rule_alg in self.primary_algorithms:
                alg_name, params = rule_alg, rule_params
            else:
                # Otherwise, use quality-based evaluation
                alg_name, params, _ = self.quality_based_evaluation(x, y, noise_metrics)
        
        return alg_name, self.all_algorithms[alg_name], params


class EnhancedDielectricPreprocessor:
    """
    Enhanced dielectric preprocessor combining the best of both approaches
    """
    
    def __init__(self):
        self.noise_analyzer = EnhancedNoiseAnalyzer()
        self.algorithm_selector = EnhancedAlgorithmSelector()
        self.processing_history = []
    
    def preprocess(self, df: pd.DataFrame, apply_smoothing: bool = True,
                  selection_method: str = 'hybrid') -> Tuple[pd.DataFrame, Dict]:
        """
        Enhanced preprocessing with spline-based methods
        
        Args:
            df: DataFrame with [Frequency_GHz, Dk, Df]
            apply_smoothing: Whether to apply smoothing
            selection_method: 'rule_based', 'quality_based', or 'hybrid'
        """
        if df.empty or len(df) < 3:
            return df.copy(), {'status': 'insufficient_data'}
        
        # Extract data
        frequency = df.iloc[:, 0].values
        dk = df.iloc[:, 1].values
        df_loss = df.iloc[:, 2].values
        
        # Enhanced noise analysis
        noise_metrics = self.noise_analyzer.comprehensive_analysis(dk, df_loss, frequency)
        
        processing_info = {
            'original_length': len(df),
            'noise_metrics': noise_metrics,
            'smoothing_applied': False,
            'dk_algorithm': None,
            'df_algorithm': None,
            'dk_params': {},
            'df_params': {},
            'selection_method': selection_method,
            'recommendations': []
        }
        
        # Apply smoothing if needed
        if apply_smoothing and noise_metrics['overall_noise_score'] > 0.15:
            # Select algorithms for Dk and Df
            dk_alg, dk_func, dk_params = self.algorithm_selector.select_algorithm(
                frequency, dk, noise_metrics, selection_method
            )
            
            df_alg, df_func, df_params = self.algorithm_selector.select_algorithm(
                frequency, df_loss, noise_metrics, selection_method
            )
            
            # Apply smoothing
            try:
                dk_smoothed = dk_func(frequency, dk, **dk_params)
                df_smoothed = df_func(frequency, df_loss, **df_params)
                
                # Create result DataFrame
                smoothed_df = df.copy()
                smoothed_df.iloc[:, 1] = dk_smoothed
                smoothed_df.iloc[:, 2] = df_smoothed
                
                processing_info.update({
                    'smoothing_applied': True,
                    'dk_algorithm': dk_alg,
                    'df_algorithm': df_alg,
                    'dk_params': dk_params,
                    'df_params': df_params
                })
                
                # Add recommendations
                self._add_recommendations(processing_info, noise_metrics)
                
                return smoothed_df, processing_info
                
            except Exception as e:
                processing_info['recommendations'].append(f'Smoothing failed: {str(e)}')
                return df.copy(), processing_info
        
        else:
            if noise_metrics['overall_noise_score'] <= 0.15:
                processing_info['recommendations'].append('Data quality is excellent - no smoothing needed')
            
            return df.copy(), processing_info
    
    def _add_recommendations(self, processing_info: Dict, noise_metrics: Dict[str, float]):
        """Add processing recommendations based on analysis"""
        recommendations = []
        
        if noise_metrics['overall_noise_score'] > 0.7:
            recommendations.append('High noise detected - consider data acquisition quality')
        
        if noise_metrics['dk_spectral_snr'] < 5:
            recommendations.append('Low spectral SNR in Dk - may affect model fitting')
        
        if noise_metrics['df_spectral_snr'] < 5:
            recommendations.append('Low spectral SNR in Df - consider longer averaging')
        
        if processing_info['dk_algorithm'] in ['median', 'lowess']:
            recommendations.append('Robust smoothing applied - data may have outliers')
        
        if processing_info['dk_algorithm'] == 'interpolating_spline':
            recommendations.append('High-quality data detected - using interpolating spline')
        
        processing_info['recommendations'] = recommendations
    
    def preprocess_manual(self, df: pd.DataFrame, algorithm_name: str, 
                         manual_params: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Manual preprocessing with specific algorithm selection
        
        Args:
            df: DataFrame with [Frequency_GHz, Dk, Df]
            algorithm_name: Specific algorithm name to use
            manual_params: Optional dictionary of algorithm-specific parameters from UI
        """
        if df.empty or len(df) < 3:
            return df.copy(), {'status': 'insufficient_data'}
        
        # Extract data
        frequency = df.iloc[:, 0].values
        dk = df.iloc[:, 1].values
        df_loss = df.iloc[:, 2].values
        
        # Analyze noise for information purposes
        noise_metrics = self.noise_analyzer.comprehensive_analysis(dk, df_loss, frequency)
        
        processing_info = {
            'original_length': len(df),
            'noise_metrics': noise_metrics,
            'smoothing_applied': False,
            'dk_algorithm': algorithm_name,
            'df_algorithm': algorithm_name,
            'dk_params': {},
            'df_params': {},
            'selection_method': 'manual',
            'preprocessing_mode': 'manual',
            'recommendations': []
        }
        
        # Get the algorithm function
        if algorithm_name not in self.algorithm_selector.all_algorithms:
            processing_info['error'] = f'Unknown algorithm: {algorithm_name}'
            return df.copy(), processing_info
        
        algorithm_func = self.algorithm_selector.all_algorithms[algorithm_name]
        
        # Get default parameters for the algorithm
        dk_params = self.algorithm_selector._get_algorithm_params(
            algorithm_name, noise_metrics, len(frequency)
        )
        df_params = dk_params.copy()
        
        # Override with manual parameters from UI if provided
        if manual_params:
            if algorithm_name == 'smoothing_spline':
                if manual_params.get('spline_s_mode') == 'manual' and manual_params.get('spline_s_value'):
                    dk_params['s'] = manual_params['spline_s_value']
                    df_params['s'] = manual_params['spline_s_value']
                else:
                    # Use automatic s = m×σ² rule
                    dk_params['noise_var'] = noise_metrics.get('dk_noise_variance')
                    df_params['noise_var'] = noise_metrics.get('df_noise_variance')
            elif algorithm_name == 'lowess':
                if manual_params.get('lowess_frac'):
                    dk_params['frac'] = manual_params['lowess_frac']
                    df_params['frac'] = manual_params['lowess_frac']
            elif algorithm_name == 'savitzky_golay':
                if manual_params.get('savgol_window'):
                    dk_params['window_size'] = int(manual_params['savgol_window'])
                    df_params['window_size'] = int(manual_params['savgol_window'])
                if manual_params.get('savgol_polyorder'):
                    dk_params['poly_order'] = int(manual_params['savgol_polyorder'])
                    df_params['poly_order'] = int(manual_params['savgol_polyorder'])
            elif algorithm_name == 'gaussian':
                if manual_params.get('gaussian_sigma'):
                    dk_params['sigma'] = manual_params['gaussian_sigma']
                    df_params['sigma'] = manual_params['gaussian_sigma']
            elif algorithm_name == 'median':
                if manual_params.get('median_window'):
                    dk_params['window_size'] = int(manual_params['median_window'])
                    df_params['window_size'] = int(manual_params['median_window'])
        else:
            # Add special parameters for spline methods (default behavior)
            if algorithm_name == 'smoothing_spline':
                dk_params['noise_var'] = noise_metrics.get('dk_noise_variance')
                df_params['noise_var'] = noise_metrics.get('df_noise_variance')
        
        try:
            # Apply manual algorithm
            dk_smoothed = algorithm_func(frequency, dk, **dk_params)
            df_smoothed = algorithm_func(frequency, df_loss, **df_params)
            
            # Create result DataFrame
            smoothed_df = df.copy()
            smoothed_df.iloc[:, 1] = dk_smoothed
            smoothed_df.iloc[:, 2] = df_smoothed
            
            processing_info.update({
                'smoothing_applied': True,
                'dk_params': dk_params,
                'df_params': df_params,
                'recommendations': [f'Manual {algorithm_name} applied successfully']
            })
            
            return smoothed_df, processing_info
            
        except Exception as e:
            processing_info['error'] = f'Manual preprocessing failed: {str(e)}'
            processing_info['recommendations'] = [f'Algorithm {algorithm_name} failed, using original data']
            return df.copy(), processing_info
    
    def get_algorithm_info(self) -> Dict[str, str]:
        """Get information about available algorithms"""
        return {
            'primary_algorithms': 'Spline-based methods optimized for dielectric data',
            'secondary_algorithms': 'General-purpose smoothing for special cases',
            'selection_methods': 'rule_based (fast), quality_based (thorough), hybrid (recommended)'
        }