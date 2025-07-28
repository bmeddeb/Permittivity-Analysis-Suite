"""
Preprocessing module for dielectric permittivity data
Handles noise analysis, smoothing algorithm selection, and data cleaning
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error
import warnings
from typing import Dict, Tuple, Optional, Callable, List


class NoiseAnalyzer:
    """Analyzes noise characteristics in dielectric data"""
    
    @staticmethod
    def calculate_roughness(data: np.ndarray) -> float:
        """
        Calculate data roughness using total variation
        Higher values indicate more noise/roughness
        """
        if len(data) < 2:
            return 0.0
        return np.sum(np.abs(np.diff(data))) / (np.max(data) - np.min(data) + 1e-10)
    
    @staticmethod
    def calculate_snr(data: np.ndarray, smoothed_data: np.ndarray = None) -> float:
        """
        Calculate Signal-to-Noise Ratio
        If smoothed_data not provided, uses moving average
        """
        if smoothed_data is None:
            window_size = max(3, len(data) // 10)
            smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        
        signal_power = np.var(smoothed_data)
        noise_power = np.var(data - smoothed_data)
        
        if noise_power == 0:
            return np.inf
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def calculate_derivative_variance(data: np.ndarray) -> float:
        """
        Calculate variance of first derivative
        High values suggest noisy data
        """
        if len(data) < 2:
            return 0.0
        return np.var(np.diff(data))
    
    @staticmethod
    def calculate_high_frequency_content(data: np.ndarray, frequency: np.ndarray) -> float:
        """
        Calculate high-frequency content using FFT
        Higher values suggest more noise
        """
        if len(data) < 4:
            return 0.0
            
        # Use FFT to analyze frequency content
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        
        # Calculate power in high frequency components (upper 30% of spectrum)
        high_freq_threshold = 0.3
        high_freq_mask = np.abs(freqs) > high_freq_threshold
        
        total_power = np.sum(np.abs(fft_data)**2)
        high_freq_power = np.sum(np.abs(fft_data[high_freq_mask])**2)
        
        if total_power == 0:
            return 0.0
        return high_freq_power / total_power
    
    @classmethod
    def analyze_noise(cls, dk: np.ndarray, df: np.ndarray, frequency: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive noise analysis for Dk and Df data
        """
        metrics = {}
        
        # Roughness metrics
        metrics['dk_roughness'] = cls.calculate_roughness(dk)
        metrics['df_roughness'] = cls.calculate_roughness(df)
        
        # SNR metrics
        metrics['dk_snr'] = cls.calculate_snr(dk)
        metrics['df_snr'] = cls.calculate_snr(df)
        
        # Derivative variance
        metrics['dk_derivative_var'] = cls.calculate_derivative_variance(dk)
        metrics['df_derivative_var'] = cls.calculate_derivative_variance(df)
        
        # High frequency content
        metrics['dk_high_freq'] = cls.calculate_high_frequency_content(dk, frequency)
        metrics['df_high_freq'] = cls.calculate_high_frequency_content(df, frequency)
        
        # Combined noise score (0-1 scale, higher = more noisy)
        roughness_score = (metrics['dk_roughness'] + metrics['df_roughness']) / 2
        snr_score = 1 / (1 + np.exp((metrics['dk_snr'] + metrics['df_snr']) / 2 - 10))  # Sigmoid
        hf_score = (metrics['dk_high_freq'] + metrics['df_high_freq']) / 2
        
        metrics['combined_noise_score'] = (roughness_score + snr_score + hf_score) / 3
        
        return metrics


class SmoothingAlgorithms:
    """Collection of smoothing algorithms for dielectric data"""
    
    @staticmethod
    def moving_average(data: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Simple moving average smoothing"""
        if window_size >= len(data):
            window_size = len(data) // 2 + 1
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    
    @staticmethod
    def savitzky_golay(data: np.ndarray, window_size: int = 5, poly_order: int = 2) -> np.ndarray:
        """Savitzky-Golay filter for preserving peak shapes"""
        if window_size >= len(data):
            window_size = len(data) // 2
            if window_size % 2 == 0:
                window_size -= 1
        if window_size < 3:
            return data.copy()
        if poly_order >= window_size:
            poly_order = window_size - 1
            
        try:
            return signal.savgol_filter(data, window_size, poly_order)
        except:
            return data.copy()
    
    @staticmethod
    def gaussian_filter(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian smoothing filter"""
        return gaussian_filter1d(data, sigma)
    
    @staticmethod
    def butterworth_filter(data: np.ndarray, frequency: np.ndarray, 
                          cutoff_ratio: float = 0.1, order: int = 4) -> np.ndarray:
        """Butterworth low-pass filter"""
        if len(data) < 10:
            return data.copy()
            
        try:
            # Calculate Nyquist frequency
            fs = 2 / np.mean(np.diff(frequency))  # Sampling frequency
            nyquist = fs / 2
            cutoff = cutoff_ratio * nyquist
            
            # Design filter
            b, a = signal.butter(order, cutoff / nyquist, btype='low')
            return signal.filtfilt(b, a, data)
        except:
            return data.copy()
    
    @staticmethod
    def median_filter(data: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Median filter for removing outliers"""
        if window_size >= len(data):
            window_size = len(data) // 2 + 1
        return signal.medfilt(data, kernel_size=window_size)
    
    @staticmethod
    def wiener_filter(data: np.ndarray, noise_var: Optional[float] = None) -> np.ndarray:
        """Wiener filter for optimal noise reduction"""
        try:
            return signal.wiener(data, noise=noise_var)
        except:
            return data.copy()


class SmoothinSelector:
    """Selects the best smoothing algorithm based on data characteristics"""
    
    def __init__(self):
        self.algorithms = {
            'moving_average': SmoothingAlgorithms.moving_average,
            'savitzky_golay': SmoothingAlgorithms.savitzky_golay,
            'gaussian': SmoothingAlgorithms.gaussian_filter,
            'butterworth': SmoothingAlgorithms.butterworth_filter,
            'median': SmoothingAlgorithms.median_filter,
            'wiener': SmoothingAlgorithms.wiener_filter
        }
    
    def evaluate_smoothing_quality(self, original: np.ndarray, smoothed: np.ndarray, 
                                 noise_score: float) -> float:
        """
        Evaluate smoothing quality based on:
        - Noise reduction
        - Feature preservation
        - Smoothness improvement
        """
        # Noise reduction score
        orig_roughness = NoiseAnalyzer.calculate_roughness(original)
        smooth_roughness = NoiseAnalyzer.calculate_roughness(smoothed)
        noise_reduction = max(0, (orig_roughness - smooth_roughness) / (orig_roughness + 1e-10))
        
        # Feature preservation (correlation)
        correlation = np.corrcoef(original, smoothed)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        # Over-smoothing penalty
        signal_loss = 1 - (np.var(smoothed) / (np.var(original) + 1e-10))
        over_smooth_penalty = max(0, signal_loss - 0.3)  # Penalty if >30% signal loss
        
        # Combined score
        quality_score = (0.4 * noise_reduction + 
                        0.5 * correlation - 
                        0.1 * over_smooth_penalty)
        
        return max(0, quality_score)
    
    def select_best_algorithm(self, data: np.ndarray, frequency: np.ndarray, 
                            noise_metrics: Dict[str, float]) -> Tuple[str, Callable, Dict]:
        """
        Select the best smoothing algorithm based on data characteristics
        """
        noise_score = noise_metrics.get('combined_noise_score', 0.5)
        data_length = len(data)
        
        # Algorithm parameters based on noise level and data characteristics
        algorithm_params = {
            'moving_average': {'window_size': max(3, min(data_length // 5, int(5 + 10 * noise_score)))},
            'savitzky_golay': {
                'window_size': max(5, min(data_length // 3, int(7 + 8 * noise_score))),
                'poly_order': 2 if noise_score > 0.5 else 3
            },
            'gaussian': {'sigma': 0.5 + 2 * noise_score},
            'butterworth': {
                'frequency': frequency,
                'cutoff_ratio': 0.05 + 0.15 * noise_score,
                'order': 4
            },
            'median': {'window_size': max(3, min(data_length // 4, int(3 + 6 * noise_score)))},
            'wiener': {'noise_var': None}
        }
        
        # Test each algorithm
        best_algorithm = 'moving_average'
        best_score = -1
        best_params = algorithm_params['moving_average']
        
        for alg_name, alg_func in self.algorithms.items():
            try:
                params = algorithm_params[alg_name]
                smoothed = alg_func(data, **params)
                score = self.evaluate_smoothing_quality(data, smoothed, noise_score)
                
                if score > best_score:
                    best_score = score
                    best_algorithm = alg_name
                    best_params = params
                    
            except Exception as e:
                # Skip algorithms that fail
                continue
        
        return best_algorithm, self.algorithms[best_algorithm], best_params


class DielectricPreprocessor:
    """Main preprocessing class for dielectric permittivity data"""
    
    def __init__(self):
        self.noise_analyzer = NoiseAnalyzer()
        self.smoothing_selector = SmoothinSelector()
        self.preprocessing_history = []
    
    def preprocess(self, df: pd.DataFrame, apply_smoothing: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Main preprocessing function
        
        Args:
            df: DataFrame with columns [Frequency_GHz, Dk, Df]
            apply_smoothing: Whether to apply smoothing based on noise analysis
            
        Returns:
            Tuple of (preprocessed_df, processing_info)
        """
        if df.empty or len(df) < 3:
            return df.copy(), {'status': 'insufficient_data'}
        
        # Extract data
        frequency = df.iloc[:, 0].values
        dk = df.iloc[:, 1].values
        df_loss = df.iloc[:, 2].values
        
        # Analyze noise
        noise_metrics = self.noise_analyzer.analyze_noise(dk, df_loss, frequency)
        
        processing_info = {
            'original_length': len(df),
            'noise_metrics': noise_metrics,
            'smoothing_applied': False,
            'dk_algorithm': None,
            'df_algorithm': None,
            'recommendations': []
        }
        
        # Apply smoothing if needed and requested
        if apply_smoothing and noise_metrics['combined_noise_score'] > 0.3:
            # Select best smoothing for Dk
            dk_alg, dk_func, dk_params = self.smoothing_selector.select_best_algorithm(
                dk, frequency, noise_metrics
            )
            
            # Select best smoothing for Df
            df_alg, df_func, df_params = self.smoothing_selector.select_best_algorithm(
                df_loss, frequency, noise_metrics
            )
            
            # Apply smoothing
            dk_smoothed = dk_func(dk, **dk_params)
            df_smoothed = df_func(df_loss, **df_params)
            
            # Create smoothed dataframe
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
            if noise_metrics['combined_noise_score'] > 0.7:
                processing_info['recommendations'].append('High noise detected - consider data quality')
            if noise_metrics['dk_snr'] < 10:
                processing_info['recommendations'].append('Low SNR in Dk data')
            if noise_metrics['df_snr'] < 10:
                processing_info['recommendations'].append('Low SNR in Df data')
            
            return smoothed_df, processing_info
        
        else:
            if noise_metrics['combined_noise_score'] <= 0.3:
                processing_info['recommendations'].append('Data quality is good - no smoothing needed')
            
            return df.copy(), processing_info
    
    def get_preprocessing_summary(self) -> str:
        """Generate a human-readable summary of preprocessing results"""
        return f"Preprocessing completed. History contains {len(self.preprocessing_history)} operations."