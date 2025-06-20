# pattern_recognition.py - Advanced Chart Pattern Recognition V3.0 Ultimate
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports - use conditional imports for flexibility
try:
    from scipy.signal import find_peaks, argrelextrema
    from scipy.stats import linregress
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("SciPy not available. Using numpy alternatives.")
    SCIPY_AVAILABLE = False

try:
    import ta
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available. Using manual calculations.")
    TALIB_AVAILABLE = False

# =============================================================================
# 📊 DATA STRUCTURES
# =============================================================================

@dataclass
class PatternResult:
    """Chart pattern recognition result"""
    pattern_type: str
    confidence: float  # 0.0 to 1.0
    direction: str  # BULLISH, BEARISH, NEUTRAL
    strength: str  # WEAK, MODERATE, STRONG
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    pattern_points: Optional[Dict] = None
    description: str = ""

@dataclass
class SupportResistance:
    """Support and resistance levels"""
    supports: List[float]
    resistances: List[float]
    major_support: Optional[float] = None
    major_resistance: Optional[float] = None
    confidence_levels: Dict[float, float] = None

@dataclass
class FibonacciLevels:
    """Fibonacci retracement and extension levels"""
    swing_high: float
    swing_low: float
    retracement_levels: Dict[str, float]
    extension_levels: Dict[str, float]
    key_level: Optional[float] = None
    current_level: Optional[str] = None

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    direction: str  # UP, DOWN, SIDEWAYS
    strength: float  # 0.0 to 1.0
    duration: int  # bars
    angle: float  # degrees
    regression_line: Dict[str, float]
    trend_breaks: List[int] = None

# =============================================================================
# 🎯 MAIN PATTERN RECOGNITION CLASS
# =============================================================================

class AdvancedPatternRecognition:
    """
    Advanced Chart Pattern Recognition System
    
    Features:
    - Multiple chart patterns (Head & Shoulders, Triangles, Double Tops/Bottoms, etc.)
    - Fibonacci analysis
    - Support/Resistance detection
    - Trend analysis
    - Pattern confidence scoring
    """
    
    def __init__(self):
        """Initialize pattern recognition system"""
        self.fibonacci_ratios = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.272, 1.414, 1.618, 2.0, 2.618]
        }
        
        self.pattern_cache = {}
        self.analysis_cache = {}
        
    # =========================================================================
    # 📈 MAIN ANALYSIS FUNCTIONS
    # =========================================================================
    
    def analyze_patterns(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Comprehensive pattern analysis
        
        Args:
            data: OHLCV dataframe
            symbol: Trading symbol
            
        Returns:
            Complete pattern analysis results
        """
        try:
            logger.info(f"Starting pattern analysis for {symbol}")
            
            # Validate data
            if not self._validate_data(data):
                return self._get_empty_analysis()
            
            # Prepare data
            df = self._prepare_data(data.copy())
            
            # Perform all analyses
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(df),
                'chart_patterns': self._detect_chart_patterns(df),
                'support_resistance': self._find_support_resistance(df),
                'fibonacci_levels': self._calculate_fibonacci_levels(df),
                'trend_analysis': self._analyze_trend(df),
                'pattern_confluence': None,
                'overall_signal': None
            }
            
            # Calculate pattern confluence
            results['pattern_confluence'] = self._calculate_pattern_confluence(results)
            
            # Generate overall signal
            results['overall_signal'] = self._generate_overall_signal(results)
            
            logger.info(f"Pattern analysis completed for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error in pattern analysis for {symbol}: {e}")
            return self._get_empty_analysis()
    
    def _detect_chart_patterns(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Detect various chart patterns"""
        patterns = {}
        
        try:
            # Head and Shoulders patterns
            patterns.update(self._detect_head_shoulders(df))
            
            # Double Top/Bottom patterns
            patterns.update(self._detect_double_patterns(df))
            
            # Triangle patterns
            patterns.update(self._detect_triangles(df))
            
            # Flag and Pennant patterns
            patterns.update(self._detect_flags_pennants(df))
            
            # Wedge patterns
            patterns.update(self._detect_wedges(df))
            
            # Channel patterns
            patterns.update(self._detect_channels(df))
            
            logger.info(f"Detected {len(patterns)} chart patterns")
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {e}")
            
        return patterns
    
    # =========================================================================
    # 🏔️ HEAD AND SHOULDERS PATTERNS
    # =========================================================================
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Detect Head and Shoulders patterns"""
        patterns = {}
        
        try:
            # Find peaks and troughs
            highs = df['high'].values
            lows = df['low'].values
            
            # Use scipy if available, otherwise manual peak detection
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=10, prominence=np.std(highs) * 0.5)
                troughs, _ = find_peaks(-lows, distance=10, prominence=np.std(lows) * 0.5)
            else:
                peaks = self._find_peaks_manual(highs, window=10)
                troughs = self._find_peaks_manual(-lows, window=10)
            
            if len(peaks) >= 3:
                # Check for Head and Shoulders pattern
                hs_pattern = self._check_head_shoulders_pattern(df, peaks, highs)
                if hs_pattern:
                    patterns['head_shoulders'] = hs_pattern
            
            if len(troughs) >= 3:
                # Check for Inverse Head and Shoulders pattern
                ihs_pattern = self._check_inverse_head_shoulders_pattern(df, troughs, lows)
                if ihs_pattern:
                    patterns['inverse_head_shoulders'] = ihs_pattern
                    
        except Exception as e:
            logger.error(f"Error detecting Head and Shoulders patterns: {e}")
            
        return patterns
    
    def _check_head_shoulders_pattern(self, df: pd.DataFrame, peaks: np.ndarray, highs: np.ndarray) -> Optional[PatternResult]:
        """Check for Head and Shoulders pattern"""
        try:
            if len(peaks) < 3:
                return None
                
            # Take last 3 peaks
            recent_peaks = peaks[-3:]
            
            left_shoulder = highs[recent_peaks[0]]
            head = highs[recent_peaks[1]]
            right_shoulder = highs[recent_peaks[2]]
            
            # Head and Shoulders criteria
            head_higher = head > left_shoulder and head > right_shoulder
            shoulders_similar = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) < 0.05
            
            if head_higher and shoulders_similar:
                # Calculate neckline
                neckline = min(df.loc[recent_peaks[0]:recent_peaks[1], 'low'].min(),
                              df.loc[recent_peaks[1]:recent_peaks[2], 'low'].min())
                
                # Calculate targets
                head_to_neckline = head - neckline
                target_price = neckline - head_to_neckline
                
                # Calculate confidence
                confidence = self._calculate_hs_confidence(left_shoulder, head, right_shoulder, neckline)
                
                return PatternResult(
                    pattern_type="HEAD_AND_SHOULDERS",
                    confidence=confidence,
                    direction="BEARISH",
                    strength="STRONG" if confidence > 0.8 else "MODERATE" if confidence > 0.6 else "WEAK",
                    entry_price=neckline * 0.995,  # Slight below neckline
                    target_price=target_price,
                    stop_loss=head * 1.02,
                    risk_reward_ratio=abs(target_price - neckline) / abs(head - neckline),
                    pattern_points={
                        'left_shoulder': {'index': recent_peaks[0], 'price': left_shoulder},
                        'head': {'index': recent_peaks[1], 'price': head},
                        'right_shoulder': {'index': recent_peaks[2], 'price': right_shoulder},
                        'neckline': neckline
                    },
                    description="Bearish reversal pattern with head higher than shoulders"
                )
                
        except Exception as e:
            logger.error(f"Error checking Head and Shoulders pattern: {e}")
            
        return None
    
    def _check_inverse_head_shoulders_pattern(self, df: pd.DataFrame, troughs: np.ndarray, lows: np.ndarray) -> Optional[PatternResult]:
        """Check for Inverse Head and Shoulders pattern"""
        try:
            if len(troughs) < 3:
                return None
                
            # Take last 3 troughs
            recent_troughs = troughs[-3:]
            
            left_shoulder = lows[recent_troughs[0]]
            head = lows[recent_troughs[1]]
            right_shoulder = lows[recent_troughs[2]]
            
            # Inverse Head and Shoulders criteria
            head_lower = head < left_shoulder and head < right_shoulder
            shoulders_similar = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) < 0.05
            
            if head_lower and shoulders_similar:
                # Calculate neckline
                neckline = max(df.loc[recent_troughs[0]:recent_troughs[1], 'high'].max(),
                              df.loc[recent_troughs[1]:recent_troughs[2], 'high'].max())
                
                # Calculate targets
                neckline_to_head = neckline - head
                target_price = neckline + neckline_to_head
                
                # Calculate confidence
                confidence = self._calculate_hs_confidence(left_shoulder, head, right_shoulder, neckline)
                
                return PatternResult(
                    pattern_type="INVERSE_HEAD_AND_SHOULDERS",
                    confidence=confidence,
                    direction="BULLISH",
                    strength="STRONG" if confidence > 0.8 else "MODERATE" if confidence > 0.6 else "WEAK",
                    entry_price=neckline * 1.005,  # Slight above neckline
                    target_price=target_price,
                    stop_loss=head * 0.98,
                    risk_reward_ratio=abs(target_price - neckline) / abs(neckline - head),
                    pattern_points={
                        'left_shoulder': {'index': recent_troughs[0], 'price': left_shoulder},
                        'head': {'index': recent_troughs[1], 'price': head},
                        'right_shoulder': {'index': recent_troughs[2], 'price': right_shoulder},
                        'neckline': neckline
                    },
                    description="Bullish reversal pattern with head lower than shoulders"
                )
                
        except Exception as e:
            logger.error(f"Error checking Inverse Head and Shoulders pattern: {e}")
            
        return None
    
    # =========================================================================
    # 🔄 DOUBLE TOP/BOTTOM PATTERNS
    # =========================================================================
    
    def _detect_double_patterns(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Detect Double Top and Double Bottom patterns"""
        patterns = {}
        
        try:
            # Find peaks and troughs
            highs = df['high'].values
            lows = df['low'].values
            
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=20, prominence=np.std(highs) * 0.3)
                troughs, _ = find_peaks(-lows, distance=20, prominence=np.std(lows) * 0.3)
            else:
                peaks = self._find_peaks_manual(highs, window=20)
                troughs = self._find_peaks_manual(-lows, window=20)
            
            # Check for Double Top
            if len(peaks) >= 2:
                double_top = self._check_double_top_pattern(df, peaks, highs)
                if double_top:
                    patterns['double_top'] = double_top
            
            # Check for Double Bottom
            if len(troughs) >= 2:
                double_bottom = self._check_double_bottom_pattern(df, troughs, lows)
                if double_bottom:
                    patterns['double_bottom'] = double_bottom
                    
        except Exception as e:
            logger.error(f"Error detecting double patterns: {e}")
            
        return patterns
    
    def _check_double_top_pattern(self, df: pd.DataFrame, peaks: np.ndarray, highs: np.ndarray) -> Optional[PatternResult]:
        """Check for Double Top pattern"""
        try:
            if len(peaks) < 2:
                return None
                
            # Take last 2 peaks
            peak1_idx, peak2_idx = peaks[-2:]
            peak1_price = highs[peak1_idx]
            peak2_price = highs[peak2_idx]
            
            # Double top criteria
            price_similarity = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price) < 0.02
            sufficient_separation = abs(peak2_idx - peak1_idx) > 10
            
            if price_similarity and sufficient_separation:
                # Find valley between peaks
                valley_idx = peak1_idx + np.argmin(df.loc[peak1_idx:peak2_idx, 'low'].values)
                valley_price = df.loc[valley_idx, 'low']
                
                # Calculate targets
                peak_to_valley = max(peak1_price, peak2_price) - valley_price
                target_price = valley_price - peak_to_valley
                
                # Calculate confidence
                volume_confirmation = self._check_volume_confirmation(df, [peak1_idx, peak2_idx], pattern_type='double_top')
                depth_ratio = peak_to_valley / max(peak1_price, peak2_price)
                confidence = min(0.9, 0.4 + depth_ratio * 0.3 + volume_confirmation * 0.2)
                
                return PatternResult(
                    pattern_type="DOUBLE_TOP",
                    confidence=confidence,
                    direction="BEARISH",
                    strength="STRONG" if confidence > 0.7 else "MODERATE" if confidence > 0.5 else "WEAK",
                    entry_price=valley_price * 0.995,
                    target_price=target_price,
                    stop_loss=max(peak1_price, peak2_price) * 1.02,
                    risk_reward_ratio=abs(target_price - valley_price) / abs(max(peak1_price, peak2_price) - valley_price),
                    pattern_points={
                        'peak1': {'index': peak1_idx, 'price': peak1_price},
                        'peak2': {'index': peak2_idx, 'price': peak2_price},
                        'valley': {'index': valley_idx, 'price': valley_price}
                    },
                    description="Bearish reversal pattern with two similar peaks"
                )
                
        except Exception as e:
            logger.error(f"Error checking Double Top pattern: {e}")
            
        return None
    
    def _check_double_bottom_pattern(self, df: pd.DataFrame, troughs: np.ndarray, lows: np.ndarray) -> Optional[PatternResult]:
        """Check for Double Bottom pattern"""
        try:
            if len(troughs) < 2:
                return None
                
            # Take last 2 troughs
            trough1_idx, trough2_idx = troughs[-2:]
            trough1_price = lows[trough1_idx]
            trough2_price = lows[trough2_idx]
            
            # Double bottom criteria
            price_similarity = abs(trough1_price - trough2_price) / max(trough1_price, trough2_price) < 0.02
            sufficient_separation = abs(trough2_idx - trough1_idx) > 10
            
            if price_similarity and sufficient_separation:
                # Find peak between troughs
                peak_idx = trough1_idx + np.argmax(df.loc[trough1_idx:trough2_idx, 'high'].values)
                peak_price = df.loc[peak_idx, 'high']
                
                # Calculate targets
                peak_to_trough = peak_price - min(trough1_price, trough2_price)
                target_price = peak_price + peak_to_trough
                
                # Calculate confidence
                volume_confirmation = self._check_volume_confirmation(df, [trough1_idx, trough2_idx], pattern_type='double_bottom')
                depth_ratio = peak_to_trough / peak_price
                confidence = min(0.9, 0.4 + depth_ratio * 0.3 + volume_confirmation * 0.2)
                
                return PatternResult(
                    pattern_type="DOUBLE_BOTTOM",
                    confidence=confidence,
                    direction="BULLISH",
                    strength="STRONG" if confidence > 0.7 else "MODERATE" if confidence > 0.5 else "WEAK",
                    entry_price=peak_price * 1.005,
                    target_price=target_price,
                    stop_loss=min(trough1_price, trough2_price) * 0.98,
                    risk_reward_ratio=abs(target_price - peak_price) / abs(peak_price - min(trough1_price, trough2_price)),
                    pattern_points={
                        'trough1': {'index': trough1_idx, 'price': trough1_price},
                        'trough2': {'index': trough2_idx, 'price': trough2_price},
                        'peak': {'index': peak_idx, 'price': peak_price}
                    },
                    description="Bullish reversal pattern with two similar troughs"
                )
                
        except Exception as e:
            logger.error(f"Error checking Double Bottom pattern: {e}")
            
        return None
    
    # =========================================================================
    # 🔺 TRIANGLE PATTERNS
    # =========================================================================
    
    def _detect_triangles(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Detect Triangle patterns (Ascending, Descending, Symmetrical)"""
        patterns = {}
        
        try:
            # Get recent data for triangle analysis (last 50 bars)
            recent_data = df.tail(50) if len(df) > 50 else df
            
            # Detect different triangle types
            ascending = self._detect_ascending_triangle(recent_data)
            if ascending:
                patterns['ascending_triangle'] = ascending
                
            descending = self._detect_descending_triangle(recent_data)
            if descending:
                patterns['descending_triangle'] = descending
                
            symmetrical = self._detect_symmetrical_triangle(recent_data)
            if symmetrical:
                patterns['symmetrical_triangle'] = symmetrical
                
        except Exception as e:
            logger.error(f"Error detecting triangle patterns: {e}")
            
        return patterns
    
    def _detect_ascending_triangle(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Ascending Triangle pattern"""
        try:
            highs = df['high'].values
            lows = df['low'].values
            
            # Find resistance level (horizontal line at highs)
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=5)
            else:
                peaks = self._find_peaks_manual(highs, window=5)
                
            if len(peaks) < 2:
                return None
            
            # Check if highs are forming horizontal resistance
            recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
            peak_prices = highs[recent_peaks]
            resistance_level = np.mean(peak_prices)
            
            # Check if resistance is horizontal (low variance)
            if np.std(peak_prices) / resistance_level > 0.02:
                return None
            
            # Check if lows are ascending
            if SCIPY_AVAILABLE:
                troughs, _ = find_peaks(-lows, distance=5)
            else:
                troughs = self._find_peaks_manual(-lows, window=5)
                
            if len(troughs) < 2:
                return None
                
            recent_troughs = troughs[-3:] if len(troughs) >= 3 else troughs
            trough_prices = lows[recent_troughs]
            
            # Check ascending trend in lows
            if len(recent_troughs) >= 2:
                slope, _, r_value, _, _ = linregress(recent_troughs, trough_prices) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                
                if slope > 0 and r_value > 0.7:  # Ascending with good correlation
                    # Calculate target
                    triangle_height = resistance_level - np.min(trough_prices)
                    target_price = resistance_level + triangle_height
                    
                    # Calculate confidence
                    confidence = min(0.85, 0.5 + r_value * 0.2 + (1 - np.std(peak_prices)/resistance_level) * 0.15)
                    
                    return PatternResult(
                        pattern_type="ASCENDING_TRIANGLE",
                        confidence=confidence,
                        direction="BULLISH",
                        strength="STRONG" if confidence > 0.7 else "MODERATE",
                        entry_price=resistance_level * 1.002,
                        target_price=target_price,
                        stop_loss=np.max(trough_prices) * 0.98,
                        risk_reward_ratio=triangle_height / (resistance_level - np.max(trough_prices)),
                        pattern_points={
                            'resistance_level': resistance_level,
                            'support_line_slope': slope,
                            'triangle_height': triangle_height
                        },
                        description="Bullish continuation pattern with horizontal resistance and ascending support"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting ascending triangle: {e}")
            
        return None
    
    def _detect_descending_triangle(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Descending Triangle pattern"""
        try:
            highs = df['high'].values
            lows = df['low'].values
            
            # Find support level (horizontal line at lows)
            if SCIPY_AVAILABLE:
                troughs, _ = find_peaks(-lows, distance=5)
            else:
                troughs = self._find_peaks_manual(-lows, window=5)
                
            if len(troughs) < 2:
                return None
            
            # Check if lows are forming horizontal support
            recent_troughs = troughs[-3:] if len(troughs) >= 3 else troughs
            trough_prices = lows[recent_troughs]
            support_level = np.mean(trough_prices)
            
            # Check if support is horizontal (low variance)
            if np.std(trough_prices) / support_level > 0.02:
                return None
            
            # Check if highs are descending
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=5)
            else:
                peaks = self._find_peaks_manual(highs, window=5)
                
            if len(peaks) < 2:
                return None
                
            recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
            peak_prices = highs[recent_peaks]
            
            # Check descending trend in highs
            if len(recent_peaks) >= 2:
                slope, _, r_value, _, _ = linregress(recent_peaks, peak_prices) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                
                if slope < 0 and r_value < -0.7:  # Descending with good correlation
                    # Calculate target
                    triangle_height = np.max(peak_prices) - support_level
                    target_price = support_level - triangle_height
                    
                    # Calculate confidence
                    confidence = min(0.85, 0.5 + abs(r_value) * 0.2 + (1 - np.std(trough_prices)/support_level) * 0.15)
                    
                    return PatternResult(
                        pattern_type="DESCENDING_TRIANGLE",
                        confidence=confidence,
                        direction="BEARISH",
                        strength="STRONG" if confidence > 0.7 else "MODERATE",
                        entry_price=support_level * 0.998,
                        target_price=target_price,
                        stop_loss=np.min(peak_prices) * 1.02,
                        risk_reward_ratio=triangle_height / (np.min(peak_prices) - support_level),
                        pattern_points={
                            'support_level': support_level,
                            'resistance_line_slope': slope,
                            'triangle_height': triangle_height
                        },
                        description="Bearish continuation pattern with horizontal support and descending resistance"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting descending triangle: {e}")
            
        return None
    
    def _detect_symmetrical_triangle(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Symmetrical Triangle pattern"""
        try:
            highs = df['high'].values
            lows = df['low'].values
            
            # Find peaks and troughs
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=5)
                troughs, _ = find_peaks(-lows, distance=5)
            else:
                peaks = self._find_peaks_manual(highs, window=5)
                troughs = self._find_peaks_manual(-lows, window=5)
                
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
            recent_troughs = troughs[-3:] if len(troughs) >= 3 else troughs
            
            peak_prices = highs[recent_peaks]
            trough_prices = lows[recent_troughs]
            
            # Check for converging lines
            if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
                high_slope, _, high_r, _, _ = linregress(recent_peaks, peak_prices) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                low_slope, _, low_r, _, _ = linregress(recent_troughs, trough_prices) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                
                # Symmetrical triangle: descending highs and ascending lows
                if (high_slope < 0 and low_slope > 0 and 
                    high_r < -0.6 and low_r > 0.6):
                    
                    # Calculate breakout target (triangle height)
                    triangle_height = np.max(peak_prices) - np.min(trough_prices)
                    current_price = df['close'].iloc[-1]
                    
                    # Determine likely breakout direction based on trend context
                    trend_direction = self._determine_trend_context(df)
                    
                    if trend_direction == "UP":
                        target_price = current_price + triangle_height
                        direction = "BULLISH"
                        stop_loss = np.min(trough_prices) * 0.98
                    else:
                        target_price = current_price - triangle_height
                        direction = "BEARISH"
                        stop_loss = np.max(peak_prices) * 1.02
                    
                    # Calculate confidence
                    line_quality = (abs(high_r) + abs(low_r)) / 2
                    confidence = min(0.8, 0.4 + line_quality * 0.4)
                    
                    return PatternResult(
                        pattern_type="SYMMETRICAL_TRIANGLE",
                        confidence=confidence,
                        direction=direction,
                        strength="MODERATE",
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        risk_reward_ratio=triangle_height / abs(current_price - stop_loss),
                        pattern_points={
                            'high_slope': high_slope,
                            'low_slope': low_slope,
                            'triangle_height': triangle_height,
                            'convergence_point': (recent_peaks[-1] + recent_troughs[-1]) / 2
                        },
                        description="Neutral pattern with converging trendlines - breakout direction determines trend"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting symmetrical triangle: {e}")
            
        return None
    
    # =========================================================================
    # 🚩 FLAG AND PENNANT PATTERNS
    # =========================================================================
    
    def _detect_flags_pennants(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Detect Flag and Pennant patterns"""
        patterns = {}
        
        try:
            # Flags and pennants are short-term continuation patterns
            # Look for strong move followed by consolidation
            
            bull_flag = self._detect_bull_flag(df)
            if bull_flag:
                patterns['bull_flag'] = bull_flag
                
            bear_flag = self._detect_bear_flag(df)
            if bear_flag:
                patterns['bear_flag'] = bear_flag
                
            pennant = self._detect_pennant(df)
            if pennant:
                patterns['pennant'] = pennant
                
        except Exception as e:
            logger.error(f"Error detecting flag/pennant patterns: {e}")
            
        return patterns
    
    def _detect_bull_flag(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Bull Flag pattern"""
        try:
            if len(df) < 20:
                return None
                
            # Look for strong upward move followed by slight downward consolidation
            recent_data = df.tail(20)
            
            # Check for initial strong move
            price_change = (recent_data['close'].iloc[-10] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            if price_change < 0.05:  # Need at least 5% move
                return None
            
            # Check for consolidation (flag part)
            flag_data = recent_data.tail(10)
            flag_slope, _, flag_r, _, _ = linregress(range(len(flag_data)), flag_data['close'].values) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
            
            # Flag should have slight downward slope
            if flag_slope > 0 or flag_r < -0.5:
                return None
            
            # Calculate target
            flag_height = recent_data['high'].max() - recent_data['low'].min()
            target_price = recent_data['close'].iloc[-1] + flag_height
            
            confidence = min(0.8, 0.5 + abs(flag_r) * 0.2 + min(price_change, 0.2) * 1.5)
            
            return PatternResult(
                pattern_type="BULL_FLAG",
                confidence=confidence,
                direction="BULLISH",
                strength="STRONG" if confidence > 0.7 else "MODERATE",
                entry_price=recent_data['high'].iloc[-5:].max() * 1.002,
                target_price=target_price,
                stop_loss=recent_data['low'].iloc[-5:].min() * 0.98,
                risk_reward_ratio=flag_height / (recent_data['high'].iloc[-5:].max() - recent_data['low'].iloc[-5:].min()),
                pattern_points={
                    'pole_height': flag_height,
                    'flag_slope': flag_slope,
                    'breakout_level': recent_data['high'].iloc[-5:].max()
                },
                description="Bullish continuation pattern after strong upward move"
            )
            
        except Exception as e:
            logger.error(f"Error detecting bull flag: {e}")
            
        return None
    
    def _detect_bear_flag(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Bear Flag pattern"""
        try:
            if len(df) < 20:
                return None
                
            # Look for strong downward move followed by slight upward consolidation
            recent_data = df.tail(20)
            
            # Check for initial strong move down
            price_change = (recent_data['close'].iloc[0] - recent_data['close'].iloc[-10]) / recent_data['close'].iloc[0]
            
            if price_change < 0.05:  # Need at least 5% drop
                return None
            
            # Check for consolidation (flag part)
            flag_data = recent_data.tail(10)
            flag_slope, _, flag_r, _, _ = linregress(range(len(flag_data)), flag_data['close'].values) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
            
            # Flag should have slight upward slope
            if flag_slope < 0 or flag_r < 0.5:
                return None
            
            # Calculate target
            flag_height = recent_data['high'].max() - recent_data['low'].min()
            target_price = recent_data['close'].iloc[-1] - flag_height
            
            confidence = min(0.8, 0.5 + flag_r * 0.2 + min(price_change, 0.2) * 1.5)
            
            return PatternResult(
                pattern_type="BEAR_FLAG",
                confidence=confidence,
                direction="BEARISH",
                strength="STRONG" if confidence > 0.7 else "MODERATE",
                entry_price=recent_data['low'].iloc[-5:].min() * 0.998,
                target_price=target_price,
                stop_loss=recent_data['high'].iloc[-5:].max() * 1.02,
                risk_reward_ratio=flag_height / (recent_data['high'].iloc[-5:].max() - recent_data['low'].iloc[-5:].min()),
                pattern_points={
                    'pole_height': flag_height,
                    'flag_slope': flag_slope,
                    'breakdown_level': recent_data['low'].iloc[-5:].min()
                },
                description="Bearish continuation pattern after strong downward move"
            )
            
        except Exception as e:
            logger.error(f"Error detecting bear flag: {e}")
            
        return None
    
    def _detect_pennant(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Pennant pattern"""
        try:
            if len(df) < 15:
                return None
                
            # Pennant is like a small symmetrical triangle after strong move
            recent_data = df.tail(15)
            
            # Check for initial strong move
            initial_move = abs(recent_data['close'].iloc[-10] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            if initial_move < 0.03:  # Need at least 3% move
                return None
            
            # Check pennant consolidation
            pennant_data = recent_data.tail(8)
            highs = pennant_data['high'].values
            lows = pennant_data['low'].values
            
            if SCIPY_AVAILABLE:
                high_slope, _, high_r, _, _ = linregress(range(len(pennant_data)), highs)
                low_slope, _, low_r, _, _ = linregress(range(len(pennant_data)), lows)
            else:
                high_slope = low_slope = high_r = low_r = 0
            
            # Pennant: converging price action
            if abs(high_slope) < abs(low_slope) * 0.5:  # Converging lines
                pennant_height = recent_data['high'].max() - recent_data['low'].min()
                direction = "BULLISH" if recent_data['close'].iloc[-10] > recent_data['close'].iloc[0] else "BEARISH"
                
                if direction == "BULLISH":
                    target_price = recent_data['close'].iloc[-1] + pennant_height
                    stop_loss = recent_data['low'].iloc[-5:].min() * 0.98
                else:
                    target_price = recent_data['close'].iloc[-1] - pennant_height
                    stop_loss = recent_data['high'].iloc[-5:].max() * 1.02
                
                confidence = min(0.75, 0.4 + initial_move * 5 + abs(high_r + low_r) * 0.15)
                
                return PatternResult(
                    pattern_type="PENNANT",
                    confidence=confidence,
                    direction=direction,
                    strength="MODERATE",
                    entry_price=recent_data['close'].iloc[-1],
                    target_price=target_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=pennant_height / abs(recent_data['close'].iloc[-1] - stop_loss),
                    pattern_points={
                        'pennant_height': pennant_height,
                        'initial_move': initial_move,
                        'convergence_quality': abs(high_r + low_r)
                    },
                    description=f"{direction.lower().title()} continuation pattern with converging consolidation"
                )
                
        except Exception as e:
            logger.error(f"Error detecting pennant: {e}")
            
        return None
    
    # =========================================================================
    # 📐 WEDGE PATTERNS
    # =========================================================================
    
    def _detect_wedges(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Detect Wedge patterns (Rising and Falling)"""
        patterns = {}
        
        try:
            rising_wedge = self._detect_rising_wedge(df)
            if rising_wedge:
                patterns['rising_wedge'] = rising_wedge
                
            falling_wedge = self._detect_falling_wedge(df)
            if falling_wedge:
                patterns['falling_wedge'] = falling_wedge
                
        except Exception as e:
            logger.error(f"Error detecting wedge patterns: {e}")
            
        return patterns
    
    def _detect_rising_wedge(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Rising Wedge pattern"""
        try:
            if len(df) < 25:
                return None
                
            recent_data = df.tail(25)
            
            # Find peaks and troughs
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=3)
                troughs, _ = find_peaks(-lows, distance=3)
            else:
                peaks = self._find_peaks_manual(highs, window=3)
                troughs = self._find_peaks_manual(-lows, window=3)
            
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            # Both support and resistance should be rising, but resistance rising slower
            if len(peaks) >= 2 and len(troughs) >= 2:
                peak_slope, _, peak_r, _, _ = linregress(peaks, highs[peaks]) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                trough_slope, _, trough_r, _, _ = linregress(troughs, lows[troughs]) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                
                # Rising wedge: both lines rising, but support rises faster (converging upward)
                if (peak_slope > 0 and trough_slope > 0 and 
                    trough_slope > peak_slope and 
                    peak_r > 0.6 and trough_r > 0.6):
                    
                    wedge_height = np.max(highs) - np.min(lows)
                    target_price = recent_data['close'].iloc[-1] - wedge_height * 0.618  # Often retraces 61.8%
                    
                    confidence = min(0.8, 0.4 + (peak_r + trough_r) * 0.2)
                    
                    return PatternResult(
                        pattern_type="RISING_WEDGE",
                        confidence=confidence,
                        direction="BEARISH",
                        strength="STRONG" if confidence > 0.7 else "MODERATE",
                        entry_price=recent_data['low'].iloc[-5:].min() * 0.998,
                        target_price=target_price,
                        stop_loss=recent_data['high'].iloc[-5:].max() * 1.02,
                        risk_reward_ratio=(recent_data['close'].iloc[-1] - target_price) / (recent_data['high'].iloc[-5:].max() - recent_data['close'].iloc[-1]),
                        pattern_points={
                            'resistance_slope': peak_slope,
                            'support_slope': trough_slope,
                            'wedge_height': wedge_height,
                            'convergence_ratio': trough_slope / peak_slope
                        },
                        description="Bearish reversal pattern with converging upward trendlines"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting rising wedge: {e}")
            
        return None
    
    def _detect_falling_wedge(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Falling Wedge pattern"""
        try:
            if len(df) < 25:
                return None
                
            recent_data = df.tail(25)
            
            # Find peaks and troughs
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=3)
                troughs, _ = find_peaks(-lows, distance=3)
            else:
                peaks = self._find_peaks_manual(highs, window=3)
                troughs = self._find_peaks_manual(-lows, window=3)
            
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            # Both support and resistance should be falling, but support falling slower
            if len(peaks) >= 2 and len(troughs) >= 2:
                peak_slope, _, peak_r, _, _ = linregress(peaks, highs[peaks]) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                trough_slope, _, trough_r, _, _ = linregress(troughs, lows[troughs]) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                
                # Falling wedge: both lines falling, but resistance falls faster (converging downward)
                if (peak_slope < 0 and trough_slope < 0 and 
                    peak_slope < trough_slope and 
                    peak_r < -0.6 and trough_r < -0.6):
                    
                    wedge_height = np.max(highs) - np.min(lows)
                    target_price = recent_data['close'].iloc[-1] + wedge_height * 0.618  # Often extends 61.8%
                    
                    confidence = min(0.8, 0.4 + abs(peak_r + trough_r) * 0.2)
                    
                    return PatternResult(
                        pattern_type="FALLING_WEDGE",
                        confidence=confidence,
                        direction="BULLISH",
                        strength="STRONG" if confidence > 0.7 else "MODERATE",
                        entry_price=recent_data['high'].iloc[-5:].max() * 1.002,
                        target_price=target_price,
                        stop_loss=recent_data['low'].iloc[-5:].min() * 0.98,
                        risk_reward_ratio=(target_price - recent_data['close'].iloc[-1]) / (recent_data['close'].iloc[-1] - recent_data['low'].iloc[-5:].min()),
                        pattern_points={
                            'resistance_slope': peak_slope,
                            'support_slope': trough_slope,
                            'wedge_height': wedge_height,
                            'convergence_ratio': trough_slope / peak_slope
                        },
                        description="Bullish reversal pattern with converging downward trendlines"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting falling wedge: {e}")
            
        return None
    
    # =========================================================================
    # 📊 CHANNEL PATTERNS
    # =========================================================================
    
    def _detect_channels(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Detect Channel patterns"""
        patterns = {}
        
        try:
            # Detect different channel types
            ascending_channel = self._detect_ascending_channel(df)
            if ascending_channel:
                patterns['ascending_channel'] = ascending_channel
                
            descending_channel = self._detect_descending_channel(df)
            if descending_channel:
                patterns['descending_channel'] = descending_channel
                
            horizontal_channel = self._detect_horizontal_channel(df)
            if horizontal_channel:
                patterns['horizontal_channel'] = horizontal_channel
                
        except Exception as e:
            logger.error(f"Error detecting channel patterns: {e}")
            
        return patterns
    
    def _detect_ascending_channel(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Ascending Channel pattern"""
        try:
            if len(df) < 30:
                return None
                
            recent_data = df.tail(30)
            
            # Find peaks and troughs
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=5)
                troughs, _ = find_peaks(-lows, distance=5)
            else:
                peaks = self._find_peaks_manual(highs, window=5)
                troughs = self._find_peaks_manual(-lows, window=5)
            
            if len(peaks) < 3 or len(troughs) < 3:
                return None
            
            # Check for parallel ascending lines
            if len(peaks) >= 3 and len(troughs) >= 3:
                peak_slope, _, peak_r, _, _ = linregress(peaks, highs[peaks]) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                trough_slope, _, trough_r, _, _ = linregress(troughs, lows[troughs]) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                
                # Both lines should be ascending and roughly parallel
                slope_diff = abs(peak_slope - trough_slope) / max(abs(peak_slope), abs(trough_slope))
                
                if (peak_slope > 0 and trough_slope > 0 and 
                    peak_r > 0.7 and trough_r > 0.7 and 
                    slope_diff < 0.3):  # Parallel lines
                    
                    channel_width = np.mean(highs[peaks]) - np.mean(lows[troughs])
                    current_position = (recent_data['close'].iloc[-1] - np.mean(lows[troughs])) / channel_width
                    
                    # Determine trade direction based on position in channel
                    if current_position < 0.3:  # Near support
                        direction = "BULLISH"
                        entry_price = recent_data['close'].iloc[-1] * 1.001
                        target_price = np.mean(highs[peaks])
                        stop_loss = np.min(lows[troughs]) * 0.98
                    elif current_position > 0.7:  # Near resistance
                        direction = "BEARISH"
                        entry_price = recent_data['close'].iloc[-1] * 0.999
                        target_price = np.mean(lows[troughs])
                        stop_loss = np.max(highs[peaks]) * 1.02
                    else:
                        direction = "NEUTRAL"
                        entry_price = recent_data['close'].iloc[-1]
                        target_price = entry_price
                        stop_loss = entry_price
                    
                    confidence = min(0.8, 0.5 + (peak_r + trough_r) * 0.15)
                    
                    return PatternResult(
                        pattern_type="ASCENDING_CHANNEL",
                        confidence=confidence,
                        direction=direction,
                        strength="MODERATE",
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        risk_reward_ratio=abs(target_price - entry_price) / abs(entry_price - stop_loss) if stop_loss != entry_price else 1,
                        pattern_points={
                            'upper_channel_slope': peak_slope,
                            'lower_channel_slope': trough_slope,
                            'channel_width': channel_width,
                            'current_position': current_position
                        },
                        description=f"Ascending channel - price at {current_position:.1%} of channel width"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting ascending channel: {e}")
            
        return None
    
    def _detect_descending_channel(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Descending Channel pattern"""
        try:
            if len(df) < 30:
                return None
                
            recent_data = df.tail(30)
            
            # Find peaks and troughs
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=5)
                troughs, _ = find_peaks(-lows, distance=5)
            else:
                peaks = self._find_peaks_manual(highs, window=5)
                troughs = self._find_peaks_manual(-lows, window=5)
            
            if len(peaks) < 3 or len(troughs) < 3:
                return None
            
            # Check for parallel descending lines
            if len(peaks) >= 3 and len(troughs) >= 3:
                peak_slope, _, peak_r, _, _ = linregress(peaks, highs[peaks]) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                trough_slope, _, trough_r, _, _ = linregress(troughs, lows[troughs]) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
                
                # Both lines should be descending and roughly parallel
                slope_diff = abs(peak_slope - trough_slope) / max(abs(peak_slope), abs(trough_slope))
                
                if (peak_slope < 0 and trough_slope < 0 and 
                    peak_r < -0.7 and trough_r < -0.7 and 
                    slope_diff < 0.3):  # Parallel lines
                    
                    channel_width = np.mean(highs[peaks]) - np.mean(lows[troughs])
                    current_position = (recent_data['close'].iloc[-1] - np.mean(lows[troughs])) / channel_width
                    
                    # Determine trade direction based on position in channel
                    if current_position < 0.3:  # Near support
                        direction = "BULLISH"
                        entry_price = recent_data['close'].iloc[-1] * 1.001
                        target_price = np.mean(highs[peaks])
                        stop_loss = np.min(lows[troughs]) * 0.98
                    elif current_position > 0.7:  # Near resistance
                        direction = "BEARISH"
                        entry_price = recent_data['close'].iloc[-1] * 0.999
                        target_price = np.mean(lows[troughs])
                        stop_loss = np.max(highs[peaks]) * 1.02
                    else:
                        direction = "NEUTRAL"
                        entry_price = recent_data['close'].iloc[-1]
                        target_price = entry_price
                        stop_loss = entry_price
                    
                    confidence = min(0.8, 0.5 + abs(peak_r + trough_r) * 0.15)
                    
                    return PatternResult(
                        pattern_type="DESCENDING_CHANNEL",
                        confidence=confidence,
                        direction=direction,
                        strength="MODERATE",
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        risk_reward_ratio=abs(target_price - entry_price) / abs(entry_price - stop_loss) if stop_loss != entry_price else 1,
                        pattern_points={
                            'upper_channel_slope': peak_slope,
                            'lower_channel_slope': trough_slope,
                            'channel_width': channel_width,
                            'current_position': current_position
                        },
                        description=f"Descending channel - price at {current_position:.1%} of channel width"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting descending channel: {e}")
            
        return None
    
    def _detect_horizontal_channel(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect Horizontal Channel (Trading Range) pattern"""
        try:
            if len(df) < 25:
                return None
                
            recent_data = df.tail(25)
            
            # Find peaks and troughs
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=4)
                troughs, _ = find_peaks(-lows, distance=4)
            else:
                peaks = self._find_peaks_manual(highs, window=4)
                troughs = self._find_peaks_manual(-lows, window=4)
            
            if len(peaks) < 3 or len(troughs) < 3:
                return None
            
            # Check for horizontal lines (low slopes)
            if len(peaks) >= 3 and len(troughs) >= 3:
                peak_prices = highs[peaks]
                trough_prices = lows[troughs]
                
                # Check if peaks and troughs are relatively horizontal
                peak_variance = np.std(peak_prices) / np.mean(peak_prices)
                trough_variance = np.std(trough_prices) / np.mean(trough_prices)
                
                if peak_variance < 0.02 and trough_variance < 0.02:  # Low variance = horizontal
                    resistance_level = np.mean(peak_prices)
                    support_level = np.mean(trough_prices)
                    channel_width = resistance_level - support_level
                    
                    current_position = (recent_data['close'].iloc[-1] - support_level) / channel_width
                    
                    # Determine trade direction based on position in channel
                    if current_position < 0.25:  # Near support
                        direction = "BULLISH"
                        entry_price = support_level * 1.002
                        target_price = resistance_level * 0.995
                        stop_loss = support_level * 0.985
                    elif current_position > 0.75:  # Near resistance
                        direction = "BEARISH"
                        entry_price = resistance_level * 0.998
                        target_price = support_level * 1.005
                        stop_loss = resistance_level * 1.015
                    else:
                        direction = "NEUTRAL"
                        entry_price = recent_data['close'].iloc[-1]
                        target_price = entry_price
                        stop_loss = entry_price
                    
                    # Confidence based on how well defined the levels are
                    level_touches = len(peaks) + len(troughs)
                    confidence = min(0.85, 0.4 + level_touches * 0.06 + (1 - peak_variance - trough_variance) * 0.2)
                    
                    return PatternResult(
                        pattern_type="HORIZONTAL_CHANNEL",
                        confidence=confidence,
                        direction=direction,
                        strength="MODERATE",
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        risk_reward_ratio=abs(target_price - entry_price) / abs(entry_price - stop_loss) if stop_loss != entry_price else 1,
                        pattern_points={
                            'resistance_level': resistance_level,
                            'support_level': support_level,
                            'channel_width': channel_width,
                            'current_position': current_position,
                            'level_touches': level_touches
                        },
                        description=f"Horizontal trading range - price at {current_position:.1%} of range"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting horizontal channel: {e}")
            
        return None
    
    # =========================================================================
    # 📐 FIBONACCI ANALYSIS
    # =========================================================================
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> FibonacciLevels:
        """Calculate Fibonacci retracement and extension levels"""
        try:
            # Find significant swing high and low
            recent_data = df.tail(100) if len(df) > 100 else df
            
            swing_high_idx = recent_data['high'].idxmax()
            swing_low_idx = recent_data['low'].idxmin()
            
            swing_high = recent_data.loc[swing_high_idx, 'high']
            swing_low = recent_data.loc[swing_low_idx, 'low']
            
            # Calculate retracement levels
            diff = swing_high - swing_low
            retracement_levels = {}
            
            for ratio in self.fibonacci_ratios['retracement']:
                if swing_high_idx > swing_low_idx:  # Uptrend
                    level = swing_high - (diff * ratio)
                    retracement_levels[f"{ratio:.1%}"] = level
                else:  # Downtrend
                    level = swing_low + (diff * ratio)
                    retracement_levels[f"{ratio:.1%}"] = level
            
            # Calculate extension levels
            extension_levels = {}
            for ratio in self.fibonacci_ratios['extension']:
                if swing_high_idx > swing_low_idx:  # Uptrend
                    level = swing_high + (diff * (ratio - 1))
                    extension_levels[f"{ratio:.1%}"] = level
                else:  # Downtrend
                    level = swing_low - (diff * (ratio - 1))
                    extension_levels[f"{ratio:.1%}"] = level
            
            # Find closest level to current price
            current_price = df['close'].iloc[-1]
            all_levels = {**retracement_levels, **extension_levels}
            
            closest_level = min(all_levels.keys(), 
                              key=lambda x: abs(all_levels[x] - current_price))
            key_level = all_levels[closest_level]
            
            return FibonacciLevels(
                swing_high=swing_high,
                swing_low=swing_low,
                retracement_levels=retracement_levels,
                extension_levels=extension_levels,
                key_level=key_level,
                current_level=closest_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return FibonacciLevels(
                swing_high=df['high'].max(),
                swing_low=df['low'].min(),
                retracement_levels={},
                extension_levels={}
            )
    
    # =========================================================================
    # 🎯 SUPPORT AND RESISTANCE
    # =========================================================================
    
    def _find_support_resistance(self, df: pd.DataFrame) -> SupportResistance:
        """Find support and resistance levels"""
        try:
            # Use recent data for support/resistance
            recent_data = df.tail(100) if len(df) > 100 else df
            
            # Find peaks and troughs
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.3)
                troughs, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.3)
            else:
                peaks = self._find_peaks_manual(highs, window=5)
                troughs = self._find_peaks_manual(-lows, window=5)
            
            # Get resistance levels from peaks
            resistance_levels = []
            if len(peaks) > 0:
                resistance_levels = sorted(highs[peaks], reverse=True)
            
            # Get support levels from troughs  
            support_levels = []
            if len(troughs) > 0:
                support_levels = sorted(lows[troughs])
            
            # Find major levels (levels tested multiple times)
            major_resistance = self._find_major_level(resistance_levels, recent_data['high'])
            major_support = self._find_major_level(support_levels, recent_data['low'])
            
            # Calculate confidence for each level
            confidence_levels = {}
            for level in resistance_levels[:5]:  # Top 5 resistance levels
                confidence = self._calculate_level_confidence(level, recent_data['high'])
                confidence_levels[level] = confidence
                
            for level in support_levels[:5]:  # Top 5 support levels
                confidence = self._calculate_level_confidence(level, recent_data['low'])
                confidence_levels[level] = confidence
            
            return SupportResistance(
                supports=support_levels[:5],  # Top 5 support levels
                resistances=resistance_levels[:5],  # Top 5 resistance levels
                major_support=major_support,
                major_resistance=major_resistance,
                confidence_levels=confidence_levels
            )
            
        except Exception as e:
            logger.error(f"Error finding support and resistance: {e}")
            return SupportResistance(
                supports=[],
                resistances=[],
                confidence_levels={}
            )
    
    def _find_major_level(self, levels: List[float], price_series: pd.Series) -> Optional[float]:
        """Find major support/resistance level (most tested)"""
        try:
            if not levels:
                return None
                
            level_tests = {}
            tolerance = 0.02  # 2% tolerance
            
            for level in levels:
                tests = 0
                for price in price_series:
                    if abs(price - level) / level <= tolerance:
                        tests += 1
                level_tests[level] = tests
            
            # Return level with most tests
            if level_tests:
                return max(level_tests.keys(), key=level_tests.get)
            
        except Exception as e:
            logger.error(f"Error finding major level: {e}")
            
        return None
    
    def _calculate_level_confidence(self, level: float, price_series: pd.Series) -> float:
        """Calculate confidence in support/resistance level"""
        try:
            tolerance = 0.015  # 1.5% tolerance
            tests = 0
            bounces = 0
            
            prices = price_series.values
            for i, price in enumerate(prices):
                if abs(price - level) / level <= tolerance:
                    tests += 1
                    # Check if price bounced from this level
                    if i < len(prices) - 2:
                        next_price = prices[i + 1]
                        if abs(next_price - level) / level > tolerance:
                            bounces += 1
            
            # Confidence based on tests and bounce rate
            if tests > 0:
                bounce_rate = bounces / tests
                confidence = min(0.9, 0.2 + tests * 0.1 + bounce_rate * 0.4)
                return confidence
            
        except Exception as e:
            logger.error(f"Error calculating level confidence: {e}")
            
        return 0.0
    
    # =========================================================================
    # 📈 TREND ANALYSIS
    # =========================================================================
    
    def _analyze_trend(self, df: pd.DataFrame) -> TrendAnalysis:
        """Analyze overall trend"""
        try:
            # Use different timeframes for trend analysis
            short_term = df.tail(20)
            medium_term = df.tail(50) if len(df) >= 50 else df
            long_term = df.tail(100) if len(df) >= 100 else df
            
            # Calculate trend for each timeframe
            short_trend = self._calculate_trend_direction(short_term)
            medium_trend = self._calculate_trend_direction(medium_term)
            long_trend = self._calculate_trend_direction(long_term)
            
            # Overall trend is weighted average
            trend_weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
            
            trend_scores = {
                'UP': (short_trend['UP'] * trend_weights['short'] +
                      medium_trend['UP'] * trend_weights['medium'] +
                      long_trend['UP'] * trend_weights['long']),
                'DOWN': (short_trend['DOWN'] * trend_weights['short'] +
                        medium_trend['DOWN'] * trend_weights['medium'] +
                        long_trend['DOWN'] * trend_weights['long']),
                'SIDEWAYS': (short_trend['SIDEWAYS'] * trend_weights['short'] +
                           medium_trend['SIDEWAYS'] * trend_weights['medium'] +
                           long_trend['SIDEWAYS'] * trend_weights['long'])
            }
            
            # Determine dominant trend
            dominant_trend = max(trend_scores.keys(), key=trend_scores.get)
            trend_strength = trend_scores[dominant_trend]
            
            # Calculate trend angle and regression line
            if SCIPY_AVAILABLE:
                x_values = range(len(medium_term))
                y_values = medium_term['close'].values
                slope, intercept, r_value, _, _ = linregress(x_values, y_values)
                
                # Convert slope to angle in degrees
                angle = np.degrees(np.arctan(slope / np.mean(y_values)))
                
                regression_line = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'start_price': intercept,
                    'end_price': slope * (len(medium_term) - 1) + intercept
                }
            else:
                angle = 0
                regression_line = {'slope': 0, 'intercept': 0, 'r_squared': 0}
            
            return TrendAnalysis(
                direction=dominant_trend,
                strength=trend_strength,
                duration=len(medium_term),
                angle=angle,
                regression_line=regression_line
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return TrendAnalysis(
                direction="SIDEWAYS",
                strength=0.5,
                duration=0,
                angle=0,
                regression_line={'slope': 0, 'intercept': 0, 'r_squared': 0}
            )
    
    def _calculate_trend_direction(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend direction for given timeframe"""
        try:
            if len(df) < 5:
                return {'UP': 0.33, 'DOWN': 0.33, 'SIDEWAYS': 0.34}
            
            # Calculate multiple trend indicators
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            
            # Moving average trend
            if len(df) >= 10:
                ma_short = df['close'].tail(5).mean()
                ma_long = df['close'].tail(10).mean() if len(df) >= 10 else df['close'].mean()
                ma_trend = (ma_short - ma_long) / ma_long
            else:
                ma_trend = price_change
            
            # Higher highs and higher lows for uptrend
            highs = df['high'].values
            lows = df['low'].values
            
            hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
            
            ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
            lh_count = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
            
            total_periods = len(df) - 1
            if total_periods > 0:
                uptrend_ratio = (hh_count + hl_count) / (total_periods * 2)
                downtrend_ratio = (ll_count + lh_count) / (total_periods * 2)
            else:
                uptrend_ratio = downtrend_ratio = 0.5
            
            # Combine indicators
            up_score = 0.4 * max(0, price_change * 10) + 0.3 * max(0, ma_trend * 10) + 0.3 * uptrend_ratio
            down_score = 0.4 * max(0, -price_change * 10) + 0.3 * max(0, -ma_trend * 10) + 0.3 * downtrend_ratio
            sideways_score = 1 - abs(price_change) * 5 - abs(ma_trend) * 5
            
            # Normalize scores
            total_score = up_score + down_score + max(0, sideways_score)
            if total_score > 0:
                return {
                    'UP': up_score / total_score,
                    'DOWN': down_score / total_score,
                    'SIDEWAYS': max(0, sideways_score) / total_score
                }
            
        except Exception as e:
            logger.error(f"Error calculating trend direction: {e}")
            
        return {'UP': 0.33, 'DOWN': 0.33, 'SIDEWAYS': 0.34}
    
    # =========================================================================
    # 🔗 PATTERN CONFLUENCE AND OVERALL SIGNAL
    # =========================================================================
    
    def _calculate_pattern_confluence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate pattern confluence score"""
        try:
            patterns = results.get('chart_patterns', {})
            fibonacci = results.get('fibonacci_levels')
            sr_levels = results.get('support_resistance')
            trend = results.get('trend_analysis')
            
            confluence_score = 0.0
            confluence_factors = []
            
            # Chart pattern confluence
            bullish_patterns = 0
            bearish_patterns = 0
            total_confidence = 0.0
            
            for pattern_name, pattern in patterns.items():
                if pattern.direction == "BULLISH":
                    bullish_patterns += 1
                    confluence_score += pattern.confidence * 0.3
                elif pattern.direction == "BEARISH":
                    bearish_patterns += 1
                    confluence_score -= pattern.confidence * 0.3
                
                total_confidence += pattern.confidence
                confluence_factors.append(f"{pattern_name}: {pattern.confidence:.2f}")
            
            # Fibonacci confluence
            if fibonacci and fibonacci.key_level:
                current_price = results.get('current_price', 0)
                if current_price and fibonacci.key_level:
                    fib_distance = abs(current_price - fibonacci.key_level) / current_price
                    if fib_distance < 0.02:  # Within 2% of key Fibonacci level
                        confluence_score += 0.2
                        confluence_factors.append(f"Fibonacci {fibonacci.current_level}: +0.2")
            
            # Support/Resistance confluence
            if sr_levels and sr_levels.confidence_levels:
                current_price = results.get('current_price', 0)
                for level, confidence in sr_levels.confidence_levels.items():
                    if current_price:
                        distance = abs(current_price - level) / current_price
                        if distance < 0.015:  # Within 1.5% of S/R level
                            confluence_score += confidence * 0.15
                            confluence_factors.append(f"S/R Level: +{confidence * 0.15:.2f}")
            
            # Trend confluence
            if trend:
                if trend.direction == "UP" and bullish_patterns > bearish_patterns:
                    confluence_score += trend.strength * 0.2
                    confluence_factors.append(f"Trend alignment: +{trend.strength * 0.2:.2f}")
                elif trend.direction == "DOWN" and bearish_patterns > bullish_patterns:
                    confluence_score += trend.strength * 0.2
                    confluence_factors.append(f"Trend alignment: +{trend.strength * 0.2:.2f}")
            
            # Normalize confluence score
            confluence_score = max(-1.0, min(1.0, confluence_score))
            
            return {
                'confluence_score': confluence_score,
                'bullish_patterns': bullish_patterns,
                'bearish_patterns': bearish_patterns,
                'total_confidence': total_confidence,
                'confluence_factors': confluence_factors,
                'strength': 'STRONG' if abs(confluence_score) > 0.7 else 'MODERATE' if abs(confluence_score) > 0.4 else 'WEAK'
            }
            
        except Exception as e:
            logger.error(f"Error calculating pattern confluence: {e}")
            return {
                'confluence_score': 0.0,
                'bullish_patterns': 0,
                'bearish_patterns': 0,
                'total_confidence': 0.0,
                'confluence_factors': [],
                'strength': 'WEAK'
            }
    
    def _generate_overall_signal(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall trading signal from all patterns"""
        try:
            confluence = results.get('pattern_confluence', {})
            patterns = results.get('chart_patterns', {})
            trend = results.get('trend_analysis')
            
            confluence_score = confluence.get('confluence_score', 0.0)
            bullish_patterns = confluence.get('bullish_patterns', 0)
            bearish_patterns = confluence.get('bearish_patterns', 0)
            
            # Determine signal direction
            if confluence_score > 0.3 and bullish_patterns > bearish_patterns:
                signal_direction = "BULLISH"
                signal_strength = min(confluence_score, 1.0)
            elif confluence_score < -0.3 and bearish_patterns > bullish_patterns:
                signal_direction = "BEARISH"
                signal_strength = min(abs(confluence_score), 1.0)
            else:
                signal_direction = "NEUTRAL"
                signal_strength = 0.5
            
            # Find best pattern for entry/exit levels
            best_pattern = None
            best_confidence = 0.0
            
            for pattern_name, pattern in patterns.items():
                if pattern.confidence > best_confidence and pattern.direction == signal_direction:
                    best_pattern = pattern
                    best_confidence = pattern.confidence
            
            # Calculate risk/reward
            risk_reward_ratio = None
            if best_pattern and best_pattern.entry_price and best_pattern.target_price and best_pattern.stop_loss:
                profit = abs(best_pattern.target_price - best_pattern.entry_price)
                risk = abs(best_pattern.entry_price - best_pattern.stop_loss)
                if risk > 0:
                    risk_reward_ratio = profit / risk
            
            return {
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'confidence': best_confidence,
                'entry_price': best_pattern.entry_price if best_pattern else None,
                'target_price': best_pattern.target_price if best_pattern else None,
                'stop_loss': best_pattern.stop_loss if best_pattern else None,
                'risk_reward_ratio': risk_reward_ratio,
                'best_pattern': best_pattern.pattern_type if best_pattern else None,
                'recommendation': self._get_recommendation(signal_direction, signal_strength, best_confidence),
                'supporting_factors': confluence.get('confluence_factors', [])
            }
            
        except Exception as e:
            logger.error(f"Error generating overall signal: {e}")
            return {
                'signal_direction': 'NEUTRAL',
                'signal_strength': 0.5,
                'confidence': 0.0,
                'recommendation': 'HOLD'
            }
    
    def _get_recommendation(self, direction: str, strength: float, confidence: float) -> str:
        """Get trading recommendation based on signal quality"""
        overall_score = (strength + confidence) / 2
        
        if direction == "BULLISH":
            if overall_score > 0.8:
                return "STRONG_BUY"
            elif overall_score > 0.6:
                return "BUY"
            elif overall_score > 0.4:
                return "WEAK_BUY"
        elif direction == "BEARISH":
            if overall_score > 0.8:
                return "STRONG_SELL"
            elif overall_score > 0.6:
                return "SELL"
            elif overall_score > 0.4:
                return "WEAK_SELL"
        
        return "HOLD"
    
    # =========================================================================
    # 🛠️ HELPER FUNCTIONS
    # =========================================================================
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        try:
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                logger.error("Missing required OHLC columns")
                return False
            
            if len(data) < 10:
                logger.error("Insufficient data points")
                return False
            
            if data[required_columns].isnull().any().any():
                logger.error("Data contains null values")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis"""
        try:
            # Ensure proper data types
            numeric_columns = ['open', 'high', 'low', 'close']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add volume if available
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            else:
                df['volume'] = 1.0  # Default volume
            
            # Reset index to ensure proper indexing
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return df
    
    def _find_peaks_manual(self, data: np.ndarray, window: int = 5) -> np.ndarray:
        """Manual peak detection when scipy is not available"""
        try:
            peaks = []
            
            for i in range(window, len(data) - window):
                is_peak = True
                current_value = data[i]
                
                # Check if current point is higher than surrounding points
                for j in range(i - window, i + window + 1):
                    if j != i and data[j] >= current_value:
                        is_peak = False
                        break
                
                if is_peak:
                    peaks.append(i)
            
            return np.array(peaks)
            
        except Exception as e:
            logger.error(f"Manual peak detection error: {e}")
            return np.array([])
    
    def _calculate_hs_confidence(self, left_shoulder: float, head: float, right_shoulder: float, neckline: float) -> float:
        """Calculate confidence for Head and Shoulders pattern"""
        try:
            # Symmetry of shoulders
            shoulder_symmetry = 1 - abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
            
            # Head prominence
            head_prominence = (head - max(left_shoulder, right_shoulder)) / head
            
            # Neckline quality
            neckline_distance = abs(head - neckline) / head
            
            confidence = (shoulder_symmetry * 0.4 + 
                         head_prominence * 0.4 + 
                         neckline_distance * 0.2)
            
            return min(0.9, max(0.1, confidence))
            
        except Exception as e:
            logger.error(f"H&S confidence calculation error: {e}")
            return 0.5
    
    def _check_volume_confirmation(self, df: pd.DataFrame, indices: List[int], pattern_type: str) -> float:
        """Check volume confirmation for patterns"""
        try:
            if 'volume' not in df.columns:
                return 0.0
            
            volume_confirmation = 0.0
            avg_volume = df['volume'].mean()
            
            for idx in indices:
                if idx < len(df):
                    volume_ratio = df.loc[idx, 'volume'] / avg_volume
                    if pattern_type in ['double_top', 'head_shoulders'] and volume_ratio > 1.2:
                        volume_confirmation += 0.1
                    elif pattern_type in ['double_bottom', 'inverse_head_shoulders'] and volume_ratio > 1.2:
                        volume_confirmation += 0.1
            
            return min(1.0, volume_confirmation)
            
        except Exception as e:
            logger.error(f"Volume confirmation error: {e}")
            return 0.0
    
    def _determine_trend_context(self, df: pd.DataFrame) -> str:
        """Determine trend context for pattern direction bias"""
        try:
            if len(df) < 20:
                return "NEUTRAL"
            
            # Simple trend determination
            recent_close = df['close'].iloc[-1]
            old_close = df['close'].iloc[-20]
            
            change_pct = (recent_close - old_close) / old_close
            
            if change_pct > 0.05:
                return "UP"
            elif change_pct < -0.05:
                return "DOWN"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Trend context error: {e}")
            return "NEUTRAL"
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'symbol': 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'data_points': 0,
            'chart_patterns': {},
            'support_resistance': SupportResistance(supports=[], resistances=[]),
            'fibonacci_levels': FibonacciLevels(swing_high=0, swing_low=0, retracement_levels={}, extension_levels={}),
            'trend_analysis': TrendAnalysis(direction="SIDEWAYS", strength=0.5, duration=0, angle=0, regression_line={}),
            'pattern_confluence': {'confluence_score': 0.0, 'strength': 'WEAK'},
            'overall_signal': {'signal_direction': 'NEUTRAL', 'recommendation': 'HOLD'}
        }

# =============================================================================
# 🧪 TEST FUNCTION
# =============================================================================

def test_pattern_recognition():
    """Test pattern recognition with sample data"""
    try:
        logger.info("Testing Pattern Recognition System...")
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, 100)
        trend = np.linspace(0, 0.1, 100)  # Slight upward trend
        
        prices = []
        current_price = base_price
        
        for i in range(100):
            change = price_changes[i] + trend[i] / 100
            current_price *= (1 + change)
            prices.append(current_price)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.uniform(1000000, 5000000)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Test pattern recognition
        pattern_analyzer = AdvancedPatternRecognition()
        results = pattern_analyzer.analyze_patterns(df, "TEST/USDT")
        
        # Print results
        print(f"\n🔍 Pattern Analysis Results for TEST/USDT:")
        print(f"📊 Data Points: {results['data_points']}")
        print(f"📈 Trend: {results['trend_analysis'].direction} ({results['trend_analysis'].strength:.2f})")
        print(f"🎯 Chart Patterns Found: {len(results['chart_patterns'])}")
        
        for pattern_name, pattern in results['chart_patterns'].items():
            print(f"   📊 {pattern_name}: {pattern.direction} ({pattern.confidence:.2f} confidence)")
        
        print(f"🎯 Support Levels: {len(results['support_resistance'].supports)}")
        print(f"🎯 Resistance Levels: {len(results['support_resistance'].resistances)}")
        
        fib = results['fibonacci_levels']
        print(f"📐 Fibonacci Key Level: {fib.current_level} at ${fib.key_level:.2f}")
        
        confluence = results['pattern_confluence']
        print(f"🔗 Pattern Confluence: {confluence['confluence_score']:.2f} ({confluence['strength']})")
        
        signal = results['overall_signal']
        print(f"📡 Overall Signal: {signal['signal_direction']} - {signal['recommendation']}")
        print(f"💪 Signal Strength: {signal['signal_strength']:.2f}")
        
        logger.info("✅ Pattern Recognition test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pattern Recognition test failed: {e}")
        return False

if __name__ == "__main__":
    test_pattern_recognition()
