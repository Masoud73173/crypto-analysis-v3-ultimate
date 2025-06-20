# analyzer.py - Advanced Cryptocurrency Technical Analysis Engine V3.0 Ultimate
import asyncio
import aiohttp
import ccxt
import pandas as pd
import numpy as np
import talib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

import config

# =============================================================================
# 🔧 CONDITIONAL IMPORTS FOR ADVANCED FEATURES
# =============================================================================

# News Analyzer (conditional import)
try:
    from news_analyzer import NewsAnalyzer, SentimentAnalysis
    NEWS_ANALYZER_AVAILABLE = True
except ImportError:
    NEWS_ANALYZER_AVAILABLE = False
    print("⚠️ News analyzer not available - creating mock")
    
    class MockSentimentAnalysis:
        def __init__(self):
            self.overall_sentiment = 0.0
            self.news_count = 0
            self.positive_ratio = 0.0
            self.negative_ratio = 0.0
            self.neutral_ratio = 1.0
            self.confidence = 0.0
            self.key_events = []
            self.impact_level = 'LOW'

# ML Predictor (conditional import)
try:
    from ml_predictor import MLPredictor, FeatureSet, PredictionResult
    ML_PREDICTOR_AVAILABLE = True
except ImportError:
    ML_PREDICTOR_AVAILABLE = False
    print("⚠️ ML predictor not available - creating mock")
    
    class MockPredictionResult:
        def __init__(self):
            self.success_probability = 0.7
            self.confidence = 0.6
            self.feature_importance = {}
            self.risk_adjusted_score = 0.65
            self.recommendation = 'BUY'

# Pattern Recognition (conditional import)
try:
    from pattern_recognition import PatternRecognition, PatternResult, FibonacciLevels
    PATTERN_RECOGNITION_AVAILABLE = True
except ImportError:
    PATTERN_RECOGNITION_AVAILABLE = False
    print("⚠️ Pattern recognition not available - creating mock")
    
    class MockFibonacciLevels:
        def __init__(self):
            self.swing_high = 0
            self.swing_low = 0
            self.current_level = "Unknown"

# Backtesting Engine (conditional import)
try:
    from backtesting_engine import BacktestingEngine
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    print("⚠️ Backtesting engine not available")

# =============================================================================
# 📊 CRYPTO ANALYZER CLASS
# =============================================================================

class CryptoAnalyzer:
    """Advanced cryptocurrency technical analysis with ML, news sentiment, and pattern recognition"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.exchange = None
        self.cache = {}
        self.cache_duration = config.PERFORMANCE_CONFIG.get('cache_duration_seconds', 300)
        
        # Initialize advanced components
        self.news_analyzer = None
        self.ml_predictor = None
        self.pattern_analyzer = None
        self.backtest_engine = None
        
        # Market regime tracking
        self.market_regime = 'sideways'
        self.btc_dominance = 0.5
        self.fear_greed_index = 50
        self.last_regime_update = None
        
        # Performance tracking
        self.analysis_count = 0
        self.successful_signals = 0
        self.start_time = datetime.now()
        self.last_analysis_time = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize exchange
        self._initialize_exchange()
        
        self.logger.info("🚀 CryptoAnalyzer V3.0 Ultimate initialized")
        self.logger.info(f"🔧 Features: News:{NEWS_ANALYZER_AVAILABLE}, ML:{ML_PREDICTOR_AVAILABLE}, Patterns:{PATTERN_RECOGNITION_AVAILABLE}")
    
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize advanced components
        await self._initialize_advanced_components()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._cleanup_resources()
    
    async def _initialize_advanced_components(self):
        """Initialize advanced analysis components"""
        try:
            # Initialize News Analyzer
            if config.NEWS_SENTIMENT_CONFIG['enabled'] and NEWS_ANALYZER_AVAILABLE:
                self.news_analyzer = NewsAnalyzer()
                await self.news_analyzer.__aenter__()
                self.logger.info("📰 News analyzer initialized")
            
            # Initialize ML Predictor
            if config.ML_CONFIG['enabled'] and ML_PREDICTOR_AVAILABLE:
                self.ml_predictor = MLPredictor()
                self.logger.info("🤖 ML predictor initialized")
            
            # Initialize Pattern Analyzer
            if config.PATTERN_RECOGNITION_CONFIG['enabled'] and PATTERN_RECOGNITION_AVAILABLE:
                self.pattern_analyzer = PatternRecognition()
                self.logger.info("📊 Pattern analyzer initialized")
            
            # Initialize Backtesting Engine
            if config.BACKTESTING_CONFIG['enabled'] and BACKTESTING_AVAILABLE:
                self.backtest_engine = BacktestingEngine()
                self.logger.info("📈 Backtesting engine initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing advanced components: {str(e)}")
    
    async def _cleanup_resources(self):
        """Cleanup resources"""
        try:
            if self.news_analyzer:
                await self.news_analyzer.__aexit__(None, None, None)
            
            if self.exchange:
                await self.exchange.close()
                
            self.logger.info("🧹 Resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def _initialize_exchange(self):
        """Initialize cryptocurrency exchange"""
        try:
            # Initialize with API keys if available
            exchange_config = {
                'enableRateLimit': True,
                'timeout': config.EXCHANGE_CONFIG.get('timeout', 30000),
                'options': {'defaultType': 'future'}
            }
            
            if config.BINANCE_API_KEY and config.BINANCE_SECRET:
                exchange_config.update({
                    'apiKey': config.BINANCE_API_KEY,
                    'secret': config.BINANCE_SECRET,
                    'sandbox': False
                })
                self.logger.info("🔑 Exchange initialized with API credentials")
            else:
                self.logger.info("📊 Exchange initialized in read-only mode")
            
            self.exchange = ccxt.binance(exchange_config)
            
        except Exception as e:
            self.logger.error(f"Error initializing exchange: {str(e)}")
            # Fallback to basic exchange
            self.exchange = ccxt.binance({'enableRateLimit': True})
    
    # =============================================================================
    # 🎯 MAIN ANALYSIS METHODS
    # =============================================================================
    
    async def analyze_multiple_cryptos(self, timeframe: str = '1h') -> List[Dict]:
        """Enhanced analysis with ML, news sentiment, and pattern recognition"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🔥 Starting ULTIMATE analysis for timeframe: {timeframe}")
            
            # Update market regime
            await self._update_market_regime()
            
            # Get top symbols
            symbols = await self.get_top_futures_symbols()
            if not symbols:
                self.logger.warning("No symbols found for analysis")
                return []
            
            # Limit symbols for extended timeframes
            if timeframe in ['1d', '3d', '1w']:
                symbols = symbols[:50]  # Focus on top 50 for longer timeframes
            
            self.logger.info(f"📊 Analyzing {len(symbols)} symbols with advanced features")
            
            signals = []
            
            # Process in batches for better performance
            batch_size = config.PERFORMANCE_CONFIG.get('max_workers', 10)
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                # Create analysis tasks for batch
                tasks = []
                for symbol in batch:
                    tasks.append(self._analyze_single_crypto_advanced(symbol, timeframe))
                
                # Execute batch analysis
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect valid signals
                for result in batch_results:
                    if isinstance(result, dict) and result.get('signal_strength', 0) >= config.MIN_SIGNAL_STRENGTH:
                        signals.append(result)
                    elif isinstance(result, Exception):
                        self.logger.debug(f"Analysis exception: {str(result)}")
                
                # Rate limiting between batches
                if i + batch_size < len(symbols):
                    await asyncio.sleep(config.REQUEST_DELAY * 5)
            
            # Enhanced signal filtering and ranking
            signals = await self._enhance_and_filter_signals(signals, timeframe)
            
            # Limit final results
            max_signals = config.SIGNAL_COUNT
            if timeframe in ['1d', '3d', '1w']:
                max_signals = min(max_signals, 5)  # Fewer signals for longer timeframes
            
            signals = signals[:max_signals]
            
            # Update statistics
            with self.lock:
                self.analysis_count += 1
                self.last_analysis_time = datetime.now()
                if signals:
                    self.successful_signals += 1
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✅ Ultimate analysis complete: {len(signals)} high-quality signals in {duration:.1f}s")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"💥 Error in enhanced analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []
    
    async def _analyze_single_crypto_advanced(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Advanced single crypto analysis with all features"""
        try:
            # Get market data
            df = await self.get_market_data(symbol, timeframe)
            if df is None or len(df) < 30:
                return None
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(df)
            if not indicators:
                return None
            
            # === ADVANCED FEATURES INTEGRATION ===
            
            # 1. News Sentiment Analysis
            news_sentiment = await self._get_news_sentiment(symbol)
            
            # 2. Pattern Recognition
            patterns = await self._get_pattern_analysis(df)
            
            # 3. Fibonacci Analysis
            fibonacci = await self._get_fibonacci_analysis(df)
            
            # 4. ML Prediction
            ml_prediction = await self._get_ml_prediction(indicators, news_sentiment, patterns)
            
            # === ENHANCED SIGNAL GENERATION ===
            
            # Generate base signal
            base_signal = self.generate_enhanced_signal(indicators, timeframe)
            if not base_signal:
                return None
            
            # Enhance with advanced features
            enhanced_signal = await self._enhance_signal_with_advanced_features(
                base_signal, news_sentiment, patterns, fibonacci, ml_prediction, timeframe
            )
            
            # Apply market regime adjustments
            final_signal = self._apply_market_regime_adjustments(enhanced_signal)
            
            return final_signal
            
        except Exception as e:
            self.logger.debug(f"Error in advanced analysis for {symbol}: {str(e)}")
            return None
    
    # =============================================================================
    # 📊 MARKET DATA METHODS
    # =============================================================================
    
    async def get_top_futures_symbols(self) -> List[str]:
        """Get top futures trading pairs by volume"""
        cache_key = "top_futures_symbols"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Load markets
            markets = await asyncio.to_thread(self.exchange.load_markets)
            
            # Get futures markets with USDT
            futures_symbols = []
            for symbol, market in markets.items():
                if (market.get('future', False) and 
                    '/USDT' in symbol and 
                    market.get('active', True)):
                    futures_symbols.append(symbol)
            
            if not futures_symbols:
                self.logger.warning("No futures symbols found, using spot markets")
                futures_symbols = [s for s in markets.keys() if '/USDT' in s][:config.CRYPTO_COUNT]
            
            # Get 24h volume data
            try:
                tickers = await asyncio.to_thread(self.exchange.fetch_tickers)
                
                # Filter and sort by volume
                volume_data = []
                for symbol in futures_symbols:
                    if symbol in tickers:
                        ticker = tickers[symbol]
                        volume = ticker.get('quoteVolume', 0) or ticker.get('baseVolume', 0)
                        if volume > 0:
                            volume_data.append({
                                'symbol': symbol,
                                'volume': volume
                            })
                
                # Sort by volume and get top symbols
                volume_data.sort(key=lambda x: x['volume'], reverse=True)
                top_symbols = [item['symbol'] for item in volume_data[:config.CRYPTO_COUNT]]
                
                if top_symbols:
                    self._cache_data(cache_key, top_symbols)
                    self.logger.info(f"📈 Found {len(top_symbols)} top futures symbols")
                    return top_symbols
                
            except Exception as e:
                self.logger.warning(f"Error fetching volume data: {str(e)}")
            
            # Fallback to first available symbols
            fallback_symbols = futures_symbols[:config.CRYPTO_COUNT]
            self._cache_data(cache_key, fallback_symbols)
            return fallback_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting top futures symbols: {str(e)}")
            # Ultimate fallback
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
    
    async def get_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch market data for analysis"""
        cache_key = f"market_data_{symbol}_{timeframe}_{limit}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Fetch OHLCV data
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv, 
                symbol, 
                timeframe, 
                limit=limit
            )
            
            if not ohlcv or len(ohlcv) < 20:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any NaN rows
            df = df.dropna()
            
            if len(df) >= 20:
                self._cache_data(cache_key, df)
                return df
            
        except Exception as e:
            self.logger.debug(f"Error fetching market data for {symbol}: {str(e)}")
        
        return None
    
    # =============================================================================
    # 📊 TECHNICAL ANALYSIS METHODS
    # =============================================================================
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate comprehensive technical indicators"""
        try:
            if len(df) < 20:
                return None
            
            indicators = {}
            
            # Price data
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # === CORE INDICATORS ===
            
            # RSI
            indicators['rsi'] = float(talib.RSI(close, timeperiod=config.RSI_PERIOD)[-1])
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, 
                                                     fastperiod=config.MACD_FAST,
                                                     slowperiod=config.MACD_SLOW, 
                                                     signalperiod=config.MACD_SIGNAL)
            indicators['macd'] = float(macd[-1])
            indicators['macd_signal'] = float(macd_signal[-1])
            indicators['macd_histogram'] = float(macd_hist[-1])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 
                                                        timeperiod=config.BB_PERIOD,
                                                        nbdevup=config.BB_DEVIATION,
                                                        nbdevdn=config.BB_DEVIATION)
            indicators['bb_upper'] = float(bb_upper[-1])
            indicators['bb_middle'] = float(bb_middle[-1])
            indicators['bb_lower'] = float(bb_lower[-1])
            
            # BB Position (0 = lower band, 1 = upper band)
            bb_range = indicators['bb_upper'] - indicators['bb_lower']
            if bb_range > 0:
                indicators['bb_position'] = (close[-1] - indicators['bb_lower']) / bb_range
            else:
                indicators['bb_position'] = 0.5
            
            # === ADDITIONAL INDICATORS ===
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close,
                                          fastk_period=config.STOCH_K_PERIOD,
                                          slowk_period=config.STOCH_D_PERIOD,
                                          slowd_period=config.STOCH_D_PERIOD)
            indicators['stoch_k'] = float(stoch_k[-1])
            indicators['stoch_d'] = float(stoch_d[-1])
            
            # Williams %R
            indicators['williams_r'] = float(talib.WILLR(high, low, close, timeperiod=14)[-1])
            
            # ATR (Average True Range)
            indicators['atr'] = float(talib.ATR(high, low, close, timeperiod=config.ATR_PERIOD)[-1])
            
            # Moving Averages
            indicators['sma_20'] = float(talib.SMA(close, timeperiod=config.SMA_SHORT)[-1])
            indicators['sma_50'] = float(talib.SMA(close, timeperiod=config.SMA_LONG)[-1])
            indicators['ema_12'] = float(talib.EMA(close, timeperiod=config.EMA_SHORT)[-1])
            indicators['ema_26'] = float(talib.EMA(close, timeperiod=config.EMA_LONG)[-1])
            
            # === VOLUME ANALYSIS ===
            
            # Volume SMA
            volume_sma = talib.SMA(volume, timeperiod=config.VOLUME_SMA_PERIOD)
            indicators['volume_sma'] = float(volume_sma[-1])
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1.0
            
            # === VOLATILITY AND MOMENTUM ===
            
            # Price changes
            returns = pd.Series(close).pct_change().dropna()
            indicators['volatility'] = float(returns.rolling(config.VOLATILITY_PERIOD).std().iloc[-1])
            
            # Trend strength
            current_price = close[-1]
            sma_50 = indicators['sma_50']
            if sma_50 > 0:
                indicators['trend_strength'] = (current_price - sma_50) / sma_50
            else:
                indicators['trend_strength'] = 0.0
            
            # Momentum score
            momentum_periods = [5, 10, 20]
            momentum_scores = []
            for period in momentum_periods:
                if len(close) > period:
                    momentum = (close[-1] - close[-period-1]) / close[-period-1]
                    momentum_scores.append(momentum)
            
            indicators['momentum_score'] = np.mean(momentum_scores) if momentum_scores else 0.0
            
            # === ADDITIONAL FEATURES ===
            
            # Current price info
            indicators['current_price'] = float(close[-1])
            indicators['price_change_24h'] = float((close[-1] - close[-24]) / close[-24] if len(close) >= 24 else 0)
            
            # Timestamp
            indicators['timestamp'] = df.index[-1].isoformat()
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return None
    
    def generate_enhanced_signal(self, indicators: Dict, timeframe: str) -> Optional[Dict]:
        """Generate enhanced trading signal with confluence analysis"""
        try:
            signal = {
                'timestamp': datetime.now(config.TIMEZONE).isoformat(),
                'timeframe': timeframe,
                'signal_strength': 0,
                'type': None,
                'reasons': [],
                'confluence_count': 0,
                'risk_level': 'MEDIUM'
            }
            
            signal_strength = 0
            reasons = []
            confluence_count = 0
            
            # === RSI ANALYSIS ===
            rsi = indicators.get('rsi', 50)
            signal['rsi'] = rsi
            
            if rsi <= config.RSI_EXTREME_OVERSOLD:  # Extreme oversold
                signal_strength += 35
                signal['type'] = 'LONG'
                reasons.append('RSI Extreme Oversold')
                confluence_count += 1
                signal['risk_level'] = 'HIGH'
            elif rsi <= config.RSI_OVERSOLD:  # Normal oversold
                signal_strength += 25
                signal['type'] = 'LONG'
                reasons.append('RSI Oversold')
                confluence_count += 1
            elif rsi >= config.RSI_EXTREME_OVERBOUGHT:  # Extreme overbought
                signal_strength += 35
                signal['type'] = 'SHORT'
                reasons.append('RSI Extreme Overbought')
                confluence_count += 1
                signal['risk_level'] = 'HIGH'
            elif rsi >= config.RSI_OVERBOUGHT:  # Normal overbought
                signal_strength += 25
                signal['type'] = 'SHORT'
                reasons.append('RSI Overbought')
                confluence_count += 1
            
            # === MACD ANALYSIS ===
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_hist = indicators.get('macd_histogram', 0)
            
            # MACD crossover detection
            if macd > macd_signal and macd_hist > 0:
                if signal['type'] == 'LONG' or signal['type'] is None:
                    signal_strength += 20
                    signal['type'] = 'LONG'
                    reasons.append('MACD Bullish Crossover')
                    confluence_count += 1
            elif macd < macd_signal and macd_hist < 0:
                if signal['type'] == 'SHORT' or signal['type'] is None:
                    signal_strength += 20
                    signal['type'] = 'SHORT'
                    reasons.append('MACD Bearish Crossover')
                    confluence_count += 1
            
            # === BOLLINGER BANDS ANALYSIS ===
            bb_position = indicators.get('bb_position', 0.5)
            
            if bb_position <= config.BB_OVERSOLD:  # Near lower band
                if signal['type'] == 'LONG' or signal['type'] is None:
                    signal_strength += 15
                    signal['type'] = 'LONG'
                    reasons.append('BB Oversold')
                    confluence_count += 1
            elif bb_position >= config.BB_OVERBOUGHT:  # Near upper band
                if signal['type'] == 'SHORT' or signal['type'] is None:
                    signal_strength += 15
                    signal['type'] = 'SHORT'
                    reasons.append('BB Overbought')
                    confluence_count += 1
            
            # === STOCHASTIC ANALYSIS ===
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            
            if stoch_k <= config.STOCH_OVERSOLD and stoch_d <= config.STOCH_OVERSOLD:
                if signal['type'] == 'LONG':
                    signal_strength += 10
                    reasons.append('Stochastic Oversold')
                    confluence_count += 1
            elif stoch_k >= config.STOCH_OVERBOUGHT and stoch_d >= config.STOCH_OVERBOUGHT:
                if signal['type'] == 'SHORT':
                    signal_strength += 10
                    reasons.append('Stochastic Overbought')
                    confluence_count += 1
            
            # === VOLUME CONFIRMATION ===
            volume_ratio = indicators.get('volume_ratio', 1.0)
            
            if volume_ratio >= config.VOLUME_SPIKE_THRESHOLD:
                signal_strength += 10
                reasons.append('Volume Spike Confirmation')
                confluence_count += 1
                signal['volume_confirmed'] = True
            else:
                signal['volume_confirmed'] = False
            
            # === TREND ANALYSIS ===
            trend_strength = indicators.get('trend_strength', 0)
            
            if abs(trend_strength) > 0.05:  # Strong trend
                if trend_strength > 0 and signal['type'] == 'LONG':
                    signal_strength += 15
                    reasons.append('Strong Uptrend Alignment')
                    confluence_count += 1
                elif trend_strength < 0 and signal['type'] == 'SHORT':
                    signal_strength += 15
                    reasons.append('Strong Downtrend Alignment')
                    confluence_count += 1
            
            # === VOLATILITY ADJUSTMENT ===
            volatility = indicators.get('volatility', 0.02)
            
            if volatility > config.HIGH_VOLATILITY_THRESHOLD:
                signal_strength *= 0.9  # Reduce strength in high volatility
                signal['risk_level'] = 'HIGH'
                reasons.append('High Volatility Warning')
            elif volatility < config.LOW_VOLATILITY_THRESHOLD:
                signal_strength *= 1.1  # Boost strength in low volatility
                if signal['risk_level'] == 'MEDIUM':
                    signal['risk_level'] = 'LOW'
            
            # === SIGNAL VALIDATION ===
            
            # Minimum confluence requirement
            if confluence_count < 2:
                return None  # Need at least 2 confirming indicators
            
            # Minimum strength requirement
            timeframe_config = config.EXTENDED_TIMEFRAMES.get(timeframe, {})
            min_strength = timeframe_config.get('min_strength', config.MIN_SIGNAL_STRENGTH)
            
            if signal_strength < min_strength:
                return None
            
            # === FINALIZE SIGNAL ===
            
            signal['signal_strength'] = signal_strength
            signal['reasons'] = reasons
            signal['confluence_count'] = confluence_count
            signal['volatility'] = volatility
            signal['volume_ratio'] = volume_ratio
            signal['trend_strength'] = trend_strength
            
            # Calculate entry points and targets
            current_price = indicators.get('current_price', 0)
            atr = indicators.get('atr', current_price * 0.02)
            
            signal.update(self._calculate_entry_and_targets(current_price, atr, signal['type'], signal['risk_level'], timeframe))
            
            # Add all indicator values for reference
            signal['indicators'] = indicators
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced signal: {str(e)}")
            return None
    
    def _calculate_entry_and_targets(self, current_price: float, atr: float, signal_type: str, risk_level: str, timeframe: str) -> Dict:
        """Calculate entry points, targets, and stop loss"""
        try:
            entry_spread = atr * 0.5  # Use ATR for entry spread
            
            # Entry zone
            if signal_type == 'LONG':
                entry_min = current_price - entry_spread
                entry_max = current_price + entry_spread
            else:  # SHORT
                entry_min = current_price - entry_spread
                entry_max = current_price + entry_spread
            
            # Base multipliers
            multipliers = config.TARGET_MULTIPLIERS[signal_type]
            
            # Timeframe adjustment
            timeframe_mult = config.TIMEFRAME_TARGET_ADJUSTMENTS.get(timeframe, 1.0)
            
            # Calculate targets
            if signal_type == 'LONG':
                target1 = current_price * (1 + multipliers['target1'] * timeframe_mult)
                target2 = current_price * (1 + multipliers['target2'] * timeframe_mult)
                target3 = current_price * (1 + multipliers['target3'] * timeframe_mult)
                stop_loss = current_price * (1 + multipliers['stop_loss'])
            else:  # SHORT
                target1 = current_price * (1 + multipliers['target1'] * timeframe_mult)
                target2 = current_price * (1 + multipliers['target2'] * timeframe_mult)
                target3 = current_price * (1 + multipliers['target3'] * timeframe_mult)
                stop_loss = current_price * (1 + multipliers['stop_loss'])
            
            # Risk/reward calculation
            risk = abs(current_price - stop_loss) / current_price
            reward = abs(target1 - current_price) / current_price
            risk_reward_ratio = reward / risk if risk > 0 else 1.0
            
            return {
                'entry_min': entry_min,
                'entry_max': entry_max,
                'entry_price': current_price,
                'targets': [target1, target2, target3],
                'stop_loss': stop_loss,
                'risk_reward_ratio': risk_reward_ratio,
                'atr_value': atr
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating entry and targets: {str(e)}")
            return {
                'entry_min': current_price,
                'entry_max': current_price,
                'entry_price': current_price,
                'targets': [current_price * 1.02],
                'stop_loss': current_price * 0.98,
                'risk_reward_ratio': 1.0,
                'atr_value': current_price * 0.02
            }
    
    # =============================================================================
    # 🤖 ADVANCED FEATURES INTEGRATION
    # =============================================================================
    
    async def _get_news_sentiment(self, symbol: str):
        """Get news sentiment for symbol"""
        try:
            if self.news_analyzer:
                return await self.news_analyzer.get_crypto_news_sentiment(symbol)
            else:
                # Return mock sentiment
                return MockSentimentAnalysis()
        except Exception as e:
            self.logger.debug(f"Error getting news sentiment: {str(e)}")
            return MockSentimentAnalysis()
    
    async def _get_pattern_analysis(self, df: pd.DataFrame) -> List:
        """Get pattern recognition analysis"""
        try:
            if self.pattern_analyzer:
                return self.pattern_analyzer.analyze_patterns(df)
            else:
                return []
        except Exception as e:
            self.logger.debug(f"Error in pattern analysis: {str(e)}")
            return []
    
    async def _get_fibonacci_analysis(self, df: pd.DataFrame):
        """Get Fibonacci analysis"""
        try:
            if self.pattern_analyzer:
                return self.pattern_analyzer.analyze_fibonacci_levels(df)
            else:
                return MockFibonacciLevels()
        except Exception as e:
            self.logger.debug(f"Error in Fibonacci analysis: {str(e)}")
            return MockFibonacciLevels()
    
    async def _get_ml_prediction(self, indicators: Dict, news_sentiment, patterns: List):
        """Get ML prediction"""
        try:
            if not self.ml_predictor:
                return MockPredictionResult()
            
            # Create feature set for ML
            pattern_score = max([getattr(p, 'confidence', 0) for p in patterns], default=0.0)
            
            # Mock FeatureSet since we don't have the actual class
            features = type('MockFeatureSet', (), {
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd', 0),
                'macd_signal': indicators.get('macd_signal', 0),
                'bb_position': indicators.get('bb_position', 0.5),
                'volume_ratio': indicators.get('volume_ratio', 1.0),
                'volatility': indicators.get('volatility', 0.02),
                'trend_strength': indicators.get('trend_strength', 0),
                'support_distance': 0.05,
                'resistance_distance': 0.05,
                'momentum_score': indicators.get('momentum_score', 0),
                'confluence_count': indicators.get('confluence_count', 1),
                'pattern_score': pattern_score,
                'news_sentiment': getattr(news_sentiment, 'overall_sentiment', 0),
                'news_confidence': getattr(news_sentiment, 'confidence', 0),
                'news_volume': getattr(news_sentiment, 'news_count', 0),
                'market_regime_score': self._get_market_regime_score(),
                'vix_level': 0.2,
                'btc_dominance': self.btc_dominance
            })()
            
            signal_strength = indicators.get('signal_strength', 50)
            return await self.ml_predictor.predict_signal_success(features, signal_strength)
            
        except Exception as e:
            self.logger.debug(f"Error in ML prediction: {str(e)}")
            return MockPredictionResult()
    
    # =============================================================================
    # 🎯 MARKET REGIME AND ENHANCEMENT METHODS
    # =============================================================================
    
    async def _update_market_regime(self):
        """Update current market regime indicators"""
        try:
            # Skip if recently updated
            if (self.last_regime_update and 
                (datetime.now() - self.last_regime_update).seconds < 3600):  # 1 hour
                return
            
            # Get BTC data for market regime analysis
            btc_df = await self.get_market_data('BTC/USDT', '1d', limit=30)
            
            if btc_df is not None and len(btc_df) >= 10:
                # Calculate BTC trend
                btc_sma_7 = btc_df['close'].rolling(7).mean()
                btc_sma_21 = btc_df['close'].rolling(21).mean()
                
                current_price = btc_df['close'].iloc[-1]
                sma_7_current = btc_sma_7.iloc[-1]
                sma_21_current = btc_sma_21.iloc[-1]
                
                # Determine market regime
                if current_price > sma_7_current > sma_21_current:
                    self.market_regime = 'bull'
                elif current_price < sma_7_current < sma_21_current:
                    self.market_regime = 'bear'
                else:
                    self.market_regime = 'sideways'
                
                # Calculate volatility
                returns = btc_df['close'].pct_change().dropna()
                volatility = returns.rolling(14).std().iloc[-1]
                
                if volatility > 0.04:  # 4% daily volatility
                    self.market_regime = 'volatile'
                
                self.last_regime_update = datetime.now()
            
            self.logger.debug(f"📊 Market regime: {self.market_regime}")
            
        except Exception as e:
            self.logger.debug(f"Error updating market regime: {str(e)}")
            self.market_regime = 'sideways'
    
    def _get_market_regime_score(self) -> float:
        """Get market regime score for ML features"""
        regime_scores = {
            'bull': 0.8,
            'bear': -0.8,
            'sideways': 0.0,
            'volatile': -0.3
        }
        return regime_scores.get(self.market_regime, 0.0)
    
    async def _enhance_and_filter_signals(self, signals: List[Dict], timeframe: str) -> List[Dict]:
        """Enhanced signal filtering and ranking with advanced features"""
        try:
            if not signals:
                return []
            
            enhanced_signals = []
            
            for signal in signals:
                # Apply final quality checks
                if self._passes_quality_checks(signal, timeframe):
                    enhanced_signals.append(signal)
            
            # Advanced ranking
            enhanced_signals.sort(key=self._calculate_signal_score, reverse=True)
            
            # Diversification filter
            diversified_signals = self._apply_diversification_filter(enhanced_signals)
            
            return diversified_signals
            
        except Exception as e:
            self.logger.error(f"Error in enhanced filtering: {str(e)}")
            return signals
    
    def _passes_quality_checks(self, signal: Dict, timeframe: str) -> bool:
        """Enhanced quality checks for signals"""
        try:
            # Basic strength check
            min_strength = config.EXTENDED_TIMEFRAMES.get(timeframe, {}).get('min_strength', config.MIN_SIGNAL_STRENGTH)
            if signal.get('signal_strength', 0) < min_strength:
                return False
            
            # Confluence check
            if signal.get('confluence_count', 0) < 2:
                return False
            
            # Volume confirmation for higher timeframes
            if timeframe in ['4h', '1d', '3d', '1w'] and not signal.get('volume_confirmed', False):
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Error in quality checks: {str(e)}")
            return True
    
    def _calculate_signal_score(self, signal: Dict) -> float:
        """Calculate comprehensive signal score for ranking"""
        try:
            score = 0.0
            
            # Base signal strength (50%)
            base_strength = signal.get('signal_strength', 50) / 100.0
            score += base_strength * 0.5
            
            # Confluence bonus (25%)
            confluence_count = signal.get('confluence_count', 1)
            confluence_score = min(confluence_count / 5.0, 1.0)
            score += confluence_score * 0.25
            
            # Volume confirmation (15%)
            if signal.get('volume_confirmed', False):
                score += 0.15
            
            # Risk/reward ratio (10%)
            rr_ratio = signal.get('risk_reward_ratio', 1.0)
            rr_score = min(rr_ratio / 3.0, 1.0)  # Normalize to max 3:1
            score += rr_score * 0.1
            
            return score
            
        except Exception as e:
            self.logger.debug(f"Error calculating signal score: {str(e)}")
            return signal.get('signal_strength', 50) / 100.0
    
    def _apply_diversification_filter(self, signals: List[Dict]) -> List[Dict]:
        """Apply diversification to avoid too many correlated signals"""
        try:
            if len(signals) <= 3:
                return signals
            
            diversified = []
            used_base_symbols = set()
            
            # Prioritize different base currencies
            for signal in signals:
                symbol = signal.get('symbol', '')
                base_currency = symbol.split('/')[0] if '/' in symbol else symbol[:3]
                
                # Allow maximum 2 signals per base currency
                base_count = sum(1 for s in diversified 
                               if s.get('symbol', '').split('/')[0] == base_currency)
                
                if base_count < 2:
                    diversified.append(signal)
                    used_base_symbols.add(base_currency)
                
                # Stop when we have enough diverse signals
                if len(diversified) >= config.SIGNAL_COUNT:
                    break
            
            return diversified
            
        except Exception as e:
            self.logger.debug(f"Error in diversification filter: {str(e)}")
            return signals
    
    # =============================================================================
    # 🛠️ UTILITY METHODS
    # =============================================================================
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        
        cache_entry = self.cache[key]
        if not isinstance(cache_entry, dict) or '_timestamp' not in cache_entry:
            return False
        
        cache_age = (datetime.now() - cache_entry['_timestamp']).total_seconds()
        return cache_age < self.cache_duration
    
    def _cache_data(self, key: str, data):
        """Cache data with timestamp"""
        self.cache[key] = {
            'data': data,
            '_timestamp': datetime.now()
        }
        
        # Get cached data
        if key in self.cache:
            return self.cache[key]['data']
        return data
    
    def get_performance_stats(self) -> Dict:
        """Get analyzer performance statistics"""
        uptime = datetime.now() - self.start_time
        
        return {
            'analysis_count': self.analysis_count,
            'successful_signals': self.successful_signals,
            'success_rate': self.successful_signals / max(self.analysis_count, 1),
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime),
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'market_regime': self.market_regime,
            'cache_size': len(self.cache),
            'features_enabled': {
                'news_analyzer': self.news_analyzer is not None,
                'ml_predictor': self.ml_predictor is not None,
                'pattern_analyzer': self.pattern_analyzer is not None,
                'backtest_engine': self.backtest_engine is not None
            }
        }

# =============================================================================
# 🧪 TESTING AND VALIDATION
# =============================================================================

async def test_analyzer():
    """Test function for the analyzer"""
    print("🧪 Testing CryptoAnalyzer V3.0 Ultimate...")
    
    async with CryptoAnalyzer() as analyzer:
        try:
            # Test getting top symbols
            symbols = await analyzer.get_top_futures_symbols()
            print(f"✅ Found {len(symbols)} symbols")
            
            # Test market data
            if symbols:
                df = await analyzer.get_market_data(symbols[0], '1h')
                if df is not None:
                    print(f"✅ Market data: {len(df)} candles for {symbols[0]}")
                    
                    # Test indicators
                    indicators = analyzer.calculate_technical_indicators(df)
                    if indicators:
                        print(f"✅ Indicators calculated: RSI={indicators.get('rsi', 'N/A'):.1f}")
                        
                        # Test signal generation
                        signal = analyzer.generate_enhanced_signal(indicators, '1h')
                        if signal:
                            print(f"✅ Signal generated: {signal['type']} with {signal['signal_strength']:.1f}% strength")
                        else:
                            print("ℹ️ No signal generated (normal)")
                    else:
                        print("❌ Failed to calculate indicators")
                else:
                    print("❌ Failed to get market data")
            
            # Test performance stats
            stats = analyzer.get_performance_stats()
            print(f"✅ Performance stats: {stats['features_enabled']}")
            
            print("🎉 CryptoAnalyzer test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_analyzer())