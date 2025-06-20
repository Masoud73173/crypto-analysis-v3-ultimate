# signal_formatter.py - Advanced Telegram Signal Formatting Engine V3.0 Ultimate
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import re
import hashlib

try:
    import config
except ImportError:
    # Fallback configuration
    class MockConfig:
        TELEGRAM_CONFIG = {
            'max_message_length': 4000,
            'parse_mode': 'HTML',
            'message_formatting': {'use_emojis': True, 'bold_headers': True}
        }
        TELEGRAM_USER = '@MasoudHaddad69'
        TIMEZONE = None
        EXTENDED_TIMEFRAMES = {}
    config = MockConfig()

# =============================================================================
# 📊 DATA CLASSES
# =============================================================================

@dataclass
class FormattedMessage:
    """Formatted message for Telegram"""
    content: str
    message_type: str  # 'signal', 'summary', 'error', 'notification'
    length: int
    is_valid: bool
    truncated: bool = False
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class SignalSummary:
    """Summary of multiple signals"""
    total_signals: int
    strong_signals: int
    bullish_count: int
    bearish_count: int
    avg_confidence: float
    timeframe: str
    market_bias: str
    top_opportunities: List[str]

# =============================================================================
# 💎 SIGNAL FORMATTER CLASS
# =============================================================================

class SignalFormatter:
    """Advanced signal formatting for professional Telegram messages"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # Formatting configuration
        self.max_length = config.TELEGRAM_CONFIG.get('max_message_length', 4000)
        self.use_emojis = config.TELEGRAM_CONFIG.get('message_formatting', {}).get('use_emojis', True)
        self.bold_headers = config.TELEGRAM_CONFIG.get('message_formatting', {}).get('bold_headers', True)
        
        # Emoji mappings
        self.emojis = {
            # Signal types
            'LONG': '🟢',
            'SHORT': '🔴', 
            'NEUTRAL': '🟡',
            
            # Timeframes
            '5m': '⚡',
            '15m': '⚡',
            '1h': '🕐',
            '4h': '🕓', 
            '1d': '📅',
            '3d': '📅',
            '1w': '📅',
            
            # Risk levels
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🔴',
            
            # Other elements
            'targets': '🎯',
            'stop_loss': '🛑',
            'entry': '💰',
            'ml': '🤖',
            'news': '📰',
            'pattern': '📊',
            'volume': '📈',
            'strength': '💎',
            'confluence': '🔗',
            'fibonacci': '📐',
            'support_resistance': '🎯'
        }
        
        # Risk level colors for HTML
        self.risk_colors = {
            'LOW': '#00ff00',      # Green
            'MEDIUM': '#ffaa00',   # Orange  
            'HIGH': '#ff0000'      # Red
        }
        
        # Pattern formatting
        self.pattern_descriptions = {
            'HEAD_AND_SHOULDERS': 'H&S Reversal',
            'INVERSE_HEAD_AND_SHOULDERS': 'Inverse H&S',
            'DOUBLE_TOP': 'Double Top',
            'DOUBLE_BOTTOM': 'Double Bottom',
            'ASCENDING_TRIANGLE': 'Ascending Triangle',
            'DESCENDING_TRIANGLE': 'Descending Triangle',
            'SYMMETRIC_TRIANGLE': 'Symmetric Triangle',
            'FLAG': 'Flag Pattern',
            'PENNANT': 'Pennant',
            'RISING_WEDGE': 'Rising Wedge',
            'FALLING_WEDGE': 'Falling Wedge'
        }
        
        self.logger.info("💎 SignalFormatter V3.0 Ultimate initialized")
    
    def _setup_logger(self):
        """Setup logging for signal formatter"""
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
    
    # =============================================================================
    # 🎯 MAIN FORMATTING METHODS
    # =============================================================================
    
    def format_signals_summary(self, 
                             signals: List[Dict], 
                             timeframe: str,
                             analysis_info: Optional[Dict] = None) -> FormattedMessage:
        """
        Format comprehensive signals summary with market analysis
        
        Args:
            signals: List of trading signals
            timeframe: Current timeframe being analyzed
            analysis_info: Additional analysis information
            
        Returns:
            Formatted message for Telegram
        """
        try:
            if not signals:
                return self._format_no_signals_message(timeframe)
            
            # Create signal summary
            summary = self._create_signal_summary(signals, timeframe)
            
            # Build message content
            content_parts = []
            
            # Header
            content_parts.append(self._format_header(timeframe, analysis_info))
            
            # Market summary
            content_parts.append(self._format_market_summary(summary, analysis_info))
            
            # Top signals
            content_parts.append(self._format_top_signals(signals[:3]))  # Top 3 signals
            
            # Analysis footer
            content_parts.append(self._format_analysis_footer(analysis_info))
            
            # Join all parts
            full_content = '\n\n'.join(content_parts)
            
            # Check length and truncate if necessary
            final_content, truncated = self._ensure_message_length(full_content)
            
            return FormattedMessage(
                content=final_content,
                message_type='summary',
                length=len(final_content),
                is_valid=True,
                truncated=truncated,
                priority=1
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting signals summary: {str(e)}")
            return self._format_error_message("Failed to format signals summary")
    
    def format_individual_signal(self, signal: Dict, enhanced_info: Optional[Dict] = None) -> FormattedMessage:
        """
        Format individual trading signal with enhanced information
        
        Args:
            signal: Trading signal data
            enhanced_info: Enhanced analysis data (ML, news, patterns)
            
        Returns:
            Formatted signal message
        """
        try:
            # Build signal content
            content_parts = []
            
            # Signal header
            content_parts.append(self._format_signal_header(signal))
            
            # Main signal info
            content_parts.append(self._format_signal_main_info(signal))
            
            # Entry and targets
            content_parts.append(self._format_signal_levels(signal))
            
            # Technical analysis
            content_parts.append(self._format_technical_analysis(signal))
            
            # Enhanced features (ML, News, Patterns)
            if enhanced_info:
                enhanced_section = self._format_enhanced_features(enhanced_info)
                if enhanced_section:
                    content_parts.append(enhanced_section)
            
            # Signal footer
            content_parts.append(self._format_signal_footer(signal))
            
            # Join all parts
            full_content = '\n\n'.join(content_parts)
            
            # Ensure proper length
            final_content, truncated = self._ensure_message_length(full_content)
            
            return FormattedMessage(
                content=final_content,
                message_type='signal',
                length=len(final_content),
                is_valid=True,
                truncated=truncated,
                priority=self._calculate_signal_priority(signal)
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting individual signal: {str(e)}")
            return self._format_error_message("Failed to format signal")
    
    # =============================================================================
    # 🔥 HEADER AND SUMMARY FORMATTING
    # =============================================================================
    
    def _format_header(self, timeframe: str, analysis_info: Optional[Dict] = None) -> str:
        """Format message header with branding and timestamp"""
        try:
            # Get timezone-aware timestamp
            if config.TIMEZONE:
                timestamp = datetime.now(config.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S (%Z)')
            else:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S (UTC)')
            
            header_parts = []
            
            # Title with emoji
            if self.bold_headers:
                header_parts.append("<b>🚀 Crypto Futures Analysis V3.0 Ultimate</b>")
            else:
                header_parts.append("🚀 Crypto Futures Analysis V3.0 Ultimate")
            
            # Timestamp
            header_parts.append(f"📅 {timestamp}")
            
            # User attribution
            if hasattr(config, 'TELEGRAM_USER') and config.TELEGRAM_USER:
                header_parts.append(f"👤 {config.TELEGRAM_USER}")
            
            # Advanced features badge
            features = []
            if analysis_info:
                if analysis_info.get('ml_enabled'):
                    features.append('🤖 AI')
                if analysis_info.get('news_enabled'):
                    features.append('📰 News')
                if analysis_info.get('patterns_enabled'):
                    features.append('📊 Patterns')
                    
                if features:
                    header_parts.append(f"🔧 {' | '.join(features)}")
            
            return '\n'.join(header_parts)
            
        except Exception as e:
            self.logger.debug(f"Error formatting header: {str(e)}")
            return "🚀 Crypto Analysis"
    
    def _format_market_summary(self, summary: SignalSummary, analysis_info: Optional[Dict] = None) -> str:
        """Format market summary section"""
        try:
            summary_parts = []
            
            # Section header
            timeframe_emoji = self.emojis.get(summary.timeframe, '📊')
            timeframe_desc = self._get_timeframe_description(summary.timeframe)
            
            if self.bold_headers:
                summary_parts.append(f"<b>📊 Market Summary ({summary.timeframe} - {timeframe_desc}):</b>")
            else:
                summary_parts.append(f"📊 Market Summary ({summary.timeframe} - {timeframe_desc}):")
            
            # Signal statistics
            stats_lines = []
            stats_lines.append(f"- Analyzed: {analysis_info.get('symbols_analyzed', 200)} top cryptocurrencies")
            stats_lines.append(f"- Strong Signals: {summary.strong_signals}")
            stats_lines.append(f"- High Confidence: {summary.strong_signals}/{summary.total_signals}")
            
            # Signal distribution
            distribution_parts = []
            if summary.bullish_count > 0:
                distribution_parts.append(f"🟢 {summary.bullish_count} LONG")
            if summary.bearish_count > 0:
                distribution_parts.append(f"🔴 {summary.bearish_count} SHORT")
            
            if distribution_parts:
                stats_lines.append(f"- Distribution: {' | '.join(distribution_parts)}")
            
            # Average confidence and ML info
            if summary.avg_confidence > 0:
                confidence_emoji = '🔥' if summary.avg_confidence > 80 else '💎' if summary.avg_confidence > 70 else '📊'
                stats_lines.append(f"- {confidence_emoji} Avg Confidence: {summary.avg_confidence:.1f}%")
            
            # Enhanced analysis summary
            if analysis_info:
                enhanced_parts = []
                
                # AI predictions
                if analysis_info.get('ml_enabled') and analysis_info.get('avg_ml_prediction'):
                    ml_score = analysis_info['avg_ml_prediction']
                    enhanced_parts.append(f"🤖 AI Predictions: {ml_score:.0f}% avg success rate")
                
                # News sentiment
                if analysis_info.get('news_enabled') and analysis_info.get('overall_news_sentiment') is not None:
                    sentiment = analysis_info['overall_news_sentiment']
                    sentiment_emoji = '📈' if sentiment > 0.1 else '📉' if sentiment < -0.1 else '📊'
                    sentiment_text = 'Bullish' if sentiment > 0.1 else 'Bearish' if sentiment < -0.1 else 'Neutral'
                    enhanced_parts.append(f"{sentiment_emoji} News Sentiment: {sentiment_text} ({sentiment:+.0%})")
                
                # Pattern analysis
                if analysis_info.get('patterns_enabled') and analysis_info.get('patterns_found'):
                    enhanced_parts.append(f"📊 Chart Patterns: {analysis_info['patterns_found']} detected")
                
                if enhanced_parts:
                    stats_lines.extend([f"- {part}" for part in enhanced_parts])
            
            summary_parts.extend(stats_lines)
            
            # Market regime/bias
            if summary.market_bias and summary.market_bias != 'NEUTRAL':
                bias_emoji = '🐂' if summary.market_bias == 'BULLISH' else '🐻' if summary.market_bias == 'BEARISH' else '🦘'
                bias_desc = self._get_market_bias_description(summary.market_bias)
                summary_parts.append(f"- {bias_emoji} Market Regime: {summary.market_bias} - {bias_desc}")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            self.logger.debug(f"Error formatting market summary: {str(e)}")
            return "📊 Market Summary: Analysis in progress..."
    
    def _format_top_signals(self, signals: List[Dict]) -> str:
        """Format top trading opportunities"""
        try:
            if not signals:
                return "💼 No high-quality signals found at this time."
            
            signal_parts = []
            
            # Section header
            if self.bold_headers:
                signal_parts.append("<b>🏆 Top Trading Opportunities:</b>")
            else:
                signal_parts.append("🏆 Top Trading Opportunities:")
            
            # Format each signal
            for i, signal in enumerate(signals, 1):
                signal_text = self._format_compact_signal(signal, i)
                signal_parts.append(signal_text)
            
            return '\n\n'.join(signal_parts)
            
        except Exception as e:
            self.logger.debug(f"Error formatting top signals: {str(e)}")
            return "🏆 Top signals processing..."
    
    def _format_compact_signal(self, signal: Dict, index: int) -> str:
        """Format a compact version of a signal for summary"""
        try:
            lines = []
            
            # Signal header with index
            signal_type = signal.get('type', 'UNKNOWN').upper()
            symbol = signal.get('symbol', 'UNKNOWN')
            signal_emoji = self.emojis.get(signal_type, '🟡')
            
            # Risk level color
            risk_level = signal.get('risk_level', 'MEDIUM')
            risk_emoji = self.emojis.get(risk_level, '🟡')
            
            lines.append(f"{index}. {signal_emoji} <b>{symbol} {signal_type}</b> {risk_emoji}")
            
            # Key metrics in one line
            strength = signal.get('signal_strength', 0)
            rsi = signal.get('indicators', {}).get('rsi', 0)
            
            strength_emoji = '🔥' if strength > 80 else '💎' if strength > 70 else '📊'
            
            # Compact metrics line
            metrics_parts = []
            metrics_parts.append(f"{strength_emoji} Signal Strength: {strength:.0f}%")
            
            if rsi:
                rsi_desc = 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'
                metrics_parts.append(f"RSI: {rsi:.1f} ({rsi_desc})")
            
            lines.append(f"   {' | '.join(metrics_parts)}")
            
            # Timeframe and duration
            timeframe = signal.get('timeframe', '1h')
            timeframe_config = config.EXTENDED_TIMEFRAMES.get(timeframe, {})
            duration_desc = timeframe_config.get('description', 'Short-term')
            
            timeframe_emoji = self.emojis.get(timeframe, '📊')
            lines.append(f"   {timeframe_emoji} Timeframe: {duration_desc} | Duration: {self._get_signal_duration(timeframe)}")
            
            # Enhanced features summary
            enhanced_parts = []
            
            # ML prediction
            if signal.get('ml_prediction'):
                ml_prob = signal['ml_prediction'].get('success_probability', 0) * 100
                if ml_prob > 75:
                    enhanced_parts.append(f"🤖 AI: {ml_prob:.0f}% success")
            
            # News impact
            if signal.get('news_sentiment'):
                news = signal['news_sentiment']
                if abs(news.get('overall_sentiment', 0)) > 0.1:
                    sentiment_emoji = '📈' if news['overall_sentiment'] > 0 else '📉'
                    enhanced_parts.append(f"{sentiment_emoji} News: {news['impact_level']} impact")
            
            # Pattern detection
            if signal.get('patterns') and len(signal['patterns']) > 0:
                pattern = signal['patterns'][0]
                pattern_name = self.pattern_descriptions.get(pattern.get('pattern_type', ''), 'Pattern')
                enhanced_parts.append(f"📊 {pattern_name}")
            
            if enhanced_parts:
                lines.append(f"   {' | '.join(enhanced_parts)}")
            
            # Entry and targets
            entry_price = signal.get('entry_price', 0)
            targets = signal.get('targets', [])
            stop_loss = signal.get('stop_loss', 0)
            
            if entry_price and targets:
                target_str = f"${targets[0]:,.4f}"
                if len(targets) > 1:
                    target_str += f" | ${targets[1]:,.4f}"
                
                levels_line = f"   💰 Entry: ${entry_price:,.4f} | 🎯 Targets: {target_str}"
                if stop_loss:
                    levels_line += f" | 🛑 SL: ${stop_loss:,.4f}"
                
                lines.append(levels_line)
            
            # Risk/reward and confluence
            info_parts = []
            
            rr_ratio = signal.get('risk_reward_ratio', 0)
            if rr_ratio > 0:
                info_parts.append(f"R/R: 1:{rr_ratio:.1f}")
            
            confluence_count = signal.get('confluence_count', 0)
            if confluence_count > 1:
                info_parts.append(f"Confluence: {confluence_count} indicators")
            
            volume_confirmed = signal.get('volume_confirmed', False)
            if volume_confirmed:
                info_parts.append("Volume ✅")
            
            if info_parts:
                lines.append(f"   📊 {' | '.join(info_parts)}")
            
            # Reasoning (compact)
            reasons = signal.get('reasons', [])
            if reasons:
                reason_text = ', '.join(reasons[:2])  # Show first 2 reasons
                if len(reasons) > 2:
                    reason_text += f" (+{len(reasons)-2} more)"
                lines.append(f"   💡 Reasoning: {reason_text}")
            
            # Validity period
            valid_until = signal.get('valid_until')
            if valid_until:
                if isinstance(valid_until, str):
                    lines.append(f"   ⏰ Valid Until: {valid_until}")
                elif isinstance(valid_until, datetime):
                    lines.append(f"   ⏰ Valid Until: {valid_until.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Signal ID for tracking
            signal_id = signal.get('signal_id')
            if signal_id:
                short_id = signal_id[-8:] if len(signal_id) > 8 else signal_id
                lines.append(f"   🆔 Signal ID: {short_id}")
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.debug(f"Error formatting compact signal: {str(e)}")
            return f"{index}. Signal formatting error"
    
    # =============================================================================
    # 🎯 INDIVIDUAL SIGNAL FORMATTING
    # =============================================================================
    
    def _format_signal_header(self, signal: Dict) -> str:
        """Format individual signal header"""
        try:
            signal_type = signal.get('type', 'UNKNOWN').upper()
            symbol = signal.get('symbol', 'UNKNOWN')
            risk_level = signal.get('risk_level', 'MEDIUM')
            
            signal_emoji = self.emojis.get(signal_type, '🟡')
            risk_emoji = self.emojis.get(risk_level, '🟡')
            
            if self.bold_headers:
                return f"{signal_emoji} <b>{symbol} {signal_type}</b> {risk_emoji}"
            else:
                return f"{signal_emoji} {symbol} {signal_type} {risk_emoji}"
                
        except Exception as e:
            self.logger.debug(f"Error formatting signal header: {str(e)}")
            return "📊 Trading Signal"
    
    def _format_signal_main_info(self, signal: Dict) -> str:
        """Format main signal information"""
        try:
            lines = []
            
            # Signal strength and RSI
            strength = signal.get('signal_strength', 0)
            indicators = signal.get('indicators', {})
            rsi = indicators.get('rsi', 0)
            
            strength_emoji = '🔥' if strength > 80 else '💎' if strength > 70 else '📊'
            
            if rsi:
                rsi_sentiment = 'Bullish sentiment' if rsi < 40 else 'Bearish sentiment' if rsi > 60 else 'Neutral sentiment'
                lines.append(f"{strength_emoji} Signal Strength: {strength:.0f}% | RSI: {rsi:.1f} ({rsi_sentiment})")
            else:
                lines.append(f"{strength_emoji} Signal Strength: {strength:.0f}%")
            
            # Timeframe and leverage info
            timeframe = signal.get('timeframe', '1h')
            timeframe_config = config.EXTENDED_TIMEFRAMES.get(timeframe, {})
            
            timeframe_emoji = self.emojis.get(timeframe, '📊')
            target_type = timeframe_config.get('target_type', 'swing')
            duration = self._get_signal_duration(timeframe)
            
            lines.append(f"{timeframe_emoji} Timeframe: {target_type.title()} | Duration: {duration}")
            
            # Leverage and risk info
            risk_level = signal.get('risk_level', 'MEDIUM')
            leverage_range = self._get_leverage_range(risk_level)
            position_size = self._get_position_size(risk_level)
            
            lines.append(f"🎚️ Leverage: {leverage_range} (Recommended: {self._get_recommended_leverage(risk_level)})")
            lines.append(f"📊 Risk Level: {risk_level} | Position Size: {position_size}% capital")
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.debug(f"Error formatting signal main info: {str(e)}")
            return "📊 Signal Information"
    
    def _format_signal_levels(self, signal: Dict) -> str:
        """Format entry, targets, and stop loss levels"""
        try:
            lines = []
            
            # Entry zone
            entry_min = signal.get('entry_min', 0)
            entry_max = signal.get('entry_max', 0)
            entry_price = signal.get('entry_price', 0)
            
            if entry_min and entry_max and entry_min != entry_max:
                lines.append(f"💰 Entry Zone: ${entry_min:,.4f} - ${entry_max:,.4f}")
            elif entry_price:
                lines.append(f"💰 Entry Price: ${entry_price:,.4f}")
            
            # Targets
            targets = signal.get('targets', [])
            if targets:
                if len(targets) == 1:
                    lines.append(f"🎯 Target: ${targets[0]:,.4f}")
                else:
                    target_strs = [f"${target:,.4f}" for target in targets[:3]]  # Max 3 targets
                    lines.append(f"🎯 Targets: {' | '.join(target_strs)}")
            
            # Stop loss
            stop_loss = signal.get('stop_loss', 0)
            if stop_loss:
                lines.append(f"🛑 Stop Loss: ${stop_loss:,.4f}")
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.debug(f"Error formatting signal levels: {str(e)}")
            return "💰 Price levels processing..."
    
    def _format_technical_analysis(self, signal: Dict) -> str:
        """Format technical analysis section"""
        try:
            lines = []
            
            # Section header
            if self.bold_headers:
                lines.append("<b>📈 Technical Analysis:</b>")
            else:
                lines.append("📈 Technical Analysis:")
            
            # Technical indicators
            indicators = signal.get('indicators', {})
            if indicators:
                tech_lines = []
                
                # Trend and volume
                trend_strength = indicators.get('trend_strength', 0)
                volume_ratio = indicators.get('volume_ratio', 1)
                
                trend_desc = 'Strong Uptrend' if trend_strength > 0.05 else 'Strong Downtrend' if trend_strength < -0.05 else 'Sideways'
                volume_desc = f"{volume_ratio:.1f}x avg" if volume_ratio != 1 else "Normal"
                
                tech_lines.append(f"   • Trend: {trend_desc} | Volume: {volume_desc}")
                
                # Confluence
                confluence_count = signal.get('confluence_count', 0)
                if confluence_count > 0:
                    tech_lines.append(f"   • Confluence: {confluence_count} indicators aligned")
                
                # Risk/reward
                rr_ratio = signal.get('risk_reward_ratio', 0)
                if rr_ratio > 0:
                    tech_lines.append(f"   • R/R Ratio: 1:{rr_ratio:.1f}")
                
                lines.extend(tech_lines)
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.debug(f"Error formatting technical analysis: {str(e)}")
            return "📈 Technical Analysis: Processing..."
    
    def _format_enhanced_features(self, enhanced_info: Dict) -> str:
        """Format enhanced features (ML, News, Patterns)"""
        try:
            sections = []
            
            # ML Prediction
            ml_prediction = enhanced_info.get('ml_prediction')
            if ml_prediction:
                ml_section = self._format_ml_section(ml_prediction)
                if ml_section:
                    sections.append(ml_section)
            
            # News Sentiment
            news_sentiment = enhanced_info.get('news_sentiment')
            if news_sentiment:
                news_section = self._format_news_section(news_sentiment)
                if news_section:
                    sections.append(news_section)
            
            # Pattern Recognition
            patterns = enhanced_info.get('patterns', [])
            if patterns:
                pattern_section = self._format_pattern_section(patterns)
                if pattern_section:
                    sections.append(pattern_section)
            
            return '\n\n'.join(sections) if sections else ""
            
        except Exception as e:
            self.logger.debug(f"Error formatting enhanced features: {str(e)}")
            return ""
    
    def _format_ml_section(self, ml_prediction: Dict) -> str:
        """Format ML prediction section"""
        try:
            lines = []
            
            success_prob = ml_prediction.get('success_probability', 0) * 100
            confidence = ml_prediction.get('confidence', 0) * 100
            recommendation = ml_prediction.get('recommendation', 'NEUTRAL')
            
            if success_prob > 0:
                if self.bold_headers:
                    lines.append("<b>🤖 AI Prediction:</b>")
                else:
                    lines.append("🤖 AI Prediction:")
                
                confidence_emoji = '🔥' if success_prob > 80 else '💎' if success_prob > 70 else '📊'
                lines.append(f"   • {confidence_emoji} Success Probability: {success_prob:.0f}%")
                lines.append(f"   • Confidence: {confidence:.0f}% | Recommendation: {recommendation}")
                
                # Feature importance (top 3)
                feature_importance = ml_prediction.get('feature_importance', {})
                if feature_importance:
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    top_features = [f"{k}: {v:.2f}" for k, v in sorted_features[:3]]
                    lines.append(f"   • Key Factors: {', '.join(top_features)}")
            
            return '\n'.join(lines) if lines else ""
            
        except Exception as e:
            self.logger.debug(f"Error formatting ML section: {str(e)}")
            return ""
    
    def _format_news_section(self, news_sentiment: Dict) -> str:
        """Format news sentiment section"""
        try:
            lines = []
            
            overall_sentiment = news_sentiment.get('overall_sentiment', 0)
            news_count = news_sentiment.get('news_count', 0)
            impact_level = news_sentiment.get('impact_level', 'LOW')
            confidence = news_sentiment.get('confidence', 0)
            
            if news_count > 0:
                if self.bold_headers:
                    lines.append("<b>📰 News Impact:</b>")
                else:
                    lines.append("📰 News Impact:")
                
                # Sentiment description
                if overall_sentiment > 0.1:
                    sentiment_desc = f"Bullish (+{overall_sentiment:.0%})"
                    sentiment_emoji = '📈'
                elif overall_sentiment < -0.1:
                    sentiment_desc = f"Bearish ({overall_sentiment:.0%})"
                    sentiment_emoji = '📉'
                else:
                    sentiment_desc = "Neutral"
                    sentiment_emoji = '📊'
                
                lines.append(f"   • {sentiment_emoji} Sentiment: {sentiment_desc} | Impact: {impact_level}")
                lines.append(f"   • News Articles: {news_count} | Confidence: {confidence:.0%}")
                
                # Key events
                key_events = news_sentiment.get('key_events', [])
                if key_events and len(key_events) > 0:
                    event = key_events[0][:50] + "..." if len(key_events[0]) > 50 else key_events[0]
                    lines.append(f"   • Latest: {event}")
            
            return '\n'.join(lines) if lines else ""
            
        except Exception as e:
            self.logger.debug(f"Error formatting news section: {str(e)}")
            return ""
    
    def _format_pattern_section(self, patterns: List[Dict]) -> str:
        """Format pattern recognition section"""
        try:
            if not patterns:
                return ""
            
            lines = []
            
            if self.bold_headers:
                lines.append("<b>📊 Chart Patterns:</b>")
            else:
                lines.append("📊 Chart Patterns:")
            
            # Show top pattern
            top_pattern = patterns[0]
            pattern_type = top_pattern.get('pattern_type', 'UNKNOWN')
            confidence = top_pattern.get('confidence', 0) * 100
            direction = top_pattern.get('signal_direction', 'NEUTRAL')
            
            pattern_name = self.pattern_descriptions.get(pattern_type, pattern_type)
            
            direction_emoji = '🟢' if direction == 'BULLISH' else '🔴' if direction == 'BEARISH' else '🟡'
            confidence_emoji = '🔥' if confidence > 80 else '💎' if confidence > 70 else '📊'
            
            lines.append(f"   • {direction_emoji} {pattern_name}: {confidence:.0f}% confidence")
            
            # Pattern details
            completion = top_pattern.get('pattern_completion', 0) * 100
            if completion > 0:
                lines.append(f"   • Completion: {completion:.0f}% | Breakout Probability: {top_pattern.get('breakout_probability', 0)*100:.0f}%")
            
            # Additional patterns count
            if len(patterns) > 1:
                lines.append(f"   • Additional Patterns: {len(patterns)-1} detected")
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.debug(f"Error formatting pattern section: {str(e)}")
            return ""
    
    def _format_signal_footer(self, signal: Dict) -> str:
        """Format signal footer with metadata"""
        try:
            lines = []
            
            # Reasoning
            reasons = signal.get('reasons', [])
            if reasons:
                reasoning_text = ', '.join(reasons[:3])  # Show first 3 reasons
                if len(reasons) > 3:
                    reasoning_text += f" (+{len(reasons)-3} more)"
                lines.append(f"💡 Reasoning: {reasoning_text}")
            
            # Market context
            market_regime = signal.get('market_context', '')
            if market_regime:
                lines.append(f"📊 Market Context: {market_regime}")
            
            # Validity
            valid_until = signal.get('valid_until')
            if valid_until:
                if isinstance(valid_until, str):
                    lines.append(f"⏰ Valid Until: {valid_until}")
                elif isinstance(valid_until, datetime):
                    lines.append(f"⏰ Valid Until: {valid_until.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Signal ID
            signal_id = signal.get('signal_id')
            if signal_id:
                short_id = signal_id[-12:] if len(signal_id) > 12 else signal_id
                lines.append(f"🆔 Signal ID: {short_id}")
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.debug(f"Error formatting signal footer: {str(e)}")
            return ""
    
    def _format_analysis_footer(self, analysis_info: Optional[Dict] = None) -> str:
        """Format analysis footer with session info"""
        try:
            lines = []
            
            if analysis_info:
                # Performance stats
                if analysis_info.get('analysis_duration'):
                    duration = analysis_info['analysis_duration']
                    lines.append(f"⚡ Analysis completed in {duration:.1f}s")
                
                # Session info
                session_id = analysis_info.get('session_id')
                if session_id:
                    short_session = session_id[-8:] if len(session_id) > 8 else session_id
                    lines.append(f"🔍 Session: {short_session}")
                
                # Feature status
                feature_status = []
                if analysis_info.get('ml_enabled'):
                    feature_status.append("🤖 AI")
                if analysis_info.get('news_enabled'):
                    feature_status.append("📰 News")
                if analysis_info.get('patterns_enabled'):
                    feature_status.append("📊 Patterns")
                
                if feature_status:
                    lines.append(f"🔧 Enhanced Features: {' | '.join(feature_status)}")
            
            # Trading session indicator
            current_hour = datetime.now().hour
            if 0 <= current_hour < 8:
                session = "🌙 Asian Session"
            elif 8 <= current_hour < 16:
                session = "🌅 European Session"
            elif 16 <= current_hour < 24:
                session = "🌇 American Session"
            else:
                session = "🌍 Global Session"
            
            lines.append(f"📍 {session}")
            
            return '\n'.join(lines) if lines else ""
            
        except Exception as e:
            self.logger.debug(f"Error formatting analysis footer: {str(e)}")
            return ""
    
    # =============================================================================
    # 🛠️ UTILITY METHODS
    # =============================================================================
    
    def _create_signal_summary(self, signals: List[Dict], timeframe: str) -> SignalSummary:
        """Create summary statistics from signals"""
        try:
            total_signals = len(signals)
            strong_signals = len([s for s in signals if s.get('signal_strength', 0) >= 75])
            
            bullish_count = len([s for s in signals if s.get('type', '').upper() == 'LONG'])
            bearish_count = len([s for s in signals if s.get('type', '').upper() == 'SHORT'])
            
            # Average confidence
            strengths = [s.get('signal_strength', 0) for s in signals]
            avg_confidence = sum(strengths) / len(strengths) if strengths else 0
            
            # Market bias
            if bullish_count > bearish_count * 1.5:
                market_bias = 'BULLISH'
            elif bearish_count > bullish_count * 1.5:
                market_bias = 'BEARISH'
            else:
                market_bias = 'NEUTRAL'
            
            # Top opportunities
            sorted_signals = sorted(signals, key=lambda x: x.get('signal_strength', 0), reverse=True)
            top_opportunities = [f"{s.get('symbol', '')} {s.get('type', '').upper()}" 
                               for s in sorted_signals[:3]]
            
            return SignalSummary(
                total_signals=total_signals,
                strong_signals=strong_signals,
                bullish_count=bullish_count,
                bearish_count=bearish_count,
                avg_confidence=avg_confidence,
                timeframe=timeframe,
                market_bias=market_bias,
                top_opportunities=top_opportunities
            )
            
        except Exception as e:
            self.logger.debug(f"Error creating signal summary: {str(e)}")
            return SignalSummary(0, 0, 0, 0, 0, timeframe, 'NEUTRAL', [])
    
    def _get_timeframe_description(self, timeframe: str) -> str:
        """Get human-readable timeframe description"""
        timeframe_config = config.EXTENDED_TIMEFRAMES.get(timeframe, {})
        return timeframe_config.get('description', 'Unknown timeframe')
    
    def _get_signal_duration(self, timeframe: str) -> str:
        """Get signal duration description"""
        timeframe_config = config.EXTENDED_TIMEFRAMES.get(timeframe, {})
        duration_hours = timeframe_config.get('duration_hours', 8)
        
        if duration_hours < 24:
            return f"{duration_hours} hours"
        elif duration_hours < 168:  # Less than a week
            days = duration_hours // 24
            return f"{days} days"
        else:
            weeks = duration_hours // 168
            return f"{weeks} weeks"
    
    def _get_leverage_range(self, risk_level: str) -> str:
        """Get leverage range for risk level"""
        leverage_ranges = {
            'LOW': '2-5x',
            'MEDIUM': '5-15x', 
            'HIGH': '10-25x'
        }
        return leverage_ranges.get(risk_level, '5-15x')
    
    def _get_recommended_leverage(self, risk_level: str) -> str:
        """Get recommended leverage for risk level"""
        recommended = {
            'LOW': '3x',
            'MEDIUM': '10x',
            'HIGH': '15x'
        }
        return recommended.get(risk_level, '10x')
    
    def _get_position_size(self, risk_level: str) -> str:
        """Get position size percentage for risk level"""
        position_sizes = {
            'LOW': '2-3',
            'MEDIUM': '3-5',
            'HIGH': '5-8'
        }
        return position_sizes.get(risk_level, '3-5')
    
    def _get_market_bias_description(self, bias: str) -> str:
        """Get market bias description"""
        descriptions = {
            'BULLISH': 'Strong uptrend - Favor LONG positions',
            'BEARISH': 'Strong downtrend - Favor SHORT positions', 
            'NEUTRAL': 'Sideways market - Mixed signals'
        }
        return descriptions.get(bias, 'Market direction unclear')
    
    def _calculate_signal_priority(self, signal: Dict) -> int:
        """Calculate signal priority for message ordering"""
        try:
            strength = signal.get('signal_strength', 0)
            confluence = signal.get('confluence_count', 0)
            
            # High priority: Strong signals with confluence
            if strength >= 80 and confluence >= 3:
                return 1
            # Medium priority: Good signals
            elif strength >= 70:
                return 2
            # Low priority: Weak signals
            else:
                return 3
                
        except:
            return 2
    
    def _ensure_message_length(self, content: str) -> Tuple[str, bool]:
        """Ensure message fits within Telegram limits"""
        try:
            if len(content) <= self.max_length:
                return content, False
            
            # Truncate intelligently
            lines = content.split('\n')
            truncated_lines = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) + 1 <= self.max_length - 50:  # Leave room for truncation notice
                    truncated_lines.append(line)
                    current_length += len(line) + 1
                else:
                    break
            
            truncated_content = '\n'.join(truncated_lines)
            truncated_content += '\n\n⚠️ Message truncated due to length limits'
            
            return truncated_content, True
            
        except Exception as e:
            self.logger.debug(f"Error ensuring message length: {str(e)}")
            return content[:self.max_length], True
    
    def _format_no_signals_message(self, timeframe: str) -> FormattedMessage:
        """Format message when no signals are found"""
        try:
            content_parts = []
            
            # Header
            if config.TIMEZONE:
                timestamp = datetime.now(config.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S (%Z)')
            else:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S (UTC)')
            
            if self.bold_headers:
                content_parts.append("<b>🚀 Crypto Futures Analysis V3.0 Ultimate</b>")
            else:
                content_parts.append("🚀 Crypto Futures Analysis V3.0 Ultimate")
            
            content_parts.append(f"📅 {timestamp}")
            
            if hasattr(config, 'TELEGRAM_USER') and config.TELEGRAM_USER:
                content_parts.append(f"👤 {config.TELEGRAM_USER}")
            
            # No signals message
            timeframe_desc = self._get_timeframe_description(timeframe)
            
            if self.bold_headers:
                content_parts.append(f"<b>📊 Market Analysis ({timeframe} - {timeframe_desc}):</b>")
            else:
                content_parts.append(f"📊 Market Analysis ({timeframe} - {timeframe_desc}):")
            
            content_parts.append("🔍 No high-quality trading signals detected at this time.")
            content_parts.append("📈 Market conditions may be:")
            content_parts.append("   • Low volatility period")
            content_parts.append("   • Consolidation phase")
            content_parts.append("   • Awaiting market catalyst")
            
            content_parts.append("⏳ Next analysis scheduled in 2 hours.")
            content_parts.append("🔔 Strong signals will be sent immediately when detected.")
            
            full_content = '\n\n'.join(content_parts)
            
            return FormattedMessage(
                content=full_content,
                message_type='no_signals',
                length=len(full_content),
                is_valid=True,
                truncated=False,
                priority=3
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting no signals message: {str(e)}")
            return self._format_error_message("No signals analysis error")
    
    def _format_error_message(self, error_description: str) -> FormattedMessage:
        """Format error message"""
        try:
            content = f"⚠️ Analysis Error\n\n"
            content += f"🔧 {error_description}\n"
            content += f"🔄 Retrying analysis...\n"
            content += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return FormattedMessage(
                content=content,
                message_type='error',
                length=len(content),
                is_valid=True,
                truncated=False,
                priority=3
            )
            
        except:
            return FormattedMessage(
                content="⚠️ System Error",
                message_type='error',
                length=15,
                is_valid=True,
                truncated=False,
                priority=3
            )
    
    # =============================================================================
    # 📱 SPECIALIZED FORMATTING METHODS
    # =============================================================================
    
    def format_test_message(self) -> FormattedMessage:
        """Format test message for bot verification"""
        try:
            content_parts = []
            
            content_parts.append("🧪 <b>Bot Test Message</b>")
            content_parts.append("✅ Crypto Analysis Bot V3.0 Ultimate is online!")
            
            if config.TIMEZONE:
                timestamp = datetime.now(config.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S (%Z)')
            else:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S (UTC)')
            
            content_parts.append(f"📅 {timestamp}")
            
            content_parts.append("🔧 <b>System Status:</b>")
            content_parts.append("   ✅ Signal Formatter: Active")
            content_parts.append("   ✅ Technical Analysis: Ready")
            content_parts.append("   ✅ Enhanced Features: Loaded")
            
            content_parts.append("🎯 <b>Features Enabled:</b>")
            content_parts.append("   🤖 AI Predictions")
            content_parts.append("   📰 News Sentiment Analysis")
            content_parts.append("   📊 Chart Pattern Recognition")
            content_parts.append("   📈 Advanced Backtesting")
            content_parts.append("   ⏰ Extended Timeframes (24h+)")
            
            content_parts.append("🚀 Ready for professional crypto analysis!")
            
            full_content = '\n'.join(content_parts)
            
            return FormattedMessage(
                content=full_content,
                message_type='test',
                length=len(full_content),
                is_valid=True,
                truncated=False,
                priority=2
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting test message: {str(e)}")
            return self._format_error_message("Test message error")
    
    def format_performance_update(self, performance_stats: Dict) -> FormattedMessage:
        """Format performance update message"""
        try:
            content_parts = []
            
            content_parts.append("📊 <b>Performance Update</b>")
            
            # Analysis stats
            analysis_count = performance_stats.get('analysis_count', 0)
            successful_signals = performance_stats.get('successful_signals', 0)
            success_rate = performance_stats.get('success_rate', 0) * 100
            
            content_parts.append(f"🎯 <b>Analysis Statistics:</b>")
            content_parts.append(f"   📈 Total Analyses: {analysis_count}")
            content_parts.append(f"   ✅ Successful Signals: {successful_signals}")
            content_parts.append(f"   📊 Success Rate: {success_rate:.1f}%")
            
            # Uptime
            uptime = performance_stats.get('uptime_formatted', 'Unknown')
            content_parts.append(f"   ⏰ Uptime: {uptime}")
            
            # Features status
            features = performance_stats.get('features_enabled', {})
            feature_status = []
            
            if features.get('news_analyzer'):
                feature_status.append("📰 News")
            if features.get('ml_predictor'):
                feature_status.append("🤖 AI")
            if features.get('pattern_analyzer'):
                feature_status.append("📊 Patterns")
            if features.get('backtest_engine'):
                feature_status.append("📈 Backtest")
            
            if feature_status:
                content_parts.append(f"🔧 <b>Active Features:</b> {' | '.join(feature_status)}")
            
            # Market regime
            market_regime = performance_stats.get('market_regime', 'Unknown')
            if market_regime != 'Unknown':
                regime_emoji = '🐂' if market_regime == 'bull' else '🐻' if market_regime == 'bear' else '🦘'
                content_parts.append(f"📊 <b>Market Regime:</b> {regime_emoji} {market_regime.title()}")
            
            full_content = '\n'.join(content_parts)
            
            return FormattedMessage(
                content=full_content,
                message_type='performance',
                length=len(full_content),
                is_valid=True,
                truncated=False,
                priority=2
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting performance update: {str(e)}")
            return self._format_error_message("Performance update error")

# =============================================================================
# 🧪 TESTING FUNCTION
# =============================================================================

async def test_signal_formatter():
    """Test function for the signal formatter"""
    print("🧪 Testing SignalFormatter V3.0 Ultimate...")
    
    formatter = SignalFormatter()
    
    try:
        # Test signal data
        test_signal = {
            'symbol': 'BTC/USDT',
            'type': 'LONG',
            'signal_strength': 85,
            'timeframe': '1h',
            'risk_level': 'MEDIUM',
            'entry_price': 67250,
            'targets': [68500, 69850, 72100],
            'stop_loss': 65800,
            'risk_reward_ratio': 1.8,
            'confluence_count': 4,
            'volume_confirmed': True,
            'reasons': ['RSI oversold', 'MACD bullish crossover', 'Volume spike'],
            'indicators': {
                'rsi': 32.5,
                'trend_strength': 0.15,
                'volume_ratio': 1.8
            },
            'valid_until': '2025-06-21 16:45:30',
            'signal_id': 'btc_long_20250620_1234'
        }
        
        # Test enhanced info
        enhanced_info = {
            'ml_prediction': {
                'success_probability': 0.82,
                'confidence': 0.76,
                'recommendation': 'STRONG_BUY',
                'feature_importance': {
                    'rsi': 0.25,
                    'volume_ratio': 0.18,
                    'news_sentiment': 0.15
                }
            },
            'news_sentiment': {
                'overall_sentiment': 0.15,
                'news_count': 12,
                'impact_level': 'MEDIUM',
                'confidence': 0.8,
                'key_events': ['ETF approval news driving positive sentiment']
            },
            'patterns': [{
                'pattern_type': 'ASCENDING_TRIANGLE',
                'confidence': 0.85,
                'signal_direction': 'BULLISH',
                'pattern_completion': 0.8,
                'breakout_probability': 0.75
            }]
        }
        
        print("📊 Testing individual signal formatting...")
        
        # Test individual signal
        individual_signal = formatter.format_individual_signal(test_signal, enhanced_info)
        print(f"✅ Individual signal formatted: {individual_signal.length} characters")
        print(f"   Valid: {individual_signal.is_valid}")
        print(f"   Truncated: {individual_signal.truncated}")
        print(f"   Priority: {individual_signal.priority}")
        
        # Print first few lines
        lines = individual_signal.content.split('\n')
        for i, line in enumerate(lines[:5]):
            print(f"   {i+1}: {line}")
        
        print("\n📈 Testing signals summary...")
        
        # Test signals summary
        test_signals = [test_signal] * 3  # Create 3 similar signals
        analysis_info = {
            'symbols_analyzed': 200,
            'ml_enabled': True,
            'news_enabled': True,
            'patterns_enabled': True,
            'avg_ml_prediction': 82,
            'overall_news_sentiment': 0.15,
            'patterns_found': 5,
            'analysis_duration': 2.3,
            'session_id': 'session_12345678'
        }
        
        summary_message = formatter.format_signals_summary(test_signals, '1h', analysis_info)
        print(f"✅ Summary formatted: {summary_message.length} characters")
        print(f"   Valid: {summary_message.is_valid}")
        
        # Print summary preview
        summary_lines = summary_message.content.split('\n')
        for i, line in enumerate(summary_lines[:8]):
            print(f"   {i+1}: {line}")
        
        print("\n🧪 Testing special messages...")
        
        # Test test message
        test_msg = formatter.format_test_message()
        print(f"✅ Test message: {test_msg.length} characters")
        
        # Test no signals message
        no_signals_msg = formatter._format_no_signals_message('4h')
        print(f"✅ No signals message: {no_signals_msg.length} characters")
        
        # Test performance update
        perf_stats = {
            'analysis_count': 125,
            'successful_signals': 95,
            'success_rate': 0.76,
            'uptime_formatted': '2 days, 15:30:45',
            'features_enabled': {
                'news_analyzer': True,
                'ml_predictor': True,
                'pattern_analyzer': True,
                'backtest_engine': True
            },
            'market_regime': 'bull'
        }
        
        perf_msg = formatter.format_performance_update(perf_stats)
        print(f"✅ Performance update: {perf_msg.length} characters")
        
        print("🎉 SignalFormatter test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run test if executed directly
    import asyncio
    asyncio.run(test_signal_formatter())
