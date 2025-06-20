# config.py - Crypto Analysis Bot V3.0 Ultimate Configuration
import os
import pytz
from datetime import datetime

# =============================================================================
# 🌍 BASIC CONFIGURATION
# =============================================================================

# Timezone Configuration
TIMEZONE = pytz.timezone('Asia/Tehran')  # GMT+3:30
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_USER = '@MasoudHaddad69'

# Binance API Configuration (Optional)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET = os.getenv('BINANCE_SECRET')

# Basic Analysis Settings
CRYPTO_COUNT = int(os.getenv('CRYPTO_COUNT', 200))
SIGNAL_COUNT = int(os.getenv('SIGNAL_COUNT', 10))
MIN_SIGNAL_STRENGTH = int(os.getenv('MIN_SIGNAL_STRENGTH', 65))
ANALYSIS_INTERVAL_MINUTES = int(os.getenv('ANALYSIS_INTERVAL_MINUTES', 120))

# Request Settings
REQUEST_DELAY = float(os.getenv('REQUEST_DELAY', 0.2))  # Seconds between requests
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))  # Request timeout
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))  # Max retry attempts

# =============================================================================
# ⏰ EXTENDED TIMEFRAMES CONFIGURATION (24h+ Signals)
# =============================================================================

EXTENDED_TIMEFRAMES = {
    '5m': {
        'duration_hours': 1,
        'target_type': 'scalp',
        'min_strength': 75,
        'description': 'Ultra short-term scalping'
    },
    '15m': {
        'duration_hours': 2,
        'target_type': 'scalp',
        'min_strength': 70,
        'description': 'Short-term scalping'
    },
    '1h': {
        'duration_hours': 8,
        'target_type': 'intraday',
        'min_strength': 65,
        'description': 'Intraday trading'
    },
    '4h': {
        'duration_hours': 16,
        'target_type': 'swing',
        'min_strength': 60,
        'description': 'Short swing trading'
    },
    '1d': {
        'duration_hours': 48,      # 2 days ✅ 24h+
        'target_type': 'swing',
        'min_strength': 55,
        'description': 'Daily swing trading'
    },
    '3d': {
        'duration_hours': 168,     # 7 days ✅ 24h+
        'target_type': 'position',
        'min_strength': 50,
        'description': 'Weekly position trading'
    },
    '1w': {
        'duration_hours': 720,     # 30 days ✅ 24h+
        'target_type': 'long_term',
        'min_strength': 45,
        'description': 'Monthly position trading'
    }
}

# =============================================================================
# 📰 NEWS SENTIMENT ANALYSIS CONFIGURATION
# =============================================================================

NEWS_SENTIMENT_CONFIG = {
    'enabled': True,
    'api_sources': ['newsapi', 'cryptocompare', 'coindesk', 'cryptopanic'],
    'sentiment_weight': 0.2,  # 20% weight in final signal score
    'news_hours_lookback': 24,  # Look back 24 hours for news
    'min_news_count': 3,  # Minimum news articles for reliable sentiment
    'confidence_threshold': 0.3,  # Minimum confidence to use sentiment
    'impact_levels': ['LOW', 'MEDIUM', 'HIGH'],
    'cache_duration_minutes': 30  # Cache news for 30 minutes
}

# News API Keys (Optional - for enhanced news analysis)
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')
CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY')

# =============================================================================
# 🤖 MACHINE LEARNING CONFIGURATION
# =============================================================================

ML_CONFIG = {
    'enabled': True,
    'model_path': './models/',
    'retrain_interval_days': 7,  # Retrain models every 7 days
    'min_training_samples': 1000,  # Minimum samples for training
    'success_prediction_threshold': 0.75,  # 75% threshold for strong signals
    'feature_importance_threshold': 0.05,  # Minimum feature importance
    'cross_validation_folds': 5,  # K-fold cross validation
    'test_size_ratio': 0.2,  # 20% for testing
    'models': {
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'random_state': 42
        },
        'gradient_boosting': {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 8,
            'random_state': 42
        },
        'logistic_regression': {
            'C': 1.0,
            'solver': 'liblinear',
            'random_state': 42
        }
    },
    'ensemble_weights': {
        'random_forest': 0.4,
        'gradient_boosting': 0.35,
        'logistic_regression': 0.25
    }
}

# =============================================================================
# 📊 PATTERN RECOGNITION CONFIGURATION
# =============================================================================

PATTERN_RECOGNITION_CONFIG = {
    'enabled': True,
    'patterns': [
        'head_shoulders',
        'double_bottom', 
        'double_top',
        'triangle_ascending',
        'triangle_descending', 
        'triangle_symmetric',
        'flag',
        'pennant',
        'wedge_rising',
        'wedge_falling',
        'channel',
        'rectangle'
    ],
    'fibonacci_levels': True,
    'support_resistance': True,
    'pattern_weight': 0.15,  # 15% weight in final signal score
    'min_pattern_confidence': 0.6,  # Minimum confidence for pattern
    'pattern_completion_threshold': 0.7,  # Minimum completion percentage
    'lookback_periods': {
        'short_term': 50,   # 50 candles for short patterns
        'medium_term': 100, # 100 candles for medium patterns
        'long_term': 200    # 200 candles for long patterns
    }
}

# =============================================================================
# 📈 BACKTESTING CONFIGURATION
# =============================================================================

BACKTESTING_CONFIG = {
    'enabled': True,
    'historical_days': 30,  # Default backtest period
    'optimization_period': 7,  # Re-optimize every 7 days
    'min_trades_for_optimization': 50,  # Minimum trades to optimize
    'success_rate_threshold': 0.7,  # 70% success rate threshold
    'max_backtest_days': 90,  # Maximum backtest period
    'transaction_costs': {
        'maker_fee': 0.001,  # 0.1% maker fee
        'taker_fee': 0.001,  # 0.1% taker fee
        'slippage': 0.0005   # 0.05% slippage
    },
    'risk_metrics': {
        'max_drawdown_threshold': 0.2,  # 20% max drawdown
        'min_sharpe_ratio': 1.0,  # Minimum Sharpe ratio
        'min_profit_factor': 1.2   # Minimum profit factor
    }
}

# =============================================================================
# 🛡️ ADVANCED RISK MANAGEMENT CONFIGURATION
# =============================================================================

RISK_LEVELS = {
    'LOW': {
        'max_leverage': 5,
        'position_size': 2,     # % of capital
        'risk_reward': 2.0,     # Minimum R/R ratio
        'stop_loss_atr': 1.5,   # ATR multiplier for stop loss
        'max_concurrent': 2     # Max concurrent positions
    },
    'MEDIUM': {
        'max_leverage': 15,
        'position_size': 5,
        'risk_reward': 1.5,
        'stop_loss_atr': 2.0,
        'max_concurrent': 3
    },
    'HIGH': {
        'max_leverage': 25,
        'position_size': 10,
        'risk_reward': 1.2,
        'stop_loss_atr': 2.5,
        'max_concurrent': 2
    }
}

ADVANCED_RISK_CONFIG = {
    'market_regime_adjustment': True,  # Adjust based on market conditions
    'volatility_adjustment': True,     # Adjust based on volatility
    'correlation_analysis': True,      # Avoid correlated positions
    'portfolio_heat': 0.1,            # Max 10% portfolio at risk
    'max_concurrent_signals': 5,      # Max active signals
    'risk_per_trade': 0.02,           # Max 2% risk per trade
    'dynamic_position_sizing': True,   # Adjust size based on confidence
    'regime_adjustments': {
        'bull': {'leverage_multiplier': 1.2, 'confidence_boost': 0.1},
        'bear': {'leverage_multiplier': 0.8, 'confidence_boost': -0.1},
        'sideways': {'leverage_multiplier': 1.0, 'confidence_boost': 0.0},
        'volatile': {'leverage_multiplier': 0.6, 'confidence_boost': -0.2}
    }
}

# =============================================================================
# ⚡ PERFORMANCE OPTIMIZATION CONFIGURATION
# =============================================================================

PERFORMANCE_CONFIG = {
    'parallel_analysis': True,         # Enable parallel processing
    'max_workers': 10,                # Maximum worker threads
    'cache_duration_seconds': 300,    # 5 minutes cache
    'request_timeout': 30,            # Request timeout in seconds
    'retry_attempts': 3,              # Maximum retry attempts
    'batch_size': 20,                 # Batch size for processing
    'memory_optimization': True,      # Enable memory optimization
    'gc_interval': 100,              # Garbage collection interval
    'connection_pool_size': 20,       # HTTP connection pool size
    'enable_compression': True,       # Enable response compression
    'rate_limiting': {
        'enabled': True,
        'max_requests_per_minute': 60,
        'burst_allowance': 10
    }
}

# =============================================================================
# 📊 TECHNICAL ANALYSIS CONFIGURATION
# =============================================================================

# RSI Settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERSOLD = 20
RSI_EXTREME_OVERBOUGHT = 80

# MACD Settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands Settings
BB_PERIOD = 20
BB_DEVIATION = 2
BB_OVERSOLD = 0.1      # Below 10% of BB range
BB_OVERBOUGHT = 0.9    # Above 90% of BB range

# Moving Averages
SMA_SHORT = 20
SMA_LONG = 50
EMA_SHORT = 12
EMA_LONG = 26

# Stochastic Settings
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80

# ATR Settings
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0  # For stop loss calculation

# Volume Settings
VOLUME_SMA_PERIOD = 20
VOLUME_SPIKE_THRESHOLD = 1.5  # 1.5x average volume

# Volatility Settings
VOLATILITY_PERIOD = 20
HIGH_VOLATILITY_THRESHOLD = 0.05  # 5% daily volatility
LOW_VOLATILITY_THRESHOLD = 0.02   # 2% daily volatility

# =============================================================================
# 🎯 TARGET AND STOP LOSS CONFIGURATION
# =============================================================================

TARGET_MULTIPLIERS = {
    'LONG': {
        'target1': 0.015,  # 1.5%
        'target2': 0.025,  # 2.5%
        'target3': 0.04,   # 4%
        'stop_loss': -0.02 # -2%
    },
    'SHORT': {
        'target1': -0.015, # -1.5%
        'target2': -0.025, # -2.5%
        'target3': -0.04,  # -4%
        'stop_loss': 0.02  # +2%
    }
}

# Dynamic target adjustment based on timeframe
TIMEFRAME_TARGET_ADJUSTMENTS = {
    '5m': 0.5,   # 50% of base targets
    '15m': 0.7,  # 70% of base targets
    '1h': 1.0,   # 100% of base targets
    '4h': 1.5,   # 150% of base targets
    '1d': 2.0,   # 200% of base targets
    '3d': 3.0,   # 300% of base targets
    '1w': 4.0    # 400% of base targets
}

# =============================================================================
# 📱 TELEGRAM MESSAGE CONFIGURATION
# =============================================================================

TELEGRAM_CONFIG = {
    'max_message_length': 4000,      # Telegram limit is 4096
    'retry_attempts': 3,             # Retry failed messages
    'retry_delay': 5,                # Seconds between retries
    'parse_mode': 'HTML',            # HTML or Markdown
    'disable_web_page_preview': True,
    'notification_settings': {
        'send_summary': True,         # Send market summary
        'send_individual_signals': True,  # Send each signal
        'send_performance_updates': False,  # Weekly performance
        'send_error_notifications': False   # Error alerts
    },
    'message_formatting': {
        'use_emojis': True,
        'bold_headers': True,
        'compact_mode': False,        # Compact vs detailed format
        'include_charts': False       # Future: chart images
    }
}

# =============================================================================
# 🔧 SYSTEM CONFIGURATION
# =============================================================================

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',                 # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True,
    'log_file': 'crypto_analysis_v3.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'console_logging': True
}

# Database Configuration (for storing results)
DATABASE_CONFIG = {
    'enabled': False,                # Enable database storage
    'type': 'sqlite',               # sqlite, postgresql, mysql
    'file': 'crypto_analysis.db',   # For SQLite
    'host': 'localhost',            # For PostgreSQL/MySQL
    'port': 5432,                   # Database port
    'username': '',                 # Database username
    'password': '',                 # Database password
    'database': 'crypto_analysis'   # Database name
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'enabled': True,
    'metrics_retention_days': 30,   # Keep metrics for 30 days
    'performance_tracking': True,
    'error_tracking': True,
    'success_rate_tracking': True,
    'response_time_tracking': True
}

# =============================================================================
# 🌐 API ENDPOINTS CONFIGURATION
# =============================================================================

API_CONFIG = {
    'enable_cors': True,
    'cors_origins': ['*'],           # Allow all origins
    'rate_limiting': {
        'enabled': True,
        'max_requests_per_minute': 60
    },
    'authentication': {
        'enabled': False,            # API key authentication
        'api_key': os.getenv('API_KEY'),
        'required_for_analysis': False
    },
    'documentation': {
        'enabled': True,
        'swagger_ui': True,
        'redoc': True
    }
}

# =============================================================================
# 🔗 EXTERNAL SERVICES CONFIGURATION
# =============================================================================

# Exchange API Settings
EXCHANGE_CONFIG = {
    'primary': 'binance',
    'backup': 'okx',
    'testnet': False,
    'rate_limit': True,
    'timeout': 30000,  # 30 seconds
    'retry_attempts': 3,
    'supported_exchanges': ['binance', 'okx', 'bybit', 'kucoin']
}

# Social Media APIs (Future Enhancement)
SOCIAL_SENTIMENT_CONFIG = {
    'enabled': False,               # Will be enabled in future updates
    'twitter_api_key': os.getenv('TWITTER_API_KEY'),
    'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
    'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    'sentiment_sources': ['twitter', 'reddit', 'telegram'],
    'twitter_weight': 0.1,
    'reddit_weight': 0.05,
    'telegram_weight': 0.05,
    'min_mentions': 10              # Minimum mentions for sentiment
}

# Fear & Greed Index
FEAR_GREED_CONFIG = {
    'enabled': True,
    'api_url': 'https://api.alternative.me/fng/',
    'weight_in_analysis': 0.05,     # 5% weight
    'cache_duration_hours': 6       # Update every 6 hours
}

# =============================================================================
# 🛠️ DEVELOPMENT AND DEBUG CONFIGURATION
# =============================================================================

# Development Settings
DEBUG_CONFIG = {
    'enabled': os.getenv('FLASK_ENV') != 'production',
    'verbose_logging': False,
    'save_analysis_data': True,     # Save analysis for debugging
    'mock_trading': False,          # Use mock data for testing
    'test_mode': False,             # Enable test mode features
    'profiling': False,             # Enable performance profiling
    'memory_monitoring': True       # Monitor memory usage
}

# Test Configuration
TEST_CONFIG = {
    'mock_exchanges': True,         # Use mock exchange data in tests
    'test_symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
    'sample_data_file': './test_data/sample_ohlcv.json',
    'test_news_file': './test_data/sample_news.json'
}

# =============================================================================
# 📦 VERSION AND METADATA
# =============================================================================

BOT_VERSION = "3.0.0-Ultimate"
BOT_NAME = "Crypto Analysis Bot V3.0 Ultimate"
BOT_DESCRIPTION = "Advanced cryptocurrency analysis with ML, news sentiment, and pattern recognition"
BOT_AUTHOR = "Advanced Trading Systems"
BOT_LICENSE = "Private Use"

# Feature Flags
FEATURE_FLAGS = {
    'extended_timeframes': True,
    'news_sentiment': True,
    'machine_learning': True,
    'pattern_recognition': True,
    'backtesting': True,
    'social_sentiment': False,      # Future feature
    'portfolio_tracking': False,   # Future feature
    'automated_trading': False,    # Future feature
    'mobile_app': False,           # Future feature
    'web_dashboard': False         # Future feature
}

# =============================================================================
# 🔍 VALIDATION FUNCTIONS
# =============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required settings
    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN is required")
    
    if not TELEGRAM_CHAT_ID:
        errors.append("TELEGRAM_CHAT_ID is required")
    
    # Check timeframe settings
    if MIN_SIGNAL_STRENGTH < 0 or MIN_SIGNAL_STRENGTH > 100:
        errors.append("MIN_SIGNAL_STRENGTH must be between 0 and 100")
    
    # Check risk settings
    for risk_level, config in RISK_LEVELS.items():
        if config['max_leverage'] < 1 or config['max_leverage'] > 100:
            errors.append(f"Invalid max_leverage for {risk_level}")
    
    return errors

def get_config_summary():
    """Get a summary of current configuration"""
    return {
        'version': BOT_VERSION,
        'timezone': str(TIMEZONE),
        'crypto_count': CRYPTO_COUNT,
        'signal_count': SIGNAL_COUNT,
        'min_signal_strength': MIN_SIGNAL_STRENGTH,
        'extended_timeframes': list(EXTENDED_TIMEFRAMES.keys()),
        'features_enabled': {
            'news_sentiment': NEWS_SENTIMENT_CONFIG['enabled'],
            'machine_learning': ML_CONFIG['enabled'],
            'pattern_recognition': PATTERN_RECOGNITION_CONFIG['enabled'],
            'backtesting': BACKTESTING_CONFIG['enabled'],
            'performance_optimization': PERFORMANCE_CONFIG['parallel_analysis']
        },
        'risk_levels': list(RISK_LEVELS.keys()),
        'telegram_configured': bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        'binance_configured': bool(BINANCE_API_KEY and BINANCE_SECRET)
    }

# =============================================================================
# 🚀 INITIALIZATION
# =============================================================================

# Validate configuration on import
config_errors = validate_config()
if config_errors:
    print("⚠️ Configuration errors found:")
    for error in config_errors:
        print(f"   - {error}")
    print("Please fix these errors before running the bot.")

# Print configuration summary in debug mode
if DEBUG_CONFIG['enabled']:
    summary = get_config_summary()
    print(f"🔧 Bot Configuration Summary:")
    print(f"   Version: {summary['version']}")
    print(f"   Timezone: {summary['timezone']}")
    print(f"   Features: {summary['features_enabled']}")
    print(f"   Timeframes: {summary['extended_timeframes']}")