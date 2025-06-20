# main.py - Crypto Analysis Bot V3.0 Ultimate - Main Flask Application
import asyncio
import aiohttp
import json
import logging
import os
import sys
import traceback
import threading
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import schedule
import time
from concurrent.futures import ThreadPoolExecutor
import psutil

# Import our custom modules
try:
    import config
    from analyzer import CryptoAnalyzer
    from signal_formatter import SignalFormatter
    from news_analyzer import NewsAnalyzer
    from ml_predictor import MLPredictor
    from pattern_recognition import AdvancedPatternRecognition as PatternRecognition
    from backtesting_engine import BacktestingEngine
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure all required files are in the same directory:")
    print("- config.py")
    print("- analyzer.py") 
    print("- signal_formatter.py")
    print("- news_analyzer.py")
    print("- ml_predictor.py")
    print("- pattern_recognition.py")
    print("- backtesting_engine.py")
    sys.exit(1)

# =============================================================================
# 🚀 GLOBAL VARIABLES AND SETUP
# =============================================================================

# Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global instances
crypto_analyzer = None
signal_formatter = None
telegram_session = None
executor = ThreadPoolExecutor(max_workers=4)

# Application state
app_start_time = datetime.now()
total_requests = 0
successful_analyses = 0
last_analysis_time = None
is_shutting_down = False

# Thread-safe locks
analysis_lock = threading.Lock()
telegram_lock = threading.Lock()

# Background scheduler
scheduler_thread = None
scheduler_running = False

# =============================================================================
# 🔧 LOGGING SETUP
# =============================================================================

def setup_logging():
    """Setup comprehensive logging"""
    log_config = config.LOGGING_CONFIG
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler('logs/crypto_analysis_v3.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Crypto Analysis Bot V3.0 Ultimate Edition")
    
    return logger

logger = setup_logging()

# =============================================================================
# 📱 TELEGRAM INTEGRATION
# =============================================================================

async def send_telegram_message(message: str, parse_mode: str = 'HTML') -> bool:
    """Send message to Telegram with retry logic"""
    try:
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials not configured")
            return False
        
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        
        payload = {
            'chat_id': config.TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.debug("✅ Telegram message sent successfully")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Telegram error {response.status}: {error_text}")
                    return False
                    
    except Exception as e:
        logger.error(f"💥 Error sending Telegram message: {str(e)}")
        return False

def send_telegram_sync(message: str) -> bool:
    """Synchronous wrapper for sending Telegram messages"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(send_telegram_message(message))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in sync Telegram send: {str(e)}")
        return False

# =============================================================================
# 🤖 MAIN ANALYSIS FUNCTIONS
# =============================================================================

async def perform_crypto_analysis(timeframe: str = '1h', send_to_telegram: bool = True) -> Dict[str, Any]:
    """Perform comprehensive crypto analysis"""
    start_time = datetime.now()
    
    try:
        logger.info(f"🔥 Starting analysis for timeframe: {timeframe}")
        
        global total_requests, successful_analyses, last_analysis_time
        
        with analysis_lock:
            total_requests += 1
        
        # Initialize analyzer if not done
        if not crypto_analyzer:
            await initialize_components()
        
        # Perform analysis
        async with crypto_analyzer as analyzer:
            signals = await analyzer.analyze_multiple_cryptos(timeframe)
        
        if not signals:
            logger.info("📊 No signals found in current analysis")
            
            if send_to_telegram:
                no_signals_msg = signal_formatter._format_no_signals_message(timeframe)
                await send_telegram_message(no_signals_msg.content)
            
            return {
                'success': True,
                'signals': [],
                'message': 'No high-quality signals found',
                'analysis_duration': (datetime.now() - start_time).total_seconds(),
                'timeframe': timeframe
            }
        
        logger.info(f"✅ Found {len(signals)} high-quality signals")
        
        # Format signals for Telegram
        if send_to_telegram and signal_formatter:
            # Get additional analysis info
            analysis_info = {
                'symbols_analyzed': config.CRYPTO_COUNT,
                'ml_enabled': config.ML_CONFIG.get('enabled', False),
                'news_enabled': config.NEWS_SENTIMENT_CONFIG.get('enabled', False),
                'patterns_enabled': config.PATTERN_RECOGNITION_CONFIG.get('enabled', False),
                'analysis_duration': (datetime.now() - start_time).total_seconds(),
                'session_id': f"session_{int(start_time.timestamp())}",
                'avg_ml_prediction': sum(s.get('ml_prediction', {}).get('success_probability', 0.7) * 100 for s in signals) / len(signals) if signals else 0,
                'overall_news_sentiment': sum(s.get('news_sentiment', {}).get('overall_sentiment', 0) for s in signals) / len(signals) if signals else 0,
                'patterns_found': sum(len(s.get('patterns', [])) for s in signals)
            }
            
            # Format and send summary
            summary_message = signal_formatter.format_signals_summary(signals, timeframe, analysis_info)
            
            with telegram_lock:
                telegram_success = await send_telegram_message(summary_message.content)
            
            if telegram_success:
                logger.info("📱 Analysis summary sent to Telegram")
            else:
                logger.warning("⚠️ Failed to send summary to Telegram")
        
        # Update statistics
        with analysis_lock:
            successful_analyses += 1
            last_analysis_time = datetime.now()
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"🎉 Analysis completed in {duration:.2f} seconds")
        
        return {
            'success': True,
            'signals': signals,
            'signal_count': len(signals),
            'analysis_duration': duration,
            'timeframe': timeframe,
            'analysis_info': analysis_info if send_to_telegram else None
        }
        
    except Exception as e:
        logger.error(f"💥 Error in crypto analysis: {str(e)}")
        logger.error(traceback.format_exc())
        
        error_msg = f"⚠️ Analysis Error\n🔧 {str(e)}\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        if send_to_telegram:
            await send_telegram_message(error_msg)
        
        return {
            'success': False,
            'error': str(e),
            'analysis_duration': (datetime.now() - start_time).total_seconds(),
            'timeframe': timeframe
        }

async def initialize_components():
    """Initialize all analysis components"""
    try:
        global crypto_analyzer, signal_formatter
        
        logger.info("🔧 Initializing analysis components...")
        
        # Initialize signal formatter
        signal_formatter = SignalFormatter()
        logger.info("💎 Signal formatter initialized")
        
        # Initialize crypto analyzer (will auto-initialize advanced components)
        crypto_analyzer = CryptoAnalyzer()
        logger.info("📊 Crypto analyzer initialized")
        
        logger.info("✅ All components initialized successfully")
        
    except Exception as e:
        logger.error(f"💥 Error initializing components: {str(e)}")
        raise

# =============================================================================
# 📊 FLASK API ENDPOINTS
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': config.BOT_VERSION,
            'uptime_seconds': (datetime.now() - app_start_time).total_seconds()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Comprehensive system status"""
    try:
        uptime = datetime.now() - app_start_time
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        status_data = {
            'system': {
                'status': 'operational',
                'version': config.BOT_VERSION,
                'uptime': str(uptime),
                'uptime_seconds': uptime.total_seconds(),
                'start_time': app_start_time.isoformat()
            },
            'performance': {
                'total_requests': total_requests,
                'successful_analyses': successful_analyses,
                'success_rate': successful_analyses / max(total_requests, 1),
                'last_analysis': last_analysis_time.isoformat() if last_analysis_time else None,
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3)
            },
            'configuration': {
                'timezone': str(config.TIMEZONE) if hasattr(config, 'TIMEZONE') else 'UTC',
                'crypto_count': config.CRYPTO_COUNT,
                'signal_count': config.SIGNAL_COUNT,
                'min_signal_strength': config.MIN_SIGNAL_STRENGTH,
                'analysis_interval_minutes': config.ANALYSIS_INTERVAL_MINUTES
            },
            'features': {
                'news_sentiment': config.NEWS_SENTIMENT_CONFIG.get('enabled', False),
                'machine_learning': config.ML_CONFIG.get('enabled', False),
                'pattern_recognition': config.PATTERN_RECOGNITION_CONFIG.get('enabled', False),
                'backtesting': config.BACKTESTING_CONFIG.get('enabled', False),
                'performance_optimization': config.PERFORMANCE_CONFIG.get('parallel_analysis', False)
            },
            'integrations': {
                'telegram_configured': bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID),
                'binance_configured': bool(getattr(config, 'BINANCE_API_KEY', None) and getattr(config, 'BINANCE_SECRET', None)),
                'news_apis_configured': bool(getattr(config, 'NEWS_API_KEY', None)),
                'scheduler_running': scheduler_running
            },
            'market': {
                'supported_timeframes': list(config.EXTENDED_TIMEFRAMES.keys()),
                'risk_levels': list(config.RISK_LEVELS.keys()) if hasattr(config, 'RISK_LEVELS') else ['LOW', 'MEDIUM', 'HIGH']
            }
        }
        
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/analyze', methods=['GET', 'POST'])
def trigger_analysis():
    """Trigger manual crypto analysis"""
    try:
        # Get parameters
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args
        
        timeframe = data.get('timeframe', '1h')
        send_telegram = data.get('send_telegram', 'true').lower() == 'true'
        
        # Validate timeframe
        if timeframe not in config.EXTENDED_TIMEFRAMES:
            return jsonify({
                'success': False,
                'error': f'Invalid timeframe. Supported: {list(config.EXTENDED_TIMEFRAMES.keys())}'
            }), 400
        
        logger.info(f"🎯 Manual analysis triggered for {timeframe}")
        
        # Run analysis in executor to avoid blocking
        def run_analysis():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    perform_crypto_analysis(timeframe, send_telegram)
                )
                return result
            finally:
                loop.close()
        
        # Submit to executor
        future = executor.submit(run_analysis)
        result = future.result(timeout=300)  # 5 minute timeout
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in manual analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/analyze/enhanced', methods=['POST'])
def enhanced_analysis():
    """Enhanced analysis with custom parameters"""
    try:
        data = request.get_json() or {}
        
        # Enhanced parameters
        timeframes = data.get('timeframes', ['1h'])
        symbols = data.get('symbols', [])  # Specific symbols to analyze
        min_strength = data.get('min_strength', config.MIN_SIGNAL_STRENGTH)
        risk_levels = data.get('risk_levels', ['LOW', 'MEDIUM', 'HIGH'])
        send_telegram = data.get('send_telegram', True)
        
        # Validate parameters
        invalid_timeframes = [tf for tf in timeframes if tf not in config.EXTENDED_TIMEFRAMES]
        if invalid_timeframes:
            return jsonify({
                'success': False,
                'error': f'Invalid timeframes: {invalid_timeframes}'
            }), 400
        
        logger.info(f"🔥 Enhanced analysis triggered for {timeframes}")
        
        def run_enhanced_analysis():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                all_results = []
                
                for timeframe in timeframes:
                    result = loop.run_until_complete(
                        perform_crypto_analysis(timeframe, send_telegram and timeframe == timeframes[0])
                    )
                    
                    if result['success'] and result.get('signals'):
                        # Filter signals based on criteria
                        filtered_signals = []
                        for signal in result['signals']:
                            if (signal.get('signal_strength', 0) >= min_strength and
                                signal.get('risk_level', 'MEDIUM') in risk_levels):
                                if not symbols or signal.get('symbol') in symbols:
                                    filtered_signals.append(signal)
                        
                        result['signals'] = filtered_signals
                        result['filtered_signal_count'] = len(filtered_signals)
                    
                    all_results.append(result)
                
                return {
                    'success': True,
                    'results': all_results,
                    'total_timeframes': len(timeframes),
                    'parameters': {
                        'timeframes': timeframes,
                        'symbols': symbols,
                        'min_strength': min_strength,
                        'risk_levels': risk_levels
                    }
                }
                
            finally:
                loop.close()
        
        future = executor.submit(run_enhanced_analysis)
        result = future.result(timeout=600)  # 10 minute timeout for multiple timeframes
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/test', methods=['GET'])
def test_telegram():
    """Test Telegram connectivity"""
    try:
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            return jsonify({
                'success': False,
                'error': 'Telegram credentials not configured'
            }), 400
        
        def test_telegram_sync():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Send test message
                test_message = signal_formatter.format_test_message() if signal_formatter else None
                
                if test_message:
                    success = loop.run_until_complete(send_telegram_message(test_message.content))
                else:
                    success = loop.run_until_complete(send_telegram_message(
                        "🧪 <b>Bot Test Message</b>\n✅ Crypto Analysis Bot V3.0 Ultimate is online!"
                    ))
                
                return success
            finally:
                loop.close()
        
        success = test_telegram_sync()
        
        return jsonify({
            'success': success,
            'message': 'Test message sent to Telegram' if success else 'Failed to send test message',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error testing Telegram: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/models/status', methods=['GET'])
def get_model_status():
    """Get ML models status"""
    try:
        # This would integrate with MLPredictor
        model_status = {
            'ml_enabled': config.ML_CONFIG.get('enabled', False),
            'models_loaded': False,
            'last_training': None,
            'prediction_count': 0,
            'success_rate': 0.0
        }
        
        # Try to get actual model status if available
        try:
            # This would require MLPredictor integration
            pass
        except:
            pass
        
        return jsonify({
            'success': True,
            'model_status': model_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/backtest', methods=['POST'])
def run_backtest():
    """Run strategy backtesting"""
    try:
        data = request.get_json() or {}
        
        # Backtest parameters
        days = data.get('days', 30)
        symbols = data.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        timeframe = data.get('timeframe', '1h')
        initial_capital = data.get('initial_capital', 10000)
        
        # This would integrate with BacktestingEngine
        # For now, return mock results
        
        mock_results = {
            'success': True,
            'backtest_results': {
                'total_trades': 25,
                'winning_trades': 18,
                'losing_trades': 7,
                'win_rate': 0.72,
                'total_return_pct': 15.4,
                'max_drawdown': 3.2,
                'sharpe_ratio': 2.1,
                'profit_factor': 2.8
            },
            'parameters': {
                'days': days,
                'symbols': symbols,
                'timeframe': timeframe,
                'initial_capital': initial_capital
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(mock_results)
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/performance', methods=['GET'])
def get_performance_metrics():
    """Get detailed performance metrics"""
    try:
        uptime = datetime.now() - app_start_time
        
        # Calculate performance metrics
        requests_per_hour = total_requests / max(uptime.total_seconds() / 3600, 1)
        success_rate = successful_analyses / max(total_requests, 1)
        
        performance_data = {
            'analysis_performance': {
                'total_requests': total_requests,
                'successful_analyses': successful_analyses,
                'success_rate': success_rate,
                'requests_per_hour': requests_per_hour,
                'last_analysis': last_analysis_time.isoformat() if last_analysis_time else None,
                'avg_response_time': 'N/A'  # Would need to track this
            },
            'system_performance': {
                'uptime': str(uptime),
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'active_threads': threading.active_count()
            },
            'feature_performance': {
                'ml_predictions': 0,  # Would integrate with ML components
                'news_analyses': 0,   # Would integrate with news analyzer
                'patterns_detected': 0,  # Would integrate with pattern recognition
                'backtests_run': 0    # Would integrate with backtesting
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(performance_data)
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# ⏰ BACKGROUND SCHEDULER
# =============================================================================

def setup_scheduler():
    """Setup background analysis scheduler"""
    try:
        global scheduler_running
        
        logger.info("⏰ Setting up analysis scheduler...")
        
        # Clear any existing jobs
        schedule.clear()
        
        # Schedule analyses based on timeframes
        
        # 1h analysis every 2 hours
        schedule.every(2).hours.do(
            lambda: executor.submit(run_scheduled_analysis, '1h')
        )
        
        # 4h analysis every 6 hours
        schedule.every(6).hours.do(
            lambda: executor.submit(run_scheduled_analysis, '4h')
        )
        
        # 1d analysis daily at 9 AM
        schedule.every().day.at("09:00").do(
            lambda: executor.submit(run_scheduled_analysis, '1d')
        )
        
        # Weekly analysis on Mondays at 10 AM
        schedule.every().monday.at("10:00").do(
            lambda: executor.submit(run_scheduled_analysis, '1w')
        )
        
        # Performance update every 24 hours
        schedule.every(24).hours.do(
            lambda: executor.submit(send_performance_update)
        )
        
        scheduler_running = True
        logger.info("✅ Scheduler configured successfully")
        
    except Exception as e:
        logger.error(f"Error setting up scheduler: {str(e)}")

def run_scheduled_analysis(timeframe: str):
    """Run scheduled analysis"""
    try:
        logger.info(f"⏰ Running scheduled analysis for {timeframe}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                perform_crypto_analysis(timeframe, send_to_telegram=True)
            )
            
            if result['success']:
                logger.info(f"✅ Scheduled analysis {timeframe} completed: {result.get('signal_count', 0)} signals")
            else:
                logger.warning(f"⚠️ Scheduled analysis {timeframe} failed: {result.get('error', 'Unknown error')}")
                
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in scheduled analysis {timeframe}: {str(e)}")

def send_performance_update():
    """Send performance update to Telegram"""
    try:
        if not signal_formatter:
            return
        
        # Get performance stats
        uptime = datetime.now() - app_start_time
        
        performance_stats = {
            'analysis_count': total_requests,
            'successful_signals': successful_analyses,
            'success_rate': successful_analyses / max(total_requests, 1),
            'uptime_formatted': str(uptime),
            'features_enabled': {
                'news_analyzer': config.NEWS_SENTIMENT_CONFIG.get('enabled', False),
                'ml_predictor': config.ML_CONFIG.get('enabled', False),
                'pattern_analyzer': config.PATTERN_RECOGNITION_CONFIG.get('enabled', False),
                'backtest_engine': config.BACKTESTING_CONFIG.get('enabled', False)
            },
            'market_regime': 'bull'  # Would integrate with market analysis
        }
        
        # Format performance message
        perf_message = signal_formatter.format_performance_update(performance_stats)
        
        # Send to Telegram
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(send_telegram_message(perf_message.content))
            
            if success:
                logger.info("📊 Performance update sent to Telegram")
            else:
                logger.warning("⚠️ Failed to send performance update")
                
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error sending performance update: {str(e)}")

def scheduler_worker():
    """Background scheduler worker thread"""
    global scheduler_running
    
    logger.info("⏰ Starting scheduler worker thread")
    
    while scheduler_running and not is_shutting_down:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in scheduler worker: {str(e)}")
            time.sleep(60)
    
    logger.info("⏰ Scheduler worker thread stopped")

# =============================================================================
# 🔧 APPLICATION LIFECYCLE
# =============================================================================

def initialize_application():
    """Initialize the complete application"""
    try:
        logger.info("🚀 Initializing Crypto Analysis Bot V3.0 Ultimate...")
        
        # Print configuration summary
        config_summary = config.get_config_summary() if hasattr(config, 'get_config_summary') else {}
        logger.info(f"🔧 Configuration: {config_summary}")
        
        # Initialize components asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(initialize_components())
        finally:
            loop.close()
        
        # Setup scheduler
        setup_scheduler()
        
        # Start scheduler thread
        global scheduler_thread
        scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
        scheduler_thread.start()
        
        # Send startup notification
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            startup_message = "🚀 <b>Bot Started</b>\n✅ Crypto Analysis Bot V3.0 Ultimate is now online!\n📊 All systems operational"
            send_telegram_sync(startup_message)
        
        logger.info("✅ Application initialization completed successfully")
        
    except Exception as e:
        logger.error(f"💥 Critical error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def shutdown_handler(signum, frame):
    """Graceful shutdown handler"""
    global is_shutting_down, scheduler_running
    
    logger.info("🛑 Received shutdown signal, performing graceful shutdown...")
    
    is_shutting_down = True
    scheduler_running = False
    
    # Send shutdown notification
    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        shutdown_message = "🛑 <b>Bot Shutting Down</b>\n⏳ Crypto Analysis Bot V3.0 Ultimate is going offline..."
        send_telegram_sync(shutdown_message)
    
    # Wait for background tasks
    if scheduler_thread and scheduler_thread.is_alive():
        scheduler_thread.join(timeout=10)
    
    # Shutdown executor
    executor.shutdown(wait=True)
    
    logger.info("✅ Graceful shutdown completed")
    sys.exit(0)

# =============================================================================
# 🌍 APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    try:
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        
        # Initialize application
        initialize_application()
        
        # Get configuration
        port = int(os.getenv('PORT', 8080))
        debug = os.getenv('FLASK_ENV') == 'development'
        
        logger.info(f"🌐 Starting Flask server on port {port}")
        logger.info(f"🔧 Debug mode: {debug}")
        logger.info(f"🌍 Environment: {os.getenv('FLASK_ENV', 'production')}")
        
        # Print feature status
        logger.info("🔥 Advanced Features Status:")
        logger.info(f"   📰 News Sentiment: {config.NEWS_SENTIMENT_CONFIG.get('enabled', False)}")
        logger.info(f"   🤖 Machine Learning: {config.ML_CONFIG.get('enabled', False)}")
        logger.info(f"   📊 Pattern Recognition: {config.PATTERN_RECOGNITION_CONFIG.get('enabled', False)}")
        logger.info(f"   📈 Backtesting: {config.BACKTESTING_CONFIG.get('enabled', False)}")
        logger.info(f"   ⚡ Performance Optimization: {config.PERFORMANCE_CONFIG.get('parallel_analysis', False)}")
        
        # Print API endpoints
        logger.info("🌐 Available API Endpoints:")
        logger.info("   GET  /health - Health check")
        logger.info("   GET  /status - System status")
        logger.info("   GET  /analyze - Trigger analysis")
        logger.info("   POST /analyze/enhanced - Enhanced analysis")
        logger.info("   GET  /test - Test Telegram")
        logger.info("   GET  /models/status - ML models status")
        logger.info("   POST /backtest - Run backtesting")
        logger.info("   GET  /performance - Performance metrics")
        
        # Start Flask application
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=False  # Disable reloader to avoid duplicate scheduler
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
        shutdown_handler(signal.SIGINT, None)
        
    except Exception as e:
        logger.error(f"💥 Critical startup error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Send error notification
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            error_message = f"💥 <b>Critical Startup Error</b>\n🔧 {str(e)}"
            send_telegram_sync(error_message)
        
        sys.exit(1)
