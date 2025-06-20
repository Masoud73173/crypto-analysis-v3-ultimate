# backtesting_engine.py - Advanced Backtesting and Strategy Optimization Engine V3.0 Ultimate
import numpy as np
import pandas as pd
import logging
import json
import os
import asyncio
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

try:
    import config
except ImportError:
    # Fallback configuration
    class MockConfig:
        BACKTESTING_CONFIG = {
            'enabled': True,
            'historical_days': 30,
            'optimization_period': 7,
            'min_trades_for_optimization': 50,
            'success_rate_threshold': 0.7,
            'transaction_costs': {'maker_fee': 0.001, 'taker_fee': 0.001, 'slippage': 0.0005}
        }
    config = MockConfig()

# =============================================================================
# 📊 DATA CLASSES
# =============================================================================

@dataclass
class TradeResult:
    """Individual trade result"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str  # 'LONG', 'SHORT'
    entry_price: float
    exit_price: float
    quantity: float
    profit_loss: float
    profit_loss_pct: float
    fees: float
    slippage: float
    net_profit: float
    trade_duration: timedelta
    exit_reason: str  # 'TARGET', 'STOP_LOSS', 'TIMEOUT', 'MANUAL'
    signal_strength: float
    max_profit: float = 0.0
    max_loss: float = 0.0
    max_drawdown: float = 0.0
    risk_reward_achieved: float = 0.0

@dataclass
class PositionInfo:
    """Active position information"""
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    target_levels: List[float]
    current_target_index: int
    signal_id: str
    signal_strength: float
    timeframe: str
    unrealized_pnl: float = 0.0
    max_profit_seen: float = 0.0
    max_loss_seen: float = 0.0

@dataclass
class BacktestResult:
    """Complete backtest results"""
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit_loss: float
    total_fees: float
    net_profit_loss: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    sharpe_ratio: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade_duration: timedelta
    total_return_pct: float
    annual_return_pct: float
    trades: List[TradeResult]
    equity_curve: List[Tuple[datetime, float]]
    monthly_returns: Dict[str, float]
    drawdown_periods: List[Tuple[datetime, datetime, float]]
    performance_by_timeframe: Dict[str, Dict[str, float]]
    performance_by_symbol: Dict[str, Dict[str, float]]

@dataclass
class OptimizationResult:
    """Strategy optimization result"""
    parameter_set: Dict[str, Any]
    backtest_result: BacktestResult
    fitness_score: float
    optimization_metric: str
    rank: int

@dataclass
class PerformanceMetrics:
    """Performance analysis metrics"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    risk_adjusted_return: float

# =============================================================================
# 📈 BACKTESTING ENGINE CLASS
# =============================================================================

class BacktestingEngine:
    """Advanced backtesting and strategy optimization engine"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # Backtesting configuration
        self.initial_capital = 10000  # Starting capital for backtests
        self.transaction_costs = config.BACKTESTING_CONFIG.get('transaction_costs', {
            'maker_fee': 0.001,   # 0.1% maker fee
            'taker_fee': 0.001,   # 0.1% taker fee
            'slippage': 0.0005    # 0.05% slippage
        })
        
        # Risk management
        self.max_position_size = 0.1     # 10% of capital per position
        self.max_concurrent_positions = 5  # Maximum simultaneous positions
        self.max_daily_loss = 0.05       # 5% max daily loss
        
        # Performance tracking
        self.cache_dir = './backtest_cache/'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Active positions (for live tracking)
        self.active_positions = {}
        self.position_lock = threading.Lock()
        
        # Historical results for optimization
        self.historical_results = []
        
        self.logger.info("📈 BacktestingEngine V3.0 Ultimate initialized")
    
    def _setup_logger(self):
        """Setup logging for backtesting engine"""
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
    # 🎯 MAIN BACKTESTING METHODS
    # =============================================================================
    
    async def backtest_strategy(self, 
                               signals: List[Dict], 
                               historical_data: Dict[str, pd.DataFrame],
                               start_date: datetime,
                               end_date: datetime,
                               initial_capital: float = None) -> BacktestResult:
        """
        Comprehensive strategy backtesting with realistic market conditions
        
        Args:
            signals: List of trading signals to test
            historical_data: Historical OHLCV data for symbols
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital (optional)
            
        Returns:
            Detailed backtest results
        """
        try:
            self.logger.info(f"🎯 Starting backtest: {start_date} to {end_date}")
            
            if initial_capital:
                capital = initial_capital
            else:
                capital = self.initial_capital
            
            # Initialize backtest state
            current_capital = capital
            active_positions = {}
            completed_trades = []
            equity_curve = [(start_date, capital)]
            daily_returns = []
            
            # Process signals chronologically
            signals_df = self._prepare_signals_for_backtest(signals, start_date, end_date)
            
            if signals_df.empty:
                self.logger.warning("No signals found in date range")
                return self._create_empty_backtest_result(start_date, end_date, capital)
            
            # Main backtest loop
            current_date = start_date
            while current_date <= end_date:
                # Check for new signals on this date
                daily_signals = signals_df[signals_df['timestamp'].dt.date == current_date.date()]
                
                # Process new signals
                for _, signal in daily_signals.iterrows():
                    trade_result = await self._process_signal(
                        signal, historical_data, current_capital, active_positions, current_date
                    )
                    
                    if trade_result:
                        current_capital += trade_result.net_profit
                        completed_trades.append(trade_result)
                
                # Update active positions
                position_updates = await self._update_active_positions(
                    active_positions, historical_data, current_date
                )
                
                for update in position_updates:
                    if update.exit_reason:  # Position closed
                        current_capital += update.net_profit
                        completed_trades.append(update)
                        del active_positions[update.symbol]
                
                # Record daily equity
                daily_equity = current_capital + sum(
                    pos.unrealized_pnl for pos in active_positions.values()
                )
                equity_curve.append((current_date, daily_equity))
                
                # Calculate daily return
                if len(equity_curve) > 1:
                    prev_equity = equity_curve[-2][1]
                    daily_return = (daily_equity - prev_equity) / prev_equity
                    daily_returns.append(daily_return)
                
                current_date += timedelta(days=1)
            
            # Close remaining positions at end date
            final_trades = await self._close_remaining_positions(
                active_positions, historical_data, end_date
            )
            
            for trade in final_trades:
                current_capital += trade.net_profit
                completed_trades.append(trade)
            
            # Calculate comprehensive results
            backtest_result = self._calculate_backtest_metrics(
                completed_trades, equity_curve, daily_returns, start_date, end_date, capital
            )
            
            self.logger.info(f"✅ Backtest completed: {len(completed_trades)} trades, "
                           f"{backtest_result.win_rate:.1%} win rate, "
                           f"{backtest_result.total_return_pct:.2f}% return")
            
            return backtest_result
            
        except Exception as e:
            self.logger.error(f"💥 Error in backtesting: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._create_empty_backtest_result(start_date, end_date, capital)
    
    async def _process_signal(self, 
                            signal: pd.Series, 
                            historical_data: Dict[str, pd.DataFrame],
                            current_capital: float,
                            active_positions: Dict[str, PositionInfo],
                            signal_date: datetime) -> Optional[TradeResult]:
        """Process a trading signal and potentially open a position"""
        try:
            symbol = signal.get('symbol', '')
            direction = signal.get('type', '').upper()
            
            if not symbol or direction not in ['LONG', 'SHORT']:
                return None
            
            # Check if we already have a position in this symbol
            if symbol in active_positions:
                return None
            
            # Check position limits
            if len(active_positions) >= self.max_concurrent_positions:
                return None
            
            # Get historical data for this symbol
            if symbol not in historical_data:
                return None
            
            symbol_data = historical_data[symbol]
            
            # Find the signal date in historical data
            signal_candle = self._find_candle_for_date(symbol_data, signal_date)
            if signal_candle is None:
                return None
            
            # Calculate position size
            signal_strength = signal.get('signal_strength', 70) / 100.0
            risk_level = signal.get('risk_level', 'MEDIUM')
            
            position_size = self._calculate_position_size(
                current_capital, signal_strength, risk_level
            )
            
            if position_size <= 0:
                return None
            
            # Entry price (use open of next candle with slippage)
            entry_price = signal_candle['open'] * (
                1 + self.transaction_costs['slippage'] if direction == 'LONG' 
                else 1 - self.transaction_costs['slippage']
            )
            
            # Calculate quantity
            quantity = position_size / entry_price
            
            # Set stop loss and targets from signal
            stop_loss = signal.get('stop_loss', entry_price * (0.98 if direction == 'LONG' else 1.02))
            targets = signal.get('targets', [entry_price * (1.02 if direction == 'LONG' else 0.98)])
            
            # Create position
            position = PositionInfo(
                symbol=symbol,
                direction=direction,
                entry_time=signal_date,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss,
                target_levels=targets if isinstance(targets, list) else [targets],
                current_target_index=0,
                signal_id=signal.get('signal_id', f"{symbol}_{signal_date}"),
                signal_strength=signal.get('signal_strength', 70),
                timeframe=signal.get('timeframe', '1h')
            )
            
            active_positions[symbol] = position
            
            self.logger.debug(f"📊 Opened {direction} position: {symbol} @ {entry_price:.4f}")
            
            return None  # Position opened, trade result will come when closed
            
        except Exception as e:
            self.logger.debug(f"Error processing signal: {str(e)}")
            return None
    
    async def _update_active_positions(self, 
                                     active_positions: Dict[str, PositionInfo],
                                     historical_data: Dict[str, pd.DataFrame],
                                     current_date: datetime) -> List[TradeResult]:
        """Update active positions and check for exits"""
        completed_trades = []
        
        for symbol, position in list(active_positions.items()):
            try:
                if symbol not in historical_data:
                    continue
                
                symbol_data = historical_data[symbol]
                current_candle = self._find_candle_for_date(symbol_data, current_date)
                
                if current_candle is None:
                    continue
                
                # Update unrealized P&L
                current_price = current_candle['close']
                position.unrealized_pnl = self._calculate_unrealized_pnl(position, current_price)
                
                # Track max profit/loss seen
                if position.unrealized_pnl > position.max_profit_seen:
                    position.max_profit_seen = position.unrealized_pnl
                if position.unrealized_pnl < position.max_loss_seen:
                    position.max_loss_seen = position.unrealized_pnl
                
                # Check for exit conditions
                exit_result = self._check_exit_conditions(position, current_candle, current_date)
                
                if exit_result:
                    trade_result = self._create_trade_result(position, exit_result, current_candle)
                    completed_trades.append(trade_result)
                    
                    self.logger.debug(f"💰 Closed {position.direction} {symbol}: "
                                    f"{trade_result.profit_loss_pct:.2f}% ({exit_result['reason']})")
                
            except Exception as e:
                self.logger.debug(f"Error updating position {symbol}: {str(e)}")
        
        return completed_trades
    
    def _check_exit_conditions(self, 
                             position: PositionInfo, 
                             current_candle: pd.Series,
                             current_date: datetime) -> Optional[Dict[str, Any]]:
        """Check if position should be closed"""
        try:
            high = current_candle['high']
            low = current_candle['low']
            close = current_candle['close']
            
            # Check stop loss
            if position.direction == 'LONG':
                if low <= position.stop_loss:
                    return {
                        'exit_price': position.stop_loss,
                        'exit_time': current_date,
                        'reason': 'STOP_LOSS'
                    }
            else:  # SHORT
                if high >= position.stop_loss:
                    return {
                        'exit_price': position.stop_loss,
                        'exit_time': current_date,
                        'reason': 'STOP_LOSS'
                    }
            
            # Check target levels
            if position.current_target_index < len(position.target_levels):
                target = position.target_levels[position.current_target_index]
                
                if position.direction == 'LONG':
                    if high >= target:
                        return {
                            'exit_price': target,
                            'exit_time': current_date,
                            'reason': f'TARGET_{position.current_target_index + 1}'
                        }
                else:  # SHORT
                    if low <= target:
                        return {
                            'exit_price': target,
                            'exit_time': current_date,
                            'reason': f'TARGET_{position.current_target_index + 1}'
                        }
            
            # Check timeout (close after 7 days if no exit)
            time_in_position = current_date - position.entry_time
            if time_in_position > timedelta(days=7):
                return {
                    'exit_price': close,
                    'exit_time': current_date,
                    'reason': 'TIMEOUT'
                }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error checking exit conditions: {str(e)}")
            return None
    
    def _create_trade_result(self, 
                           position: PositionInfo, 
                           exit_info: Dict[str, Any],
                           current_candle: pd.Series) -> TradeResult:
        """Create trade result from closed position"""
        try:
            exit_price = exit_info['exit_price']
            exit_time = exit_info['exit_time']
            exit_reason = exit_info['reason']
            
            # Calculate raw profit/loss
            if position.direction == 'LONG':
                raw_profit = (exit_price - position.entry_price) * position.quantity
            else:  # SHORT
                raw_profit = (position.entry_price - exit_price) * position.quantity
            
            # Calculate fees
            entry_fee = position.entry_price * position.quantity * self.transaction_costs['taker_fee']
            exit_fee = exit_price * position.quantity * self.transaction_costs['taker_fee']
            total_fees = entry_fee + exit_fee
            
            # Calculate slippage (already included in entry price)
            slippage_cost = abs(position.entry_price * position.quantity * self.transaction_costs['slippage'])
            
            # Net profit
            net_profit = raw_profit - total_fees - slippage_cost
            
            # Percentage return
            position_value = position.entry_price * position.quantity
            profit_loss_pct = (net_profit / position_value) * 100
            
            # Risk/reward achieved
            if position.direction == 'LONG':
                potential_loss = (position.entry_price - position.stop_loss) * position.quantity
                risk_reward_achieved = abs(raw_profit / potential_loss) if potential_loss != 0 else 0
            else:
                potential_loss = (position.stop_loss - position.entry_price) * position.quantity
                risk_reward_achieved = abs(raw_profit / potential_loss) if potential_loss != 0 else 0
            
            return TradeResult(
                entry_time=position.entry_time,
                exit_time=exit_time,
                symbol=position.symbol,
                direction=position.direction,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                profit_loss=raw_profit,
                profit_loss_pct=profit_loss_pct,
                fees=total_fees,
                slippage=slippage_cost,
                net_profit=net_profit,
                trade_duration=exit_time - position.entry_time,
                exit_reason=exit_reason,
                signal_strength=position.signal_strength,
                max_profit=position.max_profit_seen,
                max_loss=position.max_loss_seen,
                max_drawdown=abs(position.max_loss_seen) if position.max_loss_seen < 0 else 0,
                risk_reward_achieved=risk_reward_achieved
            )
            
        except Exception as e:
            self.logger.error(f"Error creating trade result: {str(e)}")
            # Return a neutral trade result
            return TradeResult(
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                symbol=position.symbol,
                direction=position.direction,
                entry_price=position.entry_price,
                exit_price=position.entry_price,
                quantity=position.quantity,
                profit_loss=0,
                profit_loss_pct=0,
                fees=0,
                slippage=0,
                net_profit=0,
                trade_duration=timedelta(hours=1),
                exit_reason='ERROR',
                signal_strength=50
            )
    
    # =============================================================================
    # 📊 PERFORMANCE ANALYSIS
    # =============================================================================
    
    def _calculate_backtest_metrics(self, 
                                  trades: List[TradeResult],
                                  equity_curve: List[Tuple[datetime, float]],
                                  daily_returns: List[float],
                                  start_date: datetime,
                                  end_date: datetime,
                                  initial_capital: float) -> BacktestResult:
        """Calculate comprehensive backtest performance metrics"""
        try:
            if not trades:
                return self._create_empty_backtest_result(start_date, end_date, initial_capital)
            
            # Basic trade statistics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.net_profit > 0])
            losing_trades = len([t for t in trades if t.net_profit < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit/Loss calculations
            total_profit_loss = sum(t.profit_loss for t in trades)
            total_fees = sum(t.fees for t in trades)
            net_profit_loss = sum(t.net_profit for t in trades)
            
            # Win/Loss statistics
            wins = [t.net_profit for t in trades if t.net_profit > 0]
            losses = [t.net_profit for t in trades if t.net_profit < 0]
            
            average_win = np.mean(wins) if wins else 0
            average_loss = np.mean(losses) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = min(losses) if losses else 0
            
            # Profit factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Returns
            final_equity = equity_curve[-1][1] if equity_curve else initial_capital
            total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
            
            # Annualized return
            days_in_backtest = (end_date - start_date).days
            years = days_in_backtest / 365.25
            annual_return_pct = ((final_equity / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
            
            # Drawdown analysis
            drawdown_info = self._calculate_drawdown_metrics(equity_curve)
            
            # Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            
            # Trade duration
            durations = [t.trade_duration for t in trades]
            average_trade_duration = sum(durations, timedelta()) / len(durations) if durations else timedelta()
            
            # Monthly returns
            monthly_returns = self._calculate_monthly_returns(equity_curve)
            
            # Performance by timeframe and symbol
            performance_by_timeframe = self._analyze_performance_by_timeframe(trades)
            performance_by_symbol = self._analyze_performance_by_symbol(trades)
            
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_profit_loss=total_profit_loss,
                total_fees=total_fees,
                net_profit_loss=net_profit_loss,
                max_drawdown=drawdown_info['max_drawdown'],
                max_drawdown_duration=drawdown_info['max_duration'],
                sharpe_ratio=sharpe_ratio,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                average_trade_duration=average_trade_duration,
                total_return_pct=total_return_pct,
                annual_return_pct=annual_return_pct,
                trades=trades,
                equity_curve=equity_curve,
                monthly_returns=monthly_returns,
                drawdown_periods=drawdown_info['drawdown_periods'],
                performance_by_timeframe=performance_by_timeframe,
                performance_by_symbol=performance_by_symbol
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating backtest metrics: {str(e)}")
            return self._create_empty_backtest_result(start_date, end_date, initial_capital)
    
    def _calculate_drawdown_metrics(self, equity_curve: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Calculate drawdown metrics from equity curve"""
        try:
            if len(equity_curve) < 2:
                return {
                    'max_drawdown': 0.0,
                    'max_duration': timedelta(),
                    'drawdown_periods': []
                }
            
            # Calculate running maximum (peak)
            equity_values = [e[1] for e in equity_curve]
            equity_dates = [e[0] for e in equity_curve]
            
            running_max = np.maximum.accumulate(equity_values)
            drawdowns = [(equity_values[i] - running_max[i]) / running_max[i] * 100 
                        for i in range(len(equity_values))]
            
            max_drawdown = min(drawdowns) if drawdowns else 0.0
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            drawdown_start = None
            
            for i, dd in enumerate(drawdowns):
                if dd < -0.1 and not in_drawdown:  # Start of drawdown (>0.1%)
                    in_drawdown = True
                    drawdown_start = equity_dates[i]
                elif dd >= 0 and in_drawdown:  # End of drawdown
                    in_drawdown = False
                    if drawdown_start:
                        drawdown_periods.append((
                            drawdown_start, 
                            equity_dates[i], 
                            min(drawdowns[equity_dates.index(drawdown_start):i+1])
                        ))
            
            # Calculate maximum drawdown duration
            max_duration = timedelta()
            if drawdown_periods:
                max_duration = max(end - start for start, end, _ in drawdown_periods)
            
            return {
                'max_drawdown': abs(max_drawdown),
                'max_duration': max_duration,
                'drawdown_periods': drawdown_periods
            }
            
        except Exception as e:
            self.logger.debug(f"Error calculating drawdown metrics: {str(e)}")
            return {'max_drawdown': 0.0, 'max_duration': timedelta(), 'drawdown_periods': []}
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from daily returns"""
        try:
            if not daily_returns or len(daily_returns) < 2:
                return 0.0
            
            returns_array = np.array(daily_returns)
            
            # Annualized return
            annual_return = np.mean(returns_array) * 252  # 252 trading days per year
            
            # Annualized volatility
            annual_volatility = np.std(returns_array) * np.sqrt(252)
            
            # Sharpe ratio
            if annual_volatility == 0:
                return 0.0
            
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            return sharpe
            
        except Exception as e:
            self.logger.debug(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def _calculate_monthly_returns(self, equity_curve: List[Tuple[datetime, float]]) -> Dict[str, float]:
        """Calculate monthly returns from equity curve"""
        try:
            monthly_returns = {}
            
            if len(equity_curve) < 2:
                return monthly_returns
            
            # Group by month
            monthly_equity = {}
            for date, equity in equity_curve:
                month_key = date.strftime('%Y-%m')
                if month_key not in monthly_equity:
                    monthly_equity[month_key] = {'start': equity, 'end': equity}
                else:
                    monthly_equity[month_key]['end'] = equity
            
            # Calculate returns
            for month, data in monthly_equity.items():
                if data['start'] > 0:
                    monthly_return = (data['end'] - data['start']) / data['start'] * 100
                    monthly_returns[month] = monthly_return
            
            return monthly_returns
            
        except Exception as e:
            self.logger.debug(f"Error calculating monthly returns: {str(e)}")
            return {}
    
    def _analyze_performance_by_timeframe(self, trades: List[TradeResult]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by trading timeframe"""
        try:
            timeframe_data = {}
            
            # Group trades by duration
            for trade in trades:
                duration_hours = trade.trade_duration.total_seconds() / 3600
                
                if duration_hours <= 4:
                    timeframe = 'scalp'
                elif duration_hours <= 24:
                    timeframe = 'intraday'
                elif duration_hours <= 168:  # 1 week
                    timeframe = 'swing'
                else:
                    timeframe = 'position'
                
                if timeframe not in timeframe_data:
                    timeframe_data[timeframe] = []
                
                timeframe_data[timeframe].append(trade)
            
            # Calculate metrics for each timeframe
            performance = {}
            for timeframe, timeframe_trades in timeframe_data.items():
                wins = len([t for t in timeframe_trades if t.net_profit > 0])
                total = len(timeframe_trades)
                win_rate = wins / total if total > 0 else 0
                
                total_return = sum(t.profit_loss_pct for t in timeframe_trades)
                avg_return = total_return / total if total > 0 else 0
                
                performance[timeframe] = {
                    'trades': total,
                    'win_rate': win_rate,
                    'avg_return_pct': avg_return,
                    'total_return_pct': total_return
                }
            
            return performance
            
        except Exception as e:
            self.logger.debug(f"Error analyzing performance by timeframe: {str(e)}")
            return {}
    
    def _analyze_performance_by_symbol(self, trades: List[TradeResult]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by trading symbol"""
        try:
            symbol_data = {}
            
            # Group trades by symbol
            for trade in trades:
                if trade.symbol not in symbol_data:
                    symbol_data[trade.symbol] = []
                symbol_data[trade.symbol].append(trade)
            
            # Calculate metrics for each symbol
            performance = {}
            for symbol, symbol_trades in symbol_data.items():
                wins = len([t for t in symbol_trades if t.net_profit > 0])
                total = len(symbol_trades)
                win_rate = wins / total if total > 0 else 0
                
                total_return = sum(t.profit_loss_pct for t in symbol_trades)
                avg_return = total_return / total if total > 0 else 0
                
                performance[symbol] = {
                    'trades': total,
                    'win_rate': win_rate,
                    'avg_return_pct': avg_return,
                    'total_return_pct': total_return
                }
            
            return performance
            
        except Exception as e:
            self.logger.debug(f"Error analyzing performance by symbol: {str(e)}")
            return {}
    
    # =============================================================================
    # 🔧 OPTIMIZATION METHODS
    # =============================================================================
    
    async def optimize_strategy(self, 
                              signals: List[Dict],
                              historical_data: Dict[str, pd.DataFrame],
                              optimization_params: Dict[str, List],
                              optimization_metric: str = 'sharpe_ratio',
                              max_iterations: int = 100) -> List[OptimizationResult]:
        """
        Optimize strategy parameters using grid search or genetic algorithm
        
        Args:
            signals: Base signals to optimize
            historical_data: Historical data for backtesting
            optimization_params: Parameters to optimize with their ranges
            optimization_metric: Metric to optimize ('sharpe_ratio', 'win_rate', 'profit_factor')
            max_iterations: Maximum optimization iterations
            
        Returns:
            List of optimization results sorted by performance
        """
        try:
            self.logger.info(f"🔧 Starting strategy optimization with {max_iterations} iterations")
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(
                optimization_params, max_iterations
            )
            
            optimization_results = []
            
            # Use ThreadPoolExecutor for parallel backtesting
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit backtest tasks
                future_to_params = {}
                
                for i, params in enumerate(param_combinations):
                    # Apply parameters to signals
                    modified_signals = self._apply_parameters_to_signals(signals, params)
                    
                    # Submit backtest task
                    future = executor.submit(
                        asyncio.run,
                        self.backtest_strategy(
                            modified_signals,
                            historical_data,
                            datetime.now() - timedelta(days=30),
                            datetime.now()
                        )
                    )
                    
                    future_to_params[future] = (params, i)
                
                # Collect results
                for future in as_completed(future_to_params):
                    params, iteration = future_to_params[future]
                    
                    try:
                        backtest_result = future.result()
                        
                        # Calculate fitness score
                        fitness_score = self._calculate_fitness_score(
                            backtest_result, optimization_metric
                        )
                        
                        optimization_results.append(OptimizationResult(
                            parameter_set=params,
                            backtest_result=backtest_result,
                            fitness_score=fitness_score,
                            optimization_metric=optimization_metric,
                            rank=0  # Will be set after sorting
                        ))
                        
                        self.logger.debug(f"✅ Completed optimization {iteration+1}/{len(param_combinations)}")
                        
                    except Exception as e:
                        self.logger.debug(f"Error in optimization iteration {iteration}: {str(e)}")
            
            # Sort results by fitness score
            optimization_results.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Set ranks
            for i, result in enumerate(optimization_results):
                result.rank = i + 1
            
            self.logger.info(f"🏆 Optimization completed: {len(optimization_results)} valid results")
            
            if optimization_results:
                best_result = optimization_results[0]
                self.logger.info(f"   Best result: {optimization_metric}={best_result.fitness_score:.4f}")
                self.logger.info(f"   Parameters: {best_result.parameter_set}")
            
            return optimization_results[:20]  # Return top 20 results
            
        except Exception as e:
            self.logger.error(f"Error in strategy optimization: {str(e)}")
            return []
    
    def _generate_parameter_combinations(self, 
                                       optimization_params: Dict[str, List],
                                       max_combinations: int) -> List[Dict[str, Any]]:
        """Generate parameter combinations for optimization"""
        try:
            from itertools import product
            
            param_names = list(optimization_params.keys())
            param_values = list(optimization_params.values())
            
            # Generate all combinations
            all_combinations = list(product(*param_values))
            
            # Limit to max_combinations
            if len(all_combinations) > max_combinations:
                # Use random sampling
                import random
                random.shuffle(all_combinations)
                all_combinations = all_combinations[:max_combinations]
            
            # Convert to list of dictionaries
            combinations = []
            for combination in all_combinations:
                param_dict = dict(zip(param_names, combination))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            self.logger.error(f"Error generating parameter combinations: {str(e)}")
            return []
    
    def _apply_parameters_to_signals(self, signals: List[Dict], params: Dict[str, Any]) -> List[Dict]:
        """Apply optimization parameters to signals"""
        try:
            modified_signals = []
            
            for signal in signals:
                modified_signal = signal.copy()
                
                # Apply parameter modifications
                if 'min_signal_strength' in params:
                    if signal.get('signal_strength', 0) < params['min_signal_strength']:
                        continue  # Skip weak signals
                
                if 'stop_loss_multiplier' in params:
                    current_sl = modified_signal.get('stop_loss', 0)
                    if current_sl:
                        modified_signal['stop_loss'] = current_sl * params['stop_loss_multiplier']
                
                if 'target_multiplier' in params:
                    current_targets = modified_signal.get('targets', [])
                    if current_targets:
                        modified_signal['targets'] = [
                            target * params['target_multiplier'] for target in current_targets
                        ]
                
                if 'risk_level_filter' in params:
                    if modified_signal.get('risk_level') not in params['risk_level_filter']:
                        continue  # Skip signals not matching risk level
                
                modified_signals.append(modified_signal)
            
            return modified_signals
            
        except Exception as e:
            self.logger.debug(f"Error applying parameters to signals: {str(e)}")
            return signals
    
    def _calculate_fitness_score(self, backtest_result: BacktestResult, metric: str) -> float:
        """Calculate fitness score for optimization"""
        try:
            if metric == 'sharpe_ratio':
                return backtest_result.sharpe_ratio
            elif metric == 'win_rate':
                return backtest_result.win_rate
            elif metric == 'profit_factor':
                return backtest_result.profit_factor
            elif metric == 'total_return':
                return backtest_result.total_return_pct
            elif metric == 'risk_adjusted_return':
                # Custom metric: return / max_drawdown
                if backtest_result.max_drawdown > 0:
                    return backtest_result.total_return_pct / backtest_result.max_drawdown
                else:
                    return backtest_result.total_return_pct
            else:
                # Default to Sharpe ratio
                return backtest_result.sharpe_ratio
                
        except Exception as e:
            self.logger.debug(f"Error calculating fitness score: {str(e)}")
            return 0.0
    
    # =============================================================================
    # 🛠️ UTILITY METHODS
    # =============================================================================
    
    def _prepare_signals_for_backtest(self, 
                                    signals: List[Dict], 
                                    start_date: datetime,
                                    end_date: datetime) -> pd.DataFrame:
        """Prepare signals for backtesting"""
        try:
            signals_data = []
            
            for signal in signals:
                signal_time = signal.get('timestamp')
                if isinstance(signal_time, str):
                    signal_time = pd.to_datetime(signal_time)
                elif isinstance(signal_time, datetime):
                    pass
                else:
                    continue
                
                if start_date <= signal_time <= end_date:
                    signals_data.append({
                        'timestamp': signal_time,
                        'symbol': signal.get('symbol', ''),
                        'type': signal.get('type', ''),
                        'signal_strength': signal.get('signal_strength', 50),
                        'stop_loss': signal.get('stop_loss'),
                        'targets': signal.get('targets', []),
                        'risk_level': signal.get('risk_level', 'MEDIUM'),
                        'timeframe': signal.get('timeframe', '1h'),
                        'signal_id': signal.get('signal_id', '')
                    })
            
            if signals_data:
                df = pd.DataFrame(signals_data)
                df = df.sort_values('timestamp')
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.debug(f"Error preparing signals for backtest: {str(e)}")
            return pd.DataFrame()
    
    def _find_candle_for_date(self, 
                            data: pd.DataFrame, 
                            target_date: datetime) -> Optional[pd.Series]:
        """Find the closest candle for a given date"""
        try:
            if data.empty:
                return None
            
            # Ensure data has datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data = data.set_index('timestamp')
                else:
                    return None
            
            # Find closest date
            closest_idx = data.index.get_indexer([target_date], method='nearest')[0]
            
            if closest_idx != -1 and closest_idx < len(data):
                return data.iloc[closest_idx]
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error finding candle for date: {str(e)}")
            return None
    
    def _calculate_position_size(self, 
                               current_capital: float, 
                               signal_strength: float,
                               risk_level: str) -> float:
        """Calculate position size based on signal strength and risk level"""
        try:
            # Base position size as percentage of capital
            base_size_pct = {
                'LOW': 0.02,     # 2%
                'MEDIUM': 0.05,  # 5%
                'HIGH': 0.08     # 8%
            }.get(risk_level, 0.05)
            
            # Adjust based on signal strength
            strength_multiplier = signal_strength / 100.0
            adjusted_size_pct = base_size_pct * strength_multiplier
            
            # Apply maximum position size limit
            final_size_pct = min(adjusted_size_pct, self.max_position_size)
            
            return current_capital * final_size_pct
            
        except Exception as e:
            self.logger.debug(f"Error calculating position size: {str(e)}")
            return current_capital * 0.02  # Default 2%
    
    def _calculate_unrealized_pnl(self, position: PositionInfo, current_price: float) -> float:
        """Calculate unrealized P&L for an active position"""
        try:
            if position.direction == 'LONG':
                price_diff = current_price - position.entry_price
            else:  # SHORT
                price_diff = position.entry_price - current_price
            
            unrealized_pnl = price_diff * position.quantity
            return unrealized_pnl
            
        except Exception as e:
            self.logger.debug(f"Error calculating unrealized P&L: {str(e)}")
            return 0.0
    
    async def _close_remaining_positions(self, 
                                       active_positions: Dict[str, PositionInfo],
                                       historical_data: Dict[str, pd.DataFrame],
                                       close_date: datetime) -> List[TradeResult]:
        """Close all remaining positions at end of backtest"""
        trades = []
        
        for symbol, position in active_positions.items():
            try:
                if symbol in historical_data:
                    symbol_data = historical_data[symbol]
                    close_candle = self._find_candle_for_date(symbol_data, close_date)
                    
                    if close_candle is not None:
                        exit_info = {
                            'exit_price': close_candle['close'],
                            'exit_time': close_date,
                            'reason': 'BACKTEST_END'
                        }
                        
                        trade_result = self._create_trade_result(position, exit_info, close_candle)
                        trades.append(trade_result)
                        
            except Exception as e:
                self.logger.debug(f"Error closing position {symbol}: {str(e)}")
        
        return trades
    
    def _create_empty_backtest_result(self, 
                                    start_date: datetime, 
                                    end_date: datetime,
                                    initial_capital: float) -> BacktestResult:
        """Create empty backtest result when no trades occurred"""
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_profit_loss=0.0,
            total_fees=0.0,
            net_profit_loss=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=timedelta(),
            sharpe_ratio=0.0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            average_trade_duration=timedelta(),
            total_return_pct=0.0,
            annual_return_pct=0.0,
            trades=[],
            equity_curve=[(start_date, initial_capital), (end_date, initial_capital)],
            monthly_returns={},
            drawdown_periods=[],
            performance_by_timeframe={},
            performance_by_symbol={}
        )
    
    # =============================================================================
    # 📈 LIVE POSITION TRACKING
    # =============================================================================
    
    def track_live_position(self, signal: Dict, entry_price: float) -> str:
        """Track a live trading position"""
        try:
            with self.position_lock:
                position_id = f"{signal['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                position = PositionInfo(
                    symbol=signal['symbol'],
                    direction=signal['type'].upper(),
                    entry_time=datetime.now(),
                    entry_price=entry_price,
                    quantity=signal.get('quantity', 1.0),
                    stop_loss=signal.get('stop_loss', entry_price * 0.98),
                    target_levels=signal.get('targets', [entry_price * 1.02]),
                    current_target_index=0,
                    signal_id=signal.get('signal_id', position_id),
                    signal_strength=signal.get('signal_strength', 70),
                    timeframe=signal.get('timeframe', '1h')
                )
                
                self.active_positions[position_id] = position
                
                self.logger.info(f"📊 Tracking live position: {position_id}")
                
                return position_id
                
        except Exception as e:
            self.logger.error(f"Error tracking live position: {str(e)}")
            return ""
    
    def update_live_position(self, position_id: str, current_price: float) -> Optional[Dict]:
        """Update live position with current price"""
        try:
            with self.position_lock:
                if position_id not in self.active_positions:
                    return None
                
                position = self.active_positions[position_id]
                position.unrealized_pnl = self._calculate_unrealized_pnl(position, current_price)
                
                # Update max profit/loss seen
                if position.unrealized_pnl > position.max_profit_seen:
                    position.max_profit_seen = position.unrealized_pnl
                if position.unrealized_pnl < position.max_loss_seen:
                    position.max_loss_seen = position.unrealized_pnl
                
                return {
                    'position_id': position_id,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100,
                    'max_profit': position.max_profit_seen,
                    'max_loss': position.max_loss_seen,
                    'duration': datetime.now() - position.entry_time
                }
                
        except Exception as e:
            self.logger.error(f"Error updating live position: {str(e)}")
            return None
    
    def close_live_position(self, position_id: str, exit_price: float, exit_reason: str) -> Optional[TradeResult]:
        """Close a live position and return trade result"""
        try:
            with self.position_lock:
                if position_id not in self.active_positions:
                    return None
                
                position = self.active_positions[position_id]
                
                # Create mock candle for trade result
                mock_candle = pd.Series({
                    'close': exit_price,
                    'high': exit_price,
                    'low': exit_price
                })
                
                exit_info = {
                    'exit_price': exit_price,
                    'exit_time': datetime.now(),
                    'reason': exit_reason
                }
                
                trade_result = self._create_trade_result(position, exit_info, mock_candle)
                
                # Remove from active positions
                del self.active_positions[position_id]
                
                # Store in historical results
                self.historical_results.append(trade_result)
                
                self.logger.info(f"💰 Closed live position {position_id}: "
                               f"{trade_result.profit_loss_pct:.2f}% ({exit_reason})")
                
                return trade_result
                
        except Exception as e:
            self.logger.error(f"Error closing live position: {str(e)}")
            return None
    
    def get_live_positions_summary(self) -> Dict[str, Any]:
        """Get summary of all live positions"""
        try:
            with self.position_lock:
                summary = {
                    'total_positions': len(self.active_positions),
                    'total_unrealized_pnl': sum(pos.unrealized_pnl for pos in self.active_positions.values()),
                    'positions': []
                }
                
                for pos_id, position in self.active_positions.items():
                    pos_summary = {
                        'position_id': pos_id,
                        'symbol': position.symbol,
                        'direction': position.direction,
                        'entry_price': position.entry_price,
                        'entry_time': position.entry_time.isoformat(),
                        'unrealized_pnl': position.unrealized_pnl,
                        'unrealized_pnl_pct': (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100,
                        'duration': str(datetime.now() - position.entry_time)
                    }
                    summary['positions'].append(pos_summary)
                
                return summary
                
        except Exception as e:
            self.logger.error(f"Error getting live positions summary: {str(e)}")
            return {'total_positions': 0, 'total_unrealized_pnl': 0, 'positions': []}

# =============================================================================
# 🧪 TESTING FUNCTION
# =============================================================================

async def test_backtesting_engine():
    """Test function for the backtesting engine"""
    print("🧪 Testing BacktestingEngine V3.0 Ultimate...")
    
    backtest_engine = BacktestingEngine()
    
    try:
        # Create sample signals
        signals = [
            {
                'timestamp': datetime.now() - timedelta(days=5),
                'symbol': 'BTC/USDT',
                'type': 'LONG',
                'signal_strength': 85,
                'stop_loss': 65000,
                'targets': [67000, 68000],
                'risk_level': 'MEDIUM',
                'timeframe': '1h'
            },
            {
                'timestamp': datetime.now() - timedelta(days=3),
                'symbol': 'ETH/USDT',
                'type': 'SHORT',
                'signal_strength': 75,
                'stop_loss': 3200,
                'targets': [3000, 2900],
                'risk_level': 'LOW',
                'timeframe': '4h'
            }
        ]
        
        print(f"📊 Created {len(signals)} test signals")
        
        # Create sample historical data
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                             end=datetime.now(), freq='1H')
        
        historical_data = {}
        for symbol in ['BTC/USDT', 'ETH/USDT']:
            # Generate realistic price data
            np.random.seed(42)
            base_price = 66000 if 'BTC' in symbol else 3100
            
            prices = []
            current_price = base_price
            
            for _ in range(len(dates)):
                # Random walk with slight upward bias
                change = np.random.normal(0.001, 0.02)  # 0.1% bias, 2% volatility
                current_price *= (1 + change)
                prices.append(current_price)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': [1000 + abs(np.random.normal(0, 200)) for _ in prices]
            })
            df.set_index('timestamp', inplace=True)
            
            historical_data[symbol] = df
        
        print(f"📈 Created historical data for {len(historical_data)} symbols")
        
        # Test backtesting
        start_date = datetime.now() - timedelta(days=6)
        end_date = datetime.now() - timedelta(days=1)
        
        result = await backtest_engine.backtest_strategy(
            signals, historical_data, start_date, end_date, 10000
        )
        
        print(f"✅ Backtest Results:")
        print(f"   Total Trades: {result.total_trades}")
        print(f"   Win Rate: {result.win_rate:.1%}")
        print(f"   Total Return: {result.total_return_pct:.2f}%")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"   Profit Factor: {result.profit_factor:.2f}")
        
        if result.trades:
            print(f"   Sample Trade:")
            trade = result.trades[0]
            print(f"     {trade.symbol} {trade.direction}: {trade.profit_loss_pct:.2f}%")
            print(f"     Duration: {trade.trade_duration}")
            print(f"     Exit Reason: {trade.exit_reason}")
        
        # Test live position tracking
        print(f"\n📊 Testing live position tracking...")
        
        test_signal = {
            'symbol': 'BTC/USDT',
            'type': 'LONG',
            'signal_strength': 80,
            'stop_loss': 65000,
            'targets': [67000],
            'quantity': 0.1
        }
        
        position_id = backtest_engine.track_live_position(test_signal, 66000)
        print(f"   Position opened: {position_id}")
        
        # Update position
        update = backtest_engine.update_live_position(position_id, 66500)
        if update:
            print(f"   Unrealized P&L: {update['unrealized_pnl_pct']:.2f}%")
        
        # Close position
        trade_result = backtest_engine.close_live_position(position_id, 66800, 'TARGET_1')
        if trade_result:
            print(f"   Position closed: {trade_result.profit_loss_pct:.2f}%")
        
        # Get positions summary
        summary = backtest_engine.get_live_positions_summary()
        print(f"   Active positions: {summary['total_positions']}")
        
        print("🎉 BacktestingEngine test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_backtesting_engine())