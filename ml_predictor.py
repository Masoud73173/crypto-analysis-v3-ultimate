# ml_predictor.py - Machine Learning Success Prediction Engine V3.0 Ultimate
import numpy as np
import pandas as pd
import logging
import json
import os
import pickle
import joblib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import traceback
import threading
import hashlib

# =============================================================================
# 🔧 CONDITIONAL IMPORTS FOR ML LIBRARIES
# =============================================================================

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
    from sklearn.utils import resample
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available - using mock ML predictor")

try:
    import config
except ImportError:
    # Fallback configuration
    class MockConfig:
        ML_CONFIG = {
            'enabled': True,
            'model_path': './models/',
            'retrain_interval_days': 7,
            'min_training_samples': 1000,
            'success_prediction_threshold': 0.75,
            'feature_importance_threshold': 0.05
        }
    config = MockConfig()

# =============================================================================
# 📊 DATA CLASSES
# =============================================================================

@dataclass
class FeatureSet:
    """Complete feature set for ML prediction"""
    # Technical indicators
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bb_position: float = 0.5
    volume_ratio: float = 1.0
    volatility: float = 0.02
    
    # Advanced technical features
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    williams_r: float = -50.0
    atr_ratio: float = 0.02
    momentum_score: float = 0.0
    
    # Market context
    trend_strength: float = 0.0
    support_distance: float = 0.05
    resistance_distance: float = 0.05
    price_position: float = 0.5  # Position in recent range
    
    # Advanced features
    confluence_count: int = 1
    pattern_score: float = 0.0
    fibonacci_confluence: float = 0.0
    
    # News sentiment
    news_sentiment: float = 0.0
    news_confidence: float = 0.0
    news_volume: int = 0
    news_impact_score: float = 0.0
    
    # Market regime
    market_regime_score: float = 0.0
    vix_level: float = 0.2
    btc_dominance: float = 0.5
    fear_greed_index: float = 50.0
    
    # Time-based features
    hour_of_day: int = 12
    day_of_week: int = 3
    trading_session: int = 1  # 0=Asian, 1=European, 2=American, 3=Off-hours
    
    # Signal characteristics
    signal_strength: float = 50.0
    risk_level_encoded: int = 1  # 0=LOW, 1=MEDIUM, 2=HIGH
    timeframe_encoded: int = 2   # 0=5m, 1=15m, 2=1h, 3=4h, 4=1d, 5=3d, 6=1w

@dataclass
class PredictionResult:
    """ML prediction result"""
    success_probability: float
    confidence: float
    feature_importance: Dict[str, float]
    risk_adjusted_score: float
    recommendation: str
    model_agreement: float
    prediction_details: Dict[str, Any]

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_mean: float
    cross_val_std: float
    feature_importance: Dict[str, float]
    confusion_matrix: List[List[int]]
    
@dataclass
class TrainingResult:
    """Training result summary"""
    models_trained: List[str]
    performance_metrics: Dict[str, ModelPerformance]
    training_samples: int
    validation_samples: int
    training_duration: float
    feature_count: int
    best_model: str

# =============================================================================
# 🤖 MACHINE LEARNING PREDICTOR CLASS
# =============================================================================

class MLPredictor:
    """Advanced machine learning-based trading signal success prediction"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.feature_importance_cache = {}
        
        # Training data buffer
        self.training_data_buffer = []
        self.max_buffer_size = 10000
        
        # Model paths
        self.model_path = config.ML_CONFIG.get('model_path', './models/')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Performance tracking
        self.prediction_count = 0
        self.successful_predictions = 0
        self.last_training_time = None
        self.model_version = "1.0.0"
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize models
        self.is_trained = False
        if SKLEARN_AVAILABLE:
            self._initialize_models()
            self._load_models()
        else:
            self.logger.warning("🤖 ML libraries not available, using rule-based prediction")
        
        self.logger.info("🤖 MLPredictor V3.0 Ultimate initialized")
    
    def _setup_logger(self):
        """Setup logging for ML predictor"""
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
    
    def _initialize_models(self):
        """Initialize ML models with optimized hyperparameters"""
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            # Get model configurations from config
            model_configs = config.ML_CONFIG.get('models', {})
            
            # Random Forest with optimized parameters
            rf_config = model_configs.get('random_forest', {})
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=rf_config.get('n_estimators', 200),
                max_depth=rf_config.get('max_depth', 15),
                min_samples_split=rf_config.get('min_samples_split', 10),
                min_samples_leaf=rf_config.get('min_samples_leaf', 5),
                max_features='sqrt',
                random_state=rf_config.get('random_state', 42),
                n_jobs=-1,
                class_weight='balanced'  # Handle imbalanced data
            )
            
            # Gradient Boosting with optimized parameters
            gb_config = model_configs.get('gradient_boosting', {})
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=gb_config.get('n_estimators', 150),
                learning_rate=gb_config.get('learning_rate', 0.1),
                max_depth=gb_config.get('max_depth', 8),
                min_samples_split=gb_config.get('min_samples_split', 15),
                min_samples_leaf=gb_config.get('min_samples_leaf', 7),
                random_state=gb_config.get('random_state', 42),
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4
            )
            
            # Logistic Regression with regularization
            lr_config = model_configs.get('logistic_regression', {})
            self.models['logistic_regression'] = LogisticRegression(
                C=lr_config.get('C', 1.0),
                solver=lr_config.get('solver', 'liblinear'),
                random_state=lr_config.get('random_state', 42),
                max_iter=lr_config.get('max_iter', 1000),
                class_weight='balanced'
            )
            
            # Initialize scalers for each model
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
            
            # Initialize feature importance cache
            self.feature_importance_cache = {name: {} for name in self.models.keys()}
            
            self.logger.info(f"🤖 Initialized {len(self.models)} ML models")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            self.models = {}
    
    # =============================================================================
    # 🎯 MAIN PREDICTION METHOD
    # =============================================================================
    
    async def predict_signal_success(self, features: FeatureSet, 
                                   signal_strength: float) -> PredictionResult:
        """Predict probability of signal success using ensemble of ML models"""
        try:
            with self.lock:
                self.prediction_count += 1
            
            if not SKLEARN_AVAILABLE or not self.is_trained:
                # Fallback to rule-based prediction
                return self._rule_based_prediction(features, signal_strength)
            
            # Convert features to array
            feature_array = self._features_to_array(features)
            feature_names = self._get_feature_names()
            
            # Get predictions from all models
            model_predictions = {}
            model_probabilities = {}
            feature_importances = {}
            
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    scaled_features = self.scalers[model_name].transform([feature_array])
                    
                    # Get prediction and probability
                    prediction = model.predict(scaled_features)[0]
                    probability = model.predict_proba(scaled_features)[0]
                    
                    # Store results
                    model_predictions[model_name] = prediction
                    model_probabilities[model_name] = probability[1] if len(probability) > 1 else 0.5
                    
                    # Get feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance_dict = dict(zip(feature_names, model.feature_importances_))
                        feature_importances[model_name] = importance_dict
                        
                except Exception as e:
                    self.logger.debug(f"Error in {model_name} prediction: {str(e)}")
                    model_predictions[model_name] = 0
                    model_probabilities[model_name] = 0.5
            
            # Ensemble prediction using weighted average
            ensemble_weights = config.ML_CONFIG.get('ensemble_weights', {
                'random_forest': 0.4,
                'gradient_boosting': 0.35,
                'logistic_regression': 0.25
            })
            
            success_probability = sum(
                model_probabilities.get(name, 0.5) * weight 
                for name, weight in ensemble_weights.items()
            )
            
            # Calculate model agreement (confidence)
            probabilities = list(model_probabilities.values())
            model_agreement = self._calculate_model_agreement(probabilities)
            
            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(
                success_probability, model_agreement, features
            )
            
            # Risk-adjusted score
            risk_adjusted_score = self._calculate_risk_adjusted_score(
                success_probability, confidence, features, signal_strength
            )
            
            # Average feature importance across models
            avg_feature_importance = self._calculate_average_feature_importance(
                feature_importances, feature_names
            )
            
            # Generate recommendation
            recommendation = self._generate_ml_recommendation(
                success_probability, confidence, risk_adjusted_score
            )
            
            # Detailed prediction information
            prediction_details = {
                'model_predictions': model_predictions,
                'model_probabilities': model_probabilities,
                'ensemble_weights': ensemble_weights,
                'feature_vector': feature_array.tolist(),
                'scaled_features': {
                    name: self.scalers[name].transform([feature_array])[0].tolist()
                    for name in self.scalers.keys()
                },
                'prediction_timestamp': datetime.now().isoformat(),
                'model_version': self.model_version
            }
            
            return PredictionResult(
                success_probability=success_probability,
                confidence=confidence,
                feature_importance=avg_feature_importance,
                risk_adjusted_score=risk_adjusted_score,
                recommendation=recommendation,
                model_agreement=model_agreement,
                prediction_details=prediction_details
            )
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {str(e)}")
            return self._rule_based_prediction(features, signal_strength)
    
    def _rule_based_prediction(self, features: FeatureSet, 
                             signal_strength: float) -> PredictionResult:
        """Fallback rule-based prediction when ML models are not available"""
        try:
            # Base probability from signal strength
            base_probability = min(0.9, max(0.1, signal_strength / 100.0))
            
            # Technical indicator adjustments
            adjustments = 0.0
            
            # RSI adjustment
            if 20 <= features.rsi <= 30 or 70 <= features.rsi <= 80:
                adjustments += 0.1  # Good RSI levels
            elif features.rsi < 20 or features.rsi > 80:
                adjustments += 0.15  # Extreme RSI levels
            
            # MACD adjustment
            if (features.macd > features.macd_signal and features.rsi < 70) or \
               (features.macd < features.macd_signal and features.rsi > 30):
                adjustments += 0.08  # MACD alignment
            
            # Volume confirmation
            if features.volume_ratio > 1.5:
                adjustments += 0.06
            elif features.volume_ratio > 2.0:
                adjustments += 0.10
            
            # Volatility adjustment
            if features.volatility > 0.05:  # High volatility
                adjustments -= 0.05
            elif features.volatility < 0.02:  # Low volatility
                adjustments += 0.03
            
            # News sentiment adjustment
            if features.news_confidence > 0.5:
                adjustments += features.news_sentiment * features.news_confidence * 0.1
            
            # Confluence bonus
            if features.confluence_count >= 3:
                adjustments += min(features.confluence_count * 0.02, 0.1)
            
            # Pattern score adjustment
            if features.pattern_score > 0.7:
                adjustments += 0.08
            
            # Market regime adjustment
            if abs(features.market_regime_score) > 0.5:
                # Strong market regime
                regime_alignment = features.market_regime_score * (1 if signal_strength > 50 else -1)
                adjustments += regime_alignment * 0.05
            
            # Final probability
            success_probability = min(0.95, max(0.05, base_probability + adjustments))
            
            # Simple confidence calculation
            confidence = min(0.8, 0.4 + (features.confluence_count * 0.1) + (features.news_confidence * 0.2))
            
            # Risk adjustment
            risk_factors = []
            if features.volatility > 0.05:
                risk_factors.append('high_volatility')
            if features.news_confidence < 0.3 and abs(features.news_sentiment) > 0.3:
                risk_factors.append('uncertain_news')
            if features.confluence_count < 2:
                risk_factors.append('low_confluence')
            
            risk_adjusted_score = success_probability * (1 - len(risk_factors) * 0.1)
            
            # Feature importance (mock)
            feature_importance = {
                'signal_strength': 0.25,
                'rsi': 0.15,
                'volume_ratio': 0.12,
                'news_sentiment': 0.10,
                'confluence_count': 0.08,
                'macd': 0.08,
                'volatility': 0.06,
                'pattern_score': 0.05,
                'market_regime_score': 0.05,
                'others': 0.06
            }
            
            recommendation = self._generate_ml_recommendation(
                success_probability, confidence, risk_adjusted_score
            )
            
            prediction_details = {
                'method': 'rule_based',
                'base_probability': base_probability,
                'adjustments': adjustments,
                'risk_factors': risk_factors,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return PredictionResult(
                success_probability=success_probability,
                confidence=confidence,
                feature_importance=feature_importance,
                risk_adjusted_score=risk_adjusted_score,
                recommendation=recommendation,
                model_agreement=confidence,  # Use confidence as agreement proxy
                prediction_details=prediction_details
            )
            
        except Exception as e:
            self.logger.error(f"Error in rule-based prediction: {str(e)}")
            # Ultra-fallback
            return PredictionResult(
                success_probability=0.65,
                confidence=0.5,
                feature_importance={'signal_strength': 1.0},
                risk_adjusted_score=0.6,
                recommendation='MODERATE_BUY',
                model_agreement=0.5,
                prediction_details={'method': 'fallback', 'error': str(e)}
            )
    
    # =============================================================================
    # 📊 FEATURE ENGINEERING
    # =============================================================================
    
    def _features_to_array(self, features: FeatureSet) -> np.ndarray:
        """Convert FeatureSet to numpy array"""
        try:
            # Define feature order (must be consistent)
            feature_values = [
                # Technical indicators
                features.rsi,
                features.macd,
                features.macd_signal,
                features.bb_position,
                features.volume_ratio,
                features.volatility,
                features.stoch_k,
                features.stoch_d,
                features.williams_r,
                features.atr_ratio,
                features.momentum_score,
                
                # Market context
                features.trend_strength,
                features.support_distance,
                features.resistance_distance,
                features.price_position,
                
                # Advanced features
                features.confluence_count,
                features.pattern_score,
                features.fibonacci_confluence,
                
                # News sentiment
                features.news_sentiment,
                features.news_confidence,
                features.news_volume,
                features.news_impact_score,
                
                # Market regime
                features.market_regime_score,
                features.vix_level,
                features.btc_dominance,
                features.fear_greed_index,
                
                # Time-based features
                features.hour_of_day,
                features.day_of_week,
                features.trading_session,
                
                # Signal characteristics
                features.signal_strength,
                features.risk_level_encoded,
                features.timeframe_encoded
            ]
            
            return np.array(feature_values, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error converting features to array: {str(e)}")
            # Return zero array as fallback
            return np.zeros(32, dtype=np.float32)
    
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names in consistent order"""
        return [
            # Technical indicators
            'rsi', 'macd', 'macd_signal', 'bb_position', 'volume_ratio',
            'volatility', 'stoch_k', 'stoch_d', 'williams_r', 'atr_ratio',
            'momentum_score',
            
            # Market context
            'trend_strength', 'support_distance', 'resistance_distance',
            'price_position',
            
            # Advanced features
            'confluence_count', 'pattern_score', 'fibonacci_confluence',
            
            # News sentiment
            'news_sentiment', 'news_confidence', 'news_volume',
            'news_impact_score',
            
            # Market regime
            'market_regime_score', 'vix_level', 'btc_dominance',
            'fear_greed_index',
            
            # Time-based features
            'hour_of_day', 'day_of_week', 'trading_session',
            
            # Signal characteristics
            'signal_strength', 'risk_level_encoded', 'timeframe_encoded'
        ]
    
    def _enhance_features(self, features: FeatureSet) -> FeatureSet:
        """Add derived features to improve prediction accuracy"""
        try:
            # RSI momentum
            if hasattr(features, 'rsi_previous'):
                features.rsi_momentum = features.rsi - features.rsi_previous
            
            # MACD divergence
            features.macd_divergence = features.macd - features.macd_signal
            
            # Volume-price trend
            features.volume_price_trend = features.volume_ratio * features.momentum_score
            
            # News-technical alignment
            if features.news_confidence > 0.3:
                signal_direction = 1 if features.signal_strength > 50 else -1
                features.news_technical_alignment = features.news_sentiment * signal_direction
            else:
                features.news_technical_alignment = 0.0
            
            # Volatility-adjusted confidence
            volatility_adjustment = 1.0 - min(features.volatility * 10, 0.5)
            features.volatility_adjusted_confidence = features.news_confidence * volatility_adjustment
            
            return features
            
        except Exception as e:
            self.logger.debug(f"Error enhancing features: {str(e)}")
            return features
    
    # =============================================================================
    # 🧮 CALCULATION METHODS
    # =============================================================================
    
    def _calculate_model_agreement(self, probabilities: List[float]) -> float:
        """Calculate agreement between model predictions"""
        try:
            if len(probabilities) < 2:
                return 0.5
            
            # Calculate standard deviation of probabilities
            mean_prob = np.mean(probabilities)
            std_prob = np.std(probabilities)
            
            # High agreement = low standard deviation
            max_possible_std = 0.5  # Maximum possible std for probabilities
            agreement = 1.0 - min(std_prob / max_possible_std, 1.0)
            
            return agreement
            
        except Exception as e:
            self.logger.debug(f"Error calculating model agreement: {str(e)}")
            return 0.5
    
    def _calculate_prediction_confidence(self, success_probability: float, 
                                       model_agreement: float, 
                                       features: FeatureSet) -> float:
        """Calculate overall prediction confidence"""
        try:
            # Base confidence from model agreement
            base_confidence = model_agreement
            
            # Adjustment based on prediction extremeness
            extremeness = abs(success_probability - 0.5) * 2  # 0 to 1 scale
            extremeness_bonus = extremeness * 0.2
            
            # Feature quality adjustment
            feature_quality = 0.0
            
            # News confidence bonus
            if features.news_confidence > 0.5:
                feature_quality += features.news_confidence * 0.2
            
            # Confluence bonus
            confluence_quality = min(features.confluence_count / 5.0, 1.0) * 0.2
            feature_quality += confluence_quality
            
            # Pattern confidence bonus
            if features.pattern_score > 0.7:
                feature_quality += 0.1
            
            # Volume confirmation bonus
            if features.volume_ratio > 1.5:
                feature_quality += 0.1
            
            # Combine all factors
            total_confidence = (
                base_confidence * 0.5 +
                extremeness_bonus +
                feature_quality
            )
            
            return min(0.95, max(0.1, total_confidence))
            
        except Exception as e:
            self.logger.debug(f"Error calculating prediction confidence: {str(e)}")
            return 0.5
    
    def _calculate_risk_adjusted_score(self, success_probability: float, 
                                     confidence: float, 
                                     features: FeatureSet, 
                                     signal_strength: float) -> float:
        """Calculate risk-adjusted prediction score"""
        try:
            base_score = success_probability * confidence
            
            # Risk factors that reduce the score
            risk_adjustments = 0.0
            
            # High volatility penalty
            if features.volatility > 0.05:
                volatility_penalty = min(0.2, (features.volatility - 0.05) * 4)
                risk_adjustments -= volatility_penalty
            
            # Low confidence news penalty
            if features.news_confidence < 0.3 and abs(features.news_sentiment) > 0.3:
                risk_adjustments -= 0.1
            
            # Low confluence penalty
            if features.confluence_count < 2:
                risk_adjustments -= 0.1
            
            # Market regime uncertainty
            if abs(features.market_regime_score) < 0.2:  # Uncertain regime
                risk_adjustments -= 0.05
            
            # Reward factors that increase the score
            reward_adjustments = 0.0
            
            # High confluence bonus
            if features.confluence_count >= 4:
                confluence_bonus = min(0.15, (features.confluence_count - 3) * 0.05)
                reward_adjustments += confluence_bonus
            
            # Strong pattern bonus
            if features.pattern_score > 0.8:
                reward_adjustments += 0.1
            
            # Volume confirmation bonus
            if features.volume_ratio > 2.0:
                reward_adjustments += 0.05
            
            # News alignment bonus
            if (features.news_confidence > 0.7 and 
                abs(features.news_sentiment) > 0.3 and
                ((features.news_sentiment > 0 and signal_strength > 50) or
                 (features.news_sentiment < 0 and signal_strength < 50))):
                reward_adjustments += 0.1
            
            # Final risk-adjusted score
            final_score = base_score + risk_adjustments + reward_adjustments
            
            return min(0.95, max(0.05, final_score))
            
        except Exception as e:
            self.logger.debug(f"Error calculating risk-adjusted score: {str(e)}")
            return success_probability * 0.8  # Conservative fallback
    
    def _calculate_average_feature_importance(self, 
                                            feature_importances: Dict[str, Dict[str, float]], 
                                            feature_names: List[str]) -> Dict[str, float]:
        """Calculate average feature importance across models"""
        try:
            if not feature_importances:
                # Return uniform importance as fallback
                uniform_importance = 1.0 / len(feature_names)
                return {name: uniform_importance for name in feature_names}
            
            avg_importance = {}
            
            for feature_name in feature_names:
                importances = []
                for model_importance in feature_importances.values():
                    if feature_name in model_importance:
                        importances.append(model_importance[feature_name])
                
                if importances:
                    avg_importance[feature_name] = np.mean(importances)
                else:
                    avg_importance[feature_name] = 0.0
            
            # Normalize to sum to 1
            total_importance = sum(avg_importance.values())
            if total_importance > 0:
                avg_importance = {
                    name: importance / total_importance 
                    for name, importance in avg_importance.items()
                }
            
            return avg_importance
            
        except Exception as e:
            self.logger.debug(f"Error calculating average feature importance: {str(e)}")
            return {}
    
    def _generate_ml_recommendation(self, success_probability: float, 
                                  confidence: float, 
                                  risk_adjusted_score: float) -> str:
        """Generate trading recommendation based on ML prediction"""
        try:
            # Strong signals with high confidence
            if risk_adjusted_score >= 0.8 and confidence >= 0.75:
                return "STRONG_BUY"
            elif risk_adjusted_score >= 0.7 and confidence >= 0.65:
                return "BUY"
            elif risk_adjusted_score >= 0.6 and confidence >= 0.55:
                return "MODERATE_BUY"
            elif risk_adjusted_score >= 0.45:
                return "NEUTRAL"
            elif risk_adjusted_score >= 0.3:
                return "WEAK"
            else:
                return "AVOID"
                
        except Exception as e:
            self.logger.debug(f"Error generating recommendation: {str(e)}")
            return "NEUTRAL"
    
    # =============================================================================
    # 🎓 MODEL TRAINING AND MANAGEMENT
    # =============================================================================
    
    async def train_models(self, training_data: List[Dict]) -> TrainingResult:
        """Train ML models with historical trading data"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("🤖 scikit-learn not available, cannot train models")
            return TrainingResult([], {}, 0, 0, 0.0, 0, "none")
        
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🎓 Starting model training with {len(training_data)} samples")
            
            if len(training_data) < config.ML_CONFIG.get('min_training_samples', 100):
                self.logger.warning(f"Insufficient training data: {len(training_data)}")
                return TrainingResult([], {}, 0, 0, 0.0, 0, "insufficient_data")
            
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if X.shape[0] == 0:
                self.logger.error("No valid training samples")
                return TrainingResult([], {}, 0, 0, 0.0, 0, "no_valid_samples")
            
            # Split data
            test_size = config.ML_CONFIG.get('test_size_ratio', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            training_results = {}
            trained_models = []
            
            # Train each model
            for model_name, model in self.models.items():
                try:
                    self.logger.info(f"🎓 Training {model_name}...")
                    
                    # Scale features
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_test_scaled = self.scalers[model_name].transform(X_test)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    performance = self._evaluate_model(
                        model, X_train_scaled, X_test_scaled, y_train, y_test
                    )
                    
                    training_results[model_name] = performance
                    trained_models.append(model_name)
                    
                    self.logger.info(f"✅ {model_name} trained - Accuracy: {performance.accuracy:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            if trained_models:
                # Save models
                self._save_models()
                self.is_trained = True
                self.last_training_time = datetime.now()
                
                # Find best performing model
                best_model = max(training_results.keys(), 
                               key=lambda k: training_results[k].accuracy)
                
                training_duration = (datetime.now() - start_time).total_seconds()
                
                result = TrainingResult(
                    models_trained=trained_models,
                    performance_metrics=training_results,
                    training_samples=X_train.shape[0],
                    validation_samples=X_test.shape[0],
                    training_duration=training_duration,
                    feature_count=X.shape[1],
                    best_model=best_model
                )
                
                self.logger.info(f"🎉 Training completed: {len(trained_models)} models, best: {best_model}")
                
                return result
            else:
                self.logger.error("No models were successfully trained")
                return TrainingResult([], {}, 0, 0, 0.0, 0, "training_failed")
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            self.logger.error(traceback.format_exc())
            return TrainingResult([], {}, 0, 0, 0.0, 0, "error")
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""
        try:
            features_list = []
            labels_list = []
            
            for sample in training_data:
                try:
                    # Extract features
                    feature_dict = sample.get('features', {})
                    
                    # Create FeatureSet with defaults for missing values
                    feature_set = FeatureSet(
                        rsi=feature_dict.get('rsi', 50),
                        macd=feature_dict.get('macd', 0),
                        macd_signal=feature_dict.get('macd_signal', 0),
                        bb_position=feature_dict.get('bb_position', 0.5),
                        volume_ratio=feature_dict.get('volume_ratio', 1.0),
                        volatility=feature_dict.get('volatility', 0.02),
                        stoch_k=feature_dict.get('stoch_k', 50),
                        stoch_d=feature_dict.get('stoch_d', 50),
                        williams_r=feature_dict.get('williams_r', -50),
                        atr_ratio=feature_dict.get('atr_ratio', 0.02),
                        momentum_score=feature_dict.get('momentum_score', 0),
                        trend_strength=feature_dict.get('trend_strength', 0),
                        support_distance=feature_dict.get('support_distance', 0.05),
                        resistance_distance=feature_dict.get('resistance_distance', 0.05),
                        price_position=feature_dict.get('price_position', 0.5),
                        confluence_count=feature_dict.get('confluence_count', 1),
                        pattern_score=feature_dict.get('pattern_score', 0),
                        fibonacci_confluence=feature_dict.get('fibonacci_confluence', 0),
                        news_sentiment=feature_dict.get('news_sentiment', 0),
                        news_confidence=feature_dict.get('news_confidence', 0),
                        news_volume=feature_dict.get('news_volume', 0),
                        news_impact_score=feature_dict.get('news_impact_score', 0),
                        market_regime_score=feature_dict.get('market_regime_score', 0),
                        vix_level=feature_dict.get('vix_level', 0.2),
                        btc_dominance=feature_dict.get('btc_dominance', 0.5),
                        fear_greed_index=feature_dict.get('fear_greed_index', 50),
                        hour_of_day=feature_dict.get('hour_of_day', 12),
                        day_of_week=feature_dict.get('day_of_week', 3),
                        trading_session=feature_dict.get('trading_session', 1),
                        signal_strength=feature_dict.get('signal_strength', 50),
                        risk_level_encoded=feature_dict.get('risk_level_encoded', 1),
                        timeframe_encoded=feature_dict.get('timeframe_encoded', 2)
                    )
                    
                    feature_array = self._features_to_array(feature_set)
                    features_list.append(feature_array)
                    
                    # Extract label (success/failure)
                    success = sample.get('success', False)
                    labels_list.append(1 if success else 0)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing training sample: {str(e)}")
                    continue
            
            if not features_list:
                return np.array([]), np.array([])
            
            X = np.array(features_list)
            y = np.array(labels_list)
            
            # Handle data quality issues
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            
            self.logger.info(f"📊 Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return np.array([]), np.array([])
    
    def _evaluate_model(self, model, X_train_scaled: np.ndarray, X_test_scaled: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Evaluate model performance comprehensively"""
        try:
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classification report
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.5
            
            # Cross-validation
            cv_folds = config.ML_CONFIG.get('cross_validation_folds', 5)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
            
            # Feature importance
            feature_names = self._get_feature_names()
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            return ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                cross_val_mean=cv_scores.mean(),
                cross_val_std=cv_scores.std(),
                feature_importance=feature_importance,
                confusion_matrix=cm
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return ModelPerformance(0, 0, 0, 0, 0.5, 0, 0, {}, [[0, 0], [0, 0]])
    
    # =============================================================================
    # 💾 MODEL PERSISTENCE
    # =============================================================================
    
    def _save_models(self):
        """Save trained models and scalers to disk"""
        try:
            model_files_saved = 0
            
            for model_name, model in self.models.items():
                try:
                    # Save model
                    model_file = os.path.join(self.model_path, f"{model_name}.pkl")
                    joblib.dump(model, model_file)
                    
                    # Save scaler
                    scaler_file = os.path.join(self.model_path, f"{model_name}_scaler.pkl")
                    joblib.dump(self.scalers[model_name], scaler_file)
                    
                    model_files_saved += 1
                    
                except Exception as e:
                    self.logger.error(f"Error saving {model_name}: {str(e)}")
            
            # Save metadata
            metadata = {
                'model_version': self.model_version,
                'trained_at': datetime.now().isoformat(),
                'feature_names': self._get_feature_names(),
                'models_saved': list(self.models.keys()),
                'sklearn_available': SKLEARN_AVAILABLE,
                'prediction_count': self.prediction_count,
                'successful_predictions': self.successful_predictions
            }
            
            metadata_file = os.path.join(self.model_path, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"💾 Saved {model_files_saved} models and metadata")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def _load_models(self):
        """Load trained models and scalers from disk"""
        try:
            metadata_file = os.path.join(self.model_path, "metadata.json")
            if not os.path.exists(metadata_file):
                self.logger.info("No trained models found")
                return
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load models and scalers
            models_loaded = 0
            for model_name in metadata.get('models_saved', []):
                try:
                    if model_name in self.models:
                        # Load model
                        model_file = os.path.join(self.model_path, f"{model_name}.pkl")
                        if os.path.exists(model_file):
                            self.models[model_name] = joblib.load(model_file)
                        
                        # Load scaler
                        scaler_file = os.path.join(self.model_path, f"{model_name}_scaler.pkl")
                        if os.path.exists(scaler_file):
                            self.scalers[model_name] = joblib.load(scaler_file)
                        
                        models_loaded += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error loading {model_name}: {str(e)}")
            
            if models_loaded > 0:
                self.is_trained = True
                self.model_version = metadata.get('model_version', '1.0.0')
                self.logger.info(f"📥 Loaded {models_loaded} trained models (v{self.model_version})")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
    
    # =============================================================================
    # 📊 PERFORMANCE TRACKING
    # =============================================================================
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        try:
            performance = {
                'status': 'trained' if self.is_trained else 'not_trained',
                'sklearn_available': SKLEARN_AVAILABLE,
                'models': list(self.models.keys()),
                'model_version': self.model_version,
                'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
                'prediction_count': self.prediction_count,
                'successful_predictions': self.successful_predictions,
                'model_path': self.model_path
            }
            
            # Add success rate if we have predictions
            if self.prediction_count > 0:
                performance['prediction_success_rate'] = self.successful_predictions / self.prediction_count
            
            # Check if models need retraining
            if self.last_training_time:
                days_since_training = (datetime.now() - self.last_training_time).days
                retrain_interval = config.ML_CONFIG.get('retrain_interval_days', 7)
                performance['needs_retraining'] = days_since_training >= retrain_interval
                performance['days_since_training'] = days_since_training
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error getting model performance: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def update_training_data(self, signal_outcome: Dict):
        """Add new signal outcome to training data buffer"""
        try:
            with self.lock:
                self.training_data_buffer.append({
                    'timestamp': datetime.now().isoformat(),
                    **signal_outcome
                })
                
                # Keep buffer size manageable
                if len(self.training_data_buffer) > self.max_buffer_size:
                    self.training_data_buffer = self.training_data_buffer[-self.max_buffer_size:]
                
                # Check if we should retrain
                buffer_size = len(self.training_data_buffer)
                if buffer_size >= 100 and buffer_size % 100 == 0:  # Every 100 new samples
                    self.logger.info(f"🎓 Triggering model retraining with {buffer_size} new samples")
                    # Schedule retraining (in production, this would be done asynchronously)
                    asyncio.create_task(self._retrain_models())
            
        except Exception as e:
            self.logger.error(f"Error updating training data: {str(e)}")
    
    async def _retrain_models(self):
        """Retrain models with new data (async background task)"""
        try:
            if len(self.training_data_buffer) < 50:
                return
            
            self.logger.info("🔄 Starting model retraining...")
            result = await self.train_models(self.training_data_buffer)
            
            if result.models_trained:
                self.logger.info(f"✅ Retraining completed: {len(result.models_trained)} models")
                # Clear buffer after successful retraining
                with self.lock:
                    self.training_data_buffer = []
            else:
                self.logger.warning("⚠️ Retraining failed")
                
        except Exception as e:
            self.logger.error(f"Error in model retraining: {str(e)}")

# =============================================================================
# 🧪 TESTING FUNCTION
# =============================================================================

async def test_ml_predictor():
    """Test function for the ML predictor"""
    print("🧪 Testing MLPredictor V3.0 Ultimate...")
    
    predictor = MLPredictor()
    
    try:
        # Create test features
        test_features = FeatureSet(
            rsi=25.0,              # Oversold
            macd=5.0,              # Bullish
            macd_signal=3.0,
            bb_position=0.1,       # Near lower band
            volume_ratio=2.0,      # High volume
            volatility=0.03,       # Normal volatility
            confluence_count=4,    # Strong confluence
            news_sentiment=0.3,    # Positive news
            news_confidence=0.8,   # High confidence
            pattern_score=0.9,     # Strong pattern
            signal_strength=85.0   # Strong signal
        )
        
        # Test prediction
        result = await predictor.predict_signal_success(test_features, 85.0)
        
        print(f"✅ Prediction Results:")
        print(f"   Success Probability: {result.success_probability:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Risk-Adjusted Score: {result.risk_adjusted_score:.3f}")
        print(f"   Recommendation: {result.recommendation}")
        print(f"   Model Agreement: {result.model_agreement:.3f}")
        
        if result.feature_importance:
            print(f"   Top Features:")
            sorted_features = sorted(result.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                print(f"     {feature}: {importance:.3f}")
        
        # Test performance metrics
        performance = predictor.get_model_performance()
        print(f"✅ Model Performance:")
        print(f"   Status: {performance['status']}")
        print(f"   Models: {performance['models']}")
        print(f"   Predictions: {performance['prediction_count']}")
        
        print("🎉 MLPredictor test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_ml_predictor())