# news_analyzer.py - Advanced News Sentiment Analysis for Crypto Markets V3.0 Ultimate
import asyncio
import aiohttp
import json
import re
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import os
import traceback

try:
    import config
except ImportError:
    # Fallback configuration if config not available
    class MockConfig:
        NEWS_SENTIMENT_CONFIG = {
            'enabled': True,
            'api_sources': ['newsapi', 'cryptocompare', 'coindesk'],
            'sentiment_weight': 0.2,
            'news_hours_lookback': 24,
            'min_news_count': 3,
            'confidence_threshold': 0.3,
            'cache_duration_minutes': 30
        }
        NEWS_API_KEY = None
        CRYPTOCOMPARE_API_KEY = None
        CRYPTOPANIC_API_KEY = None
    
    config = MockConfig()

# =============================================================================
# 📊 DATA CLASSES
# =============================================================================

@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    published_at: datetime
    sentiment_score: float
    relevance_score: float
    impact_level: str
    url: str = ""
    author: str = ""

@dataclass
class SentimentAnalysis:
    overall_sentiment: float  # -1 to +1
    news_count: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    confidence: float
    key_events: List[str]
    impact_level: str
    sentiment_distribution: Dict[str, int] = None
    source_breakdown: Dict[str, float] = None
    time_decay_factor: float = 1.0

# =============================================================================
# 📰 NEWS SENTIMENT ANALYZER CLASS
# =============================================================================

class NewsAnalyzer:
    """Advanced news sentiment analysis for crypto markets with multiple sources"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.session = None
        self.cache = {}
        self.sentiment_keywords = self._load_sentiment_keywords()
        self.crypto_terms = self._load_crypto_terms()
        self.impact_keywords = self._load_impact_keywords()
        
        # API endpoints
        self.endpoints = {
            'newsapi': 'https://newsapi.org/v2/everything',
            'cryptocompare': 'https://min-api.cryptocompare.com/data/v2/news/',
            'coindesk_rss': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cryptopanic': 'https://cryptopanic.com/api/v1/posts/',
            'cointelegraph_rss': 'https://cointelegraph.com/rss',
            'decrypt_rss': 'https://decrypt.co/feed'
        }
        
        # Rate limiting
        self.request_delays = {
            'newsapi': 1.0,      # 1 second between requests
            'cryptocompare': 0.5, # 0.5 seconds
            'rss_feeds': 2.0,    # 2 seconds for RSS
            'cryptopanic': 1.5   # 1.5 seconds
        }
        
        self.logger.info("📰 NewsAnalyzer V3.0 Ultimate initialized")
    
    def _setup_logger(self):
        """Setup logging for news analyzer"""
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
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Crypto-Analysis-Bot/3.0 (News Sentiment Analysis)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _load_sentiment_keywords(self) -> Dict[str, List[str]]:
        """Load comprehensive sentiment keyword dictionaries"""
        return {
            'extremely_positive': [
                'moon', 'rocket', 'massive surge', 'explosive growth', 'breakthrough',
                'revolutionary', 'game-changer', 'unprecedented', 'skyrocketing'
            ],
            'positive': [
                'bullish', 'pump', 'surge', 'rally', 'breakout', 'bull', 'green',
                'gains', 'profit', 'buy', 'institutional', 'adoption', 'partnership',
                'upgrade', 'launch', 'success', 'milestone', 'achievement', 'growth',
                'rising', 'up', 'increase', 'positive', 'strong', 'boost', 'soars',
                'climbs', 'jumps', 'bullish momentum', 'upward trend', 'buying pressure'
            ],
            'neutral': [
                'stable', 'sideways', 'consolidation', 'range', 'waiting', 'pending',
                'analysis', 'review', 'study', 'research', 'neutral', 'mixed'
            ],
            'negative': [
                'bearish', 'crash', 'dump', 'fall', 'decline', 'drop', 'correction',
                'bear', 'red', 'loss', 'sell', 'liquidation', 'hack', 'regulation',
                'ban', 'restriction', 'fear', 'uncertainty', 'panic', 'concern',
                'warning', 'risk', 'plunge', 'tumble', 'slides', 'weakness',
                'pressure', 'selling pressure', 'downward trend', 'bearish momentum'
            ],
            'extremely_negative': [
                'collapse', 'catastrophic', 'devastating', 'massive crash', 'wipeout',
                'destruction', 'disaster', 'emergency', 'crisis', 'meltdown'
            ]
        }
    
    def _load_crypto_terms(self) -> List[str]:
        """Load cryptocurrency-related terms for relevance scoring"""
        return [
            # Major cryptocurrencies
            'bitcoin', 'btc', 'ethereum', 'eth', 'binance', 'bnb', 'cardano', 'ada',
            'solana', 'sol', 'polkadot', 'dot', 'chainlink', 'link', 'litecoin', 'ltc',
            'polygon', 'matic', 'avalanche', 'avax', 'cosmos', 'atom', 'algorand', 'algo',
            
            # General crypto terms
            'crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft', 'altcoin',
            'trading', 'exchange', 'wallet', 'mining', 'staking', 'yield farming',
            'liquidity', 'smart contract', 'dapp', 'web3', 'metaverse', 'dao',
            
            # Market terms
            'market cap', 'volume', 'price action', 'technical analysis', 'candlestick',
            'support', 'resistance', 'trend', 'volatility', 'correlation'
        ]
    
    def _load_impact_keywords(self) -> Dict[str, List[str]]:
        """Load keywords that indicate market impact level"""
        return {
            'high_impact': [
                'sec', 'regulation', 'government', 'federal', 'legal', 'lawsuit',
                'etf', 'institutional', 'whale', 'major exchange', 'binance', 'coinbase',
                'tesla', 'microstrategy', 'blackrock', 'fidelity', 'grayscale',
                'fork', 'upgrade', 'hack', 'security breach', 'partnership',
                'acquisition', 'merger', 'listing', 'delisting'
            ],
            'medium_impact': [
                'price', 'trading', 'volume', 'market', 'analysis', 'forecast',
                'prediction', 'bull', 'bear', 'trend', 'technical', 'fundamental',
                'developer', 'community', 'update', 'release', 'announcement'
            ],
            'low_impact': [
                'opinion', 'discussion', 'social media', 'tweet', 'reddit',
                'influencer', 'analyst', 'commentary', 'blog', 'article'
            ]
        }
    
    # =============================================================================
    # 🎯 MAIN SENTIMENT ANALYSIS METHOD
    # =============================================================================
    
    async def get_crypto_news_sentiment(self, symbol: str) -> SentimentAnalysis:
        """Get comprehensive news sentiment analysis for a cryptocurrency"""
        try:
            self.logger.info(f"📰 Analyzing news sentiment for {symbol}")
            
            # Extract coin name from symbol
            coin_name = self._extract_coin_name(symbol)
            
            # Check cache first
            cache_key = f"news_sentiment_{coin_name.lower()}"
            if self._is_cache_valid(cache_key):
                self.logger.debug(f"📱 Using cached sentiment for {coin_name}")
                return self.cache[cache_key]
            
            # Gather news from multiple sources
            news_items = await self._gather_news_from_all_sources(coin_name)
            
            if not news_items or len(news_items) < config.NEWS_SENTIMENT_CONFIG.get('min_news_count', 3):
                self.logger.info(f"📰 Insufficient news for {coin_name}, returning neutral sentiment")
                return self._create_neutral_sentiment(f"Insufficient news data for {coin_name}")
            
            # Analyze sentiment comprehensively
            sentiment_analysis = await self._analyze_comprehensive_sentiment(news_items, coin_name)
            
            # Cache results
            self._cache_result(cache_key, sentiment_analysis)
            
            self.logger.info(f"✅ Sentiment analysis complete for {coin_name}: {sentiment_analysis.overall_sentiment:+.2f}")
            
            return sentiment_analysis
            
        except Exception as e:
            self.logger.error(f"💥 Error in news sentiment analysis for {symbol}: {str(e)}")
            return self._create_neutral_sentiment(f"Error analyzing {symbol}")
    
    def _extract_coin_name(self, symbol: str) -> str:
        """Extract coin name from trading symbol"""
        # Remove trading pairs
        coin_name = symbol.split('/')[0] if '/' in symbol else symbol
        
        # Map common symbols to full names
        symbol_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'BNB': 'binance',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOT': 'polkadot',
            'LINK': 'chainlink',
            'LTC': 'litecoin',
            'MATIC': 'polygon',
            'AVAX': 'avalanche',
            'ATOM': 'cosmos',
            'ALGO': 'algorand'
        }
        
        return symbol_mapping.get(coin_name.upper(), coin_name.lower())
    
    # =============================================================================
    # 📡 NEWS SOURCE INTEGRATION
    # =============================================================================
    
    async def _gather_news_from_all_sources(self, coin_name: str) -> List[NewsItem]:
        """Gather news from all available sources"""
        all_news = []
        
        # Create tasks for all news sources
        tasks = [
            self._fetch_newsapi_data(coin_name),
            self._fetch_cryptocompare_news(coin_name),
            self._fetch_rss_feeds(coin_name),
            self._fetch_cryptopanic_news(coin_name),
            self._fetch_additional_sources(coin_name)
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                self.logger.debug(f"News source error: {str(result)}")
        
        # Remove duplicates and sort by relevance
        unique_news = self._deduplicate_news(all_news)
        
        # Sort by relevance and recency
        sorted_news = sorted(unique_news, 
                           key=lambda x: (x.relevance_score * self._calculate_time_decay(x.published_at)), 
                           reverse=True)
        
        return sorted_news[:50]  # Return top 50 most relevant articles
    
    async def _fetch_newsapi_data(self, coin_name: str) -> List[NewsItem]:
        """Fetch news from NewsAPI"""
        try:
            api_key = getattr(config, 'NEWS_API_KEY', None)
            if not api_key:
                self.logger.debug("NewsAPI key not configured")
                return []
            
            # Build search query
            query_terms = [coin_name]
            if coin_name == 'bitcoin':
                query_terms.extend(['btc', 'bitcoin'])
            elif coin_name == 'ethereum':
                query_terms.extend(['eth', 'ethereum'])
            
            query = ' OR '.join(query_terms) + ' AND cryptocurrency'
            
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(hours=24)).isoformat(),
                'pageSize': 20,
                'apiKey': api_key
            }
            
            async with self.session.get(self.endpoints['newsapi'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    
                    news_items = []
                    for article in articles:
                        news_item = self._parse_newsapi_article(article, coin_name)
                        if news_item and news_item.relevance_score > 0.3:
                            news_items.append(news_item)
                    
                    self.logger.debug(f"📰 NewsAPI: {len(news_items)} relevant articles")
                    await asyncio.sleep(self.request_delays['newsapi'])
                    return news_items
                else:
                    self.logger.warning(f"NewsAPI error: {response.status}")
            
        except Exception as e:
            self.logger.debug(f"NewsAPI fetch error: {str(e)}")
        
        return []
    
    async def _fetch_cryptocompare_news(self, coin_name: str) -> List[NewsItem]:
        """Fetch crypto-specific news from CryptoCompare"""
        try:
            params = {
                'lang': 'EN',
                'sortOrder': 'latest',
                'lmt': 30
            }
            
            async with self.session.get(self.endpoints['cryptocompare'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('Data', [])
                    
                    relevant_news = []
                    for article in articles:
                        news_item = self._parse_cryptocompare_article(article, coin_name)
                        if news_item and news_item.relevance_score > 0.2:
                            relevant_news.append(news_item)
                    
                    self.logger.debug(f"📊 CryptoCompare: {len(relevant_news)} relevant articles")
                    await asyncio.sleep(self.request_delays['cryptocompare'])
                    return relevant_news
                    
        except Exception as e:
            self.logger.debug(f"CryptoCompare fetch error: {str(e)}")
        
        return []
    
    async def _fetch_rss_feeds(self, coin_name: str) -> List[NewsItem]:
        """Fetch news from RSS feeds"""
        news_items = []
        
        rss_feeds = [
            ('CoinDesk', self.endpoints['coindesk_rss']),
            ('CoinTelegraph', self.endpoints['cointelegraph_rss']),
            ('Decrypt', self.endpoints['decrypt_rss'])
        ]
        
        for source_name, feed_url in rss_feeds:
            try:
                async with self.session.get(feed_url) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        feed_items = self._parse_rss_feed(xml_content, source_name, coin_name)
                        news_items.extend(feed_items)
                        
                await asyncio.sleep(self.request_delays['rss_feeds'])
                
            except Exception as e:
                self.logger.debug(f"RSS feed error for {source_name}: {str(e)}")
        
        self.logger.debug(f"📡 RSS feeds: {len(news_items)} relevant articles")
        return news_items
    
    async def _fetch_cryptopanic_news(self, coin_name: str) -> List[NewsItem]:
        """Fetch news from CryptoPanic"""
        try:
            api_key = getattr(config, 'CRYPTOPANIC_API_KEY', None)
            if not api_key:
                self.logger.debug("CryptoPanic API key not configured")
                return []
            
            # Map coin names to CryptoPanic currencies
            currency_mapping = {
                'bitcoin': 'BTC',
                'ethereum': 'ETH',
                'binance': 'BNB',
                'cardano': 'ADA',
                'solana': 'SOL'
            }
            
            currency = currency_mapping.get(coin_name.lower(), coin_name.upper())
            
            params = {
                'auth_token': api_key,
                'public': 'true',
                'kind': 'news',
                'filter': 'hot',
                'currencies': currency
            }
            
            async with self.session.get(self.endpoints['cryptopanic'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = data.get('results', [])
                    
                    news_items = []
                    for post in posts:
                        news_item = self._parse_cryptopanic_post(post, coin_name)
                        if news_item:
                            news_items.append(news_item)
                    
                    self.logger.debug(f"🚨 CryptoPanic: {len(news_items)} articles")
                    await asyncio.sleep(self.request_delays['cryptopanic'])
                    return news_items
                    
        except Exception as e:
            self.logger.debug(f"CryptoPanic fetch error: {str(e)}")
        
        return []
    
    async def _fetch_additional_sources(self, coin_name: str) -> List[NewsItem]:
        """Fetch from additional free sources"""
        news_items = []
        
        try:
            # Alternative.me API for Fear & Greed Index news
            fear_greed_url = "https://api.alternative.me/fng/"
            async with self.session.get(fear_greed_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and data['data']:
                        fng_data = data['data'][0]
                        
                        # Create synthetic news item for Fear & Greed Index
                        news_item = NewsItem(
                            title=f"Crypto Fear & Greed Index: {fng_data.get('value_classification', 'Neutral')}",
                            content=f"Market sentiment indicator shows {fng_data.get('value_classification', 'neutral')} at {fng_data.get('value', 50)}/100",
                            source="Alternative.me",
                            published_at=datetime.now(),
                            sentiment_score=self._fear_greed_to_sentiment(int(fng_data.get('value', 50))),
                            relevance_score=0.8,  # High relevance for market sentiment
                            impact_level='MEDIUM'
                        )
                        news_items.append(news_item)
            
        except Exception as e:
            self.logger.debug(f"Additional sources error: {str(e)}")
        
        return news_items
    
    # =============================================================================
    # 📄 NEWS PARSING METHODS
    # =============================================================================
    
    def _parse_newsapi_article(self, article: dict, coin_name: str) -> Optional[NewsItem]:
        """Parse NewsAPI article to NewsItem"""
        try:
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description} {article.get('content', '')}"
            
            if not self._is_relevant_to_coin(content, coin_name):
                return None
            
            published_str = article.get('publishedAt', '')
            try:
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            except:
                published_at = datetime.now()
            
            return NewsItem(
                title=title,
                content=content,
                source='NewsAPI',
                published_at=published_at,
                sentiment_score=self._calculate_text_sentiment(content),
                relevance_score=self._calculate_relevance(content, coin_name),
                impact_level=self._determine_impact_level(content),
                url=article.get('url', ''),
                author=article.get('author', '')
            )
            
        except Exception as e:
            self.logger.debug(f"Error parsing NewsAPI article: {str(e)}")
            return None
    
    def _parse_cryptocompare_article(self, article: dict, coin_name: str) -> Optional[NewsItem]:
        """Parse CryptoCompare article to NewsItem"""
        try:
            title = article.get('title', '')
            body = article.get('body', '')
            content = f"{title} {body}"
            
            if not self._is_relevant_to_coin(content, coin_name):
                return None
            
            published_at = datetime.fromtimestamp(article.get('published_on', 0))
            
            return NewsItem(
                title=title,
                content=content,
                source='CryptoCompare',
                published_at=published_at,
                sentiment_score=self._calculate_text_sentiment(content),
                relevance_score=self._calculate_relevance(content, coin_name),
                impact_level=self._determine_impact_level(content),
                url=article.get('url', ''),
                author=article.get('source_info', {}).get('name', '')
            )
            
        except Exception as e:
            self.logger.debug(f"Error parsing CryptoCompare article: {str(e)}")
            return None
    
    def _parse_rss_feed(self, xml_content: str, source_name: str, coin_name: str) -> List[NewsItem]:
        """Parse RSS feed XML content"""
        news_items = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Handle different RSS formats
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            for item in items[:20]:  # Limit to 20 items per feed
                try:
                    # Extract title
                    title_elem = item.find('title') or item.find('{http://www.w3.org/2005/Atom}title')
                    title = title_elem.text if title_elem is not None else ''
                    
                    # Extract description/content
                    desc_elem = (item.find('description') or 
                               item.find('{http://www.w3.org/2005/Atom}summary') or
                               item.find('{http://www.w3.org/2005/Atom}content'))
                    description = desc_elem.text if desc_elem is not None else ''
                    
                    content = f"{title} {description}"
                    
                    if not self._is_relevant_to_coin(content, coin_name):
                        continue
                    
                    # Extract publish date
                    pub_date_elem = (item.find('pubDate') or 
                                   item.find('{http://www.w3.org/2005/Atom}published'))
                    
                    published_at = datetime.now()
                    if pub_date_elem is not None:
                        try:
                            # Try parsing different date formats
                            date_str = pub_date_elem.text
                            if 'T' in date_str:
                                published_at = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            else:
                                from email.utils import parsedate_to_datetime
                                published_at = parsedate_to_datetime(date_str)
                        except:
                            pass
                    
                    # Extract URL
                    link_elem = (item.find('link') or 
                               item.find('{http://www.w3.org/2005/Atom}link'))
                    url = ''
                    if link_elem is not None:
                        url = link_elem.text or link_elem.get('href', '')
                    
                    news_item = NewsItem(
                        title=title,
                        content=content,
                        source=source_name,
                        published_at=published_at,
                        sentiment_score=self._calculate_text_sentiment(content),
                        relevance_score=self._calculate_relevance(content, coin_name),
                        impact_level=self._determine_impact_level(content),
                        url=url
                    )
                    
                    news_items.append(news_item)
                    
                except Exception as e:
                    self.logger.debug(f"Error parsing RSS item: {str(e)}")
                    continue
            
        except Exception as e:
            self.logger.debug(f"Error parsing RSS feed: {str(e)}")
        
        return news_items
    
    def _parse_cryptopanic_post(self, post: dict, coin_name: str) -> Optional[NewsItem]:
        """Parse CryptoPanic post to NewsItem"""
        try:
            title = post.get('title', '')
            
            if not self._is_relevant_to_coin(title, coin_name):
                return None
            
            published_str = post.get('published_at', '')
            try:
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            except:
                published_at = datetime.now()
            
            return NewsItem(
                title=title,
                content=title,  # CryptoPanic usually only has titles
                source='CryptoPanic',
                published_at=published_at,
                sentiment_score=self._calculate_text_sentiment(title),
                relevance_score=self._calculate_relevance(title, coin_name),
                impact_level=self._determine_impact_level(title),
                url=post.get('url', '')
            )
            
        except Exception as e:
            self.logger.debug(f"Error parsing CryptoPanic post: {str(e)}")
            return None
    
    # =============================================================================
    # 🧠 SENTIMENT ANALYSIS METHODS
    # =============================================================================
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text (-1 to +1)"""
        try:
            text_lower = text.lower()
            
            # Count different sentiment categories
            extremely_positive = sum(1 for word in self.sentiment_keywords['extremely_positive'] 
                                   if word in text_lower)
            positive = sum(1 for word in self.sentiment_keywords['positive'] 
                          if word in text_lower)
            neutral = sum(1 for word in self.sentiment_keywords['neutral'] 
                         if word in text_lower)
            negative = sum(1 for word in self.sentiment_keywords['negative'] 
                          if word in text_lower)
            extremely_negative = sum(1 for word in self.sentiment_keywords['extremely_negative'] 
                                   if word in text_lower)
            
            # Weight the sentiment scores
            weighted_score = (
                extremely_positive * 2.0 +
                positive * 1.0 +
                neutral * 0.0 +
                negative * -1.0 +
                extremely_negative * -2.0
            )
            
            total_sentiment_words = (extremely_positive + positive + neutral + 
                                   negative + extremely_negative)
            
            if total_sentiment_words == 0:
                return 0.0
            
            # Normalize to -1 to +1 range
            normalized_score = weighted_score / (total_sentiment_words * 2.0)
            
            # Apply intensity multipliers for certain patterns
            if any(phrase in text_lower for phrase in ['massive surge', 'explosive growth', 'to the moon']):
                normalized_score *= 1.5
            elif any(phrase in text_lower for phrase in ['massive crash', 'devastating loss', 'market collapse']):
                normalized_score *= 1.5
            
            return max(-1.0, min(1.0, normalized_score))
            
        except Exception as e:
            self.logger.debug(f"Error calculating sentiment: {str(e)}")
            return 0.0
    
    def _calculate_relevance(self, text: str, coin_name: str) -> float:
        """Calculate relevance score for the specific coin (0 to 1)"""
        try:
            text_lower = text.lower()
            coin_lower = coin_name.lower()
            
            relevance_score = 0.0
            
            # Direct coin mention (highest relevance)
            if coin_lower in text_lower:
                relevance_score += 0.4
                
                # Bonus for multiple mentions
                mention_count = text_lower.count(coin_lower)
                relevance_score += min(0.2, mention_count * 0.05)
            
            # Symbol mentions for major coins
            symbol_mapping = {
                'bitcoin': ['btc', '$btc'],
                'ethereum': ['eth', '$eth'],
                'binance': ['bnb', '$bnb'],
                'cardano': ['ada', '$ada'],
                'solana': ['sol', '$sol']
            }
            
            if coin_lower in symbol_mapping:
                for symbol in symbol_mapping[coin_lower]:
                    if symbol in text_lower:
                        relevance_score += 0.3
                        break
            
            # Crypto-related terms
            crypto_terms_count = sum(1 for term in self.crypto_terms 
                                   if term in text_lower)
            relevance_score += min(0.3, crypto_terms_count * 0.05)
            
            # Market-wide relevance for major coins
            if coin_lower in ['bitcoin', 'ethereum'] and any(term in text_lower 
                for term in ['crypto market', 'cryptocurrency market', 'digital assets']):
                relevance_score += 0.2
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            self.logger.debug(f"Error calculating relevance: {str(e)}")
            return 0.0
    
    def _determine_impact_level(self, text: str) -> str:
        """Determine the potential market impact level"""
        try:
            text_lower = text.lower()
            
            # Check for high impact keywords
            high_impact_count = sum(1 for term in self.impact_keywords['high_impact'] 
                                  if term in text_lower)
            
            # Check for medium impact keywords
            medium_impact_count = sum(1 for term in self.impact_keywords['medium_impact'] 
                                    if term in text_lower)
            
            # Determine impact level
            if high_impact_count >= 2:
                return 'HIGH'
            elif high_impact_count >= 1 or medium_impact_count >= 3:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            self.logger.debug(f"Error determining impact level: {str(e)}")
            return 'LOW'
    
    def _is_relevant_to_coin(self, text: str, coin_name: str) -> bool:
        """Check if news is relevant to the specific coin"""
        try:
            relevance_score = self._calculate_relevance(text, coin_name)
            return relevance_score >= 0.2  # Minimum relevance threshold
            
        except Exception as e:
            self.logger.debug(f"Error checking relevance: {str(e)}")
            return False
    
    # =============================================================================
    # 📊 COMPREHENSIVE SENTIMENT ANALYSIS
    # =============================================================================
    
    async def _analyze_comprehensive_sentiment(self, news_items: List[NewsItem], coin_name: str) -> SentimentAnalysis:
        """Analyze overall sentiment from news items with advanced weighting"""
        try:
            if not news_items:
                return self._create_neutral_sentiment(f"No news for {coin_name}")
            
            # Separate by sentiment category
            positive_items = []
            negative_items = []
            neutral_items = []
            
            weighted_sentiments = []
            source_sentiments = {}
            impact_weights = {'HIGH': 3.0, 'MEDIUM': 2.0, 'LOW': 1.0}
            
            for item in news_items:
                # Time decay factor (newer news is more important)
                time_decay = self._calculate_time_decay(item.published_at)
                
                # Impact weight
                impact_weight = impact_weights.get(item.impact_level, 1.0)
                
                # Combined weight
                total_weight = item.relevance_score * time_decay * impact_weight
                
                # Weighted sentiment
                weighted_sentiment = item.sentiment_score * total_weight
                weighted_sentiments.append(weighted_sentiment)
                
                # Categorize sentiment
                if item.sentiment_score > 0.2:
                    positive_items.append(item)
                elif item.sentiment_score < -0.2:
                    negative_items.append(item)
                else:
                    neutral_items.append(item)
                
                # Source breakdown
                if item.source not in source_sentiments:
                    source_sentiments[item.source] = []
                source_sentiments[item.source].append(item.sentiment_score)
            
            # Calculate overall sentiment
            total_items = len(news_items)
            overall_sentiment = sum(weighted_sentiments) / len(weighted_sentiments) if weighted_sentiments else 0.0
            
            # Calculate ratios
            positive_ratio = len(positive_items) / total_items
            negative_ratio = len(negative_items) / total_items
            neutral_ratio = len(neutral_items) / total_items
            
            # Calculate confidence
            confidence = self._calculate_sentiment_confidence(news_items, overall_sentiment)
            
            # Extract key events
            key_events = self._extract_key_events(news_items)
            
            # Determine overall impact level
            impact_level = self._determine_overall_impact(news_items)
            
            # Source breakdown
            source_breakdown = {}
            for source, sentiments in source_sentiments.items():
                source_breakdown[source] = sum(sentiments) / len(sentiments)
            
            # Sentiment distribution
            sentiment_distribution = {
                'extremely_positive': len([i for i in positive_items if i.sentiment_score > 0.6]),
                'positive': len([i for i in positive_items if 0.2 < i.sentiment_score <= 0.6]),
                'neutral': len(neutral_items),
                'negative': len([i for i in negative_items if -0.6 <= i.sentiment_score < -0.2]),
                'extremely_negative': len([i for i in negative_items if i.sentiment_score < -0.6])
            }
            
            return SentimentAnalysis(
                overall_sentiment=overall_sentiment,
                news_count=total_items,
                positive_ratio=positive_ratio,
                negative_ratio=negative_ratio,
                neutral_ratio=neutral_ratio,
                confidence=confidence,
                key_events=key_events,
                impact_level=impact_level,
                sentiment_distribution=sentiment_distribution,
                source_breakdown=source_breakdown,
                time_decay_factor=sum([self._calculate_time_decay(item.published_at) for item in news_items]) / total_items
            )
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive sentiment analysis: {str(e)}")
            return self._create_neutral_sentiment(f"Error analyzing {coin_name}")
    
    def _calculate_time_decay(self, published_at: datetime) -> float:
        """Calculate time decay factor (newer = higher weight)"""
        try:
            hours_old = (datetime.now() - published_at).total_seconds() / 3600
            
            # Exponential decay over 24 hours
            if hours_old < 0:  # Future date (shouldn't happen)
                return 1.0
            elif hours_old <= 1:  # Very recent
                return 1.0
            elif hours_old <= 6:  # Recent
                return 0.9
            elif hours_old <= 12:  # Moderate
                return 0.7
            elif hours_old <= 24:  # Old
                return 0.5
            else:  # Very old
                return 0.3
                
        except Exception as e:
            self.logger.debug(f"Error calculating time decay: {str(e)}")
            return 0.5
    
    def _calculate_sentiment_confidence(self, news_items: List[NewsItem], overall_sentiment: float) -> float:
        """Calculate confidence in sentiment analysis"""
        try:
            # Base confidence on news volume
            volume_confidence = min(1.0, len(news_items) / 10)  # Max confidence at 10+ articles
            
            # Agreement factor (how much do articles agree)
            sentiments = [item.sentiment_score for item in news_items]
            if len(sentiments) > 1:
                sentiment_std = np.std(sentiments) if 'np' in globals() else self._calculate_std(sentiments)
                agreement_confidence = max(0.1, 1.0 - sentiment_std)
            else:
                agreement_confidence = 0.5
            
            # Source diversity (more sources = higher confidence)
            sources = set(item.source for item in news_items)
            source_confidence = min(1.0, len(sources) / 4)  # Max confidence at 4+ sources
            
            # Impact level consideration
            high_impact_count = sum(1 for item in news_items if item.impact_level == 'HIGH')
            impact_confidence = min(1.0, high_impact_count / 2 + 0.5)  # Base 0.5 + impact bonus
            
            # Combined confidence
            total_confidence = (
                volume_confidence * 0.3 +
                agreement_confidence * 0.3 +
                source_confidence * 0.2 +
                impact_confidence * 0.2
            )
            
            return min(0.95, max(0.1, total_confidence))
            
        except Exception as e:
            self.logger.debug(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _calculate_std(self, values: List[float]) -> float:
        """Simple standard deviation calculation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _extract_key_events(self, news_items: List[NewsItem]) -> List[str]:
        """Extract key events from news items"""
        try:
            # Sort by impact and sentiment strength
            key_items = sorted(news_items, 
                             key=lambda x: (
                                 {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x.impact_level] + 
                                 abs(x.sentiment_score)
                             ), 
                             reverse=True)
            
            key_events = []
            for item in key_items[:3]:  # Top 3 key events
                event_description = f"{item.impact_level}: {item.title[:80]}..."
                key_events.append(event_description)
            
            return key_events
            
        except Exception as e:
            self.logger.debug(f"Error extracting key events: {str(e)}")
            return []
    
    def _determine_overall_impact(self, news_items: List[NewsItem]) -> str:
        """Determine overall market impact level"""
        try:
            high_impact_count = sum(1 for item in news_items if item.impact_level == 'HIGH')
            medium_impact_count = sum(1 for item in news_items if item.impact_level == 'MEDIUM')
            
            if high_impact_count >= 3:
                return 'HIGH'
            elif high_impact_count >= 1 or medium_impact_count >= 5:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            self.logger.debug(f"Error determining overall impact: {str(e)}")
            return 'LOW'
    
    def _fear_greed_to_sentiment(self, fng_value: int) -> float:
        """Convert Fear & Greed Index to sentiment score"""
        # Fear & Greed Index: 0-24 = Extreme Fear, 25-49 = Fear, 50-74 = Greed, 75-100 = Extreme Greed
        if fng_value <= 24:
            return -0.8  # Extreme Fear
        elif fng_value <= 49:
            return -0.4  # Fear
        elif fng_value <= 74:
            return 0.4   # Greed
        else:
            return 0.8   # Extreme Greed
    
    # =============================================================================
    # 🛠️ UTILITY METHODS
    # =============================================================================
    
    def _deduplicate_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news items"""
        try:
            seen_hashes = set()
            unique_items = []
            
            for item in news_items:
                # Create hash based on title similarity
                title_normalized = re.sub(r'[^\w\s]', '', item.title.lower())
                title_hash = hashlib.md5(title_normalized.encode()).hexdigest()
                
                if title_hash not in seen_hashes:
                    seen_hashes.add(title_hash)
                    unique_items.append(item)
            
            return unique_items
            
        except Exception as e:
            self.logger.debug(f"Error deduplicating news: {str(e)}")
            return news_items
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        try:
            if key not in self.cache:
                return False
            
            cache_entry = self.cache[key]
            if not hasattr(cache_entry, '_cache_time'):
                return False
            
            cache_age = (datetime.now() - cache_entry._cache_time).total_seconds()
            cache_duration = config.NEWS_SENTIMENT_CONFIG.get('cache_duration_minutes', 30) * 60
            
            return cache_age < cache_duration
            
        except Exception as e:
            self.logger.debug(f"Error checking cache validity: {str(e)}")
            return False
    
    def _cache_result(self, key: str, result: SentimentAnalysis):
        """Cache sentiment analysis result"""
        try:
            result._cache_time = datetime.now()
            self.cache[key] = result
            
            # Clean old cache entries
            if len(self.cache) > 100:  # Limit cache size
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: getattr(self.cache[k], '_cache_time', datetime.now()))
                del self.cache[oldest_key]
                
        except Exception as e:
            self.logger.debug(f"Error caching result: {str(e)}")
    
    def _create_neutral_sentiment(self, reason: str = "No data") -> SentimentAnalysis:
        """Create neutral sentiment analysis when no news found"""
        return SentimentAnalysis(
            overall_sentiment=0.0,
            news_count=0,
            positive_ratio=0.0,
            negative_ratio=0.0,
            neutral_ratio=1.0,
            confidence=0.0,
            key_events=[reason],
            impact_level='LOW',
            sentiment_distribution={'neutral': 1},
            source_breakdown={},
            time_decay_factor=1.0
        )

# =============================================================================
# 🧪 TESTING FUNCTION
# =============================================================================

async def test_news_analyzer():
    """Test function for the news analyzer"""
    print("🧪 Testing NewsAnalyzer V3.0 Ultimate...")
    
    async with NewsAnalyzer() as analyzer:
        try:
            # Test Bitcoin sentiment
            btc_sentiment = await analyzer.get_crypto_news_sentiment('BTC/USDT')
            print(f"✅ BTC Sentiment: {btc_sentiment.overall_sentiment:+.2f}")
            print(f"   News Count: {btc_sentiment.news_count}")
            print(f"   Confidence: {btc_sentiment.confidence:.2f}")
            print(f"   Impact Level: {btc_sentiment.impact_level}")
            
            if btc_sentiment.key_events:
                print(f"   Key Events: {btc_sentiment.key_events[0]}")
            
            # Test Ethereum sentiment
            eth_sentiment = await analyzer.get_crypto_news_sentiment('ETH/USDT')
            print(f"✅ ETH Sentiment: {eth_sentiment.overall_sentiment:+.2f}")
            
            print("🎉 NewsAnalyzer test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_news_analyzer())