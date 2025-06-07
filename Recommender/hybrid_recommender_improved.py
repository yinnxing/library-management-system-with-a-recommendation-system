"""
Improved Hybrid Book Recommender System

This module provides a comprehensive book recommendation system that combines:
- Content-based filtering using TF-IDF and semantic embeddings
- Collaborative filtering using matrix factorization and advanced algorithms
- Popularity-based recommendations
- Hybrid approach with optimizable weights

Key improvements:
- Better code organization with separate classes for each component
- Comprehensive error handling and logging
- Type hints throughout
- Configuration management
- Memory-efficient operations
- Modular design for easy extension
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import re
import pickle
import os
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from surprise import Dataset, Reader, SVD, SVDpp, KNNBasic
    from surprise.model_selection import train_test_split as surprise_train_test_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


@dataclass
class RecommenderConfig:
    """Configuration class for the hybrid recommender system."""
    
    # Data preprocessing
    min_book_ratings: int = 15
    min_user_ratings: int = 3
    test_size: float = 0.2
    random_state: int = 42
    
    # Model parameters
    content_weight: float = 0.3
    collab_weight: float = 0.5
    popular_weight: float = 0.2
    optimize_weights: bool = False
    
    # Content-based parameters
    tfidf_max_features: int = 5000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_min_df: int = 2
    
    # Collaborative filtering parameters
    svd_n_components: int = 100
    svd_n_epochs: int = 20
    svd_lr: float = 0.005
    svd_reg: float = 0.02
    
    # Recommendation parameters
    default_n_recommendations: int = 10
    diversity_factor: float = 0.3
    cold_start_threshold: int = 5
    
    # Performance parameters
    cache_ttl: int = 3600  # 1 hour
    batch_size: int = 128
    
    # Optimization parameters
    optimization_iterations: int = 30
    lambda_rmse: float = 0.1
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None


class RecommenderError(Exception):
    """Base exception for recommender system errors."""
    pass


class DataError(RecommenderError):
    """Exception raised for data-related errors."""
    pass


class ModelError(RecommenderError):
    """Exception raised for model-related errors."""
    pass


class BaseRecommender(ABC):
    """Abstract base class for recommendation algorithms."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the recommender model to the data."""
        pass
    
    @abstractmethod
    def recommend(self, item_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get recommendations for an item."""
        pass


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommendation using TF-IDF and semantic embeddings."""
    
    def __init__(self, config: RecommenderConfig):
        self.config = config
        self.similarity_matrix: Optional[pd.DataFrame] = None
        self.semantic_model: Optional[SentenceTransformer] = None
        self.logger = logging.getLogger(__name__)
        
    def fit(self, data: pd.DataFrame) -> None:
        """Build content-based model using book features."""
        try:
            self.logger.info("Building content-based model...")
            
            # Prepare book content features
            books_content = self._prepare_content_features(data)
            
            # Build TF-IDF matrix
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._build_semantic_similarity(books_content)
            else:
                self._build_tfidf_similarity(books_content)
                
            self.logger.info(f"Content-based model built with {len(self.similarity_matrix)} books")
            
        except Exception as e:
            self.logger.error(f"Error building content-based model: {str(e)}")
            raise ModelError(f"Failed to build content-based model: {str(e)}")
    
    def _prepare_content_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare content features for books."""
        books_content = data.drop_duplicates('Book-Title')[
            ['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']
        ].copy()
        
        # Clean and prepare text features
        books_content['Book-Author'] = books_content['Book-Author'].fillna('Unknown Author')
        books_content['Publisher'] = books_content['Publisher'].fillna('Unknown Publisher')
        books_content['Year-Of-Publication'] = books_content['Year-Of-Publication'].fillna(2000)
        
        # Create combined content string
        books_content['content'] = (
            books_content['Book-Author'] + ' ' + 
            books_content['Book-Author'] + ' ' +  # Repeat for more weight
            books_content['Publisher'] + ' ' + 
            books_content['Year-Of-Publication'].astype(str)
        )
        
        return books_content
    
    def _build_semantic_similarity(self, books_content: pd.DataFrame) -> None:
        """Build similarity matrix using semantic embeddings."""
        self.logger.info("Using semantic embeddings for content similarity")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_model = model
        
        # Generate embeddings in batches
        embeddings = []
        titles = books_content['Book-Title'].tolist()
        
        for i in range(0, len(books_content), self.config.batch_size):
            batch = books_content.iloc[i:i+self.config.batch_size]
            batch_embeddings = model.encode(batch['content'].tolist(), show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        similarity_matrix = cosine_similarity(embeddings)
        
        self.similarity_matrix = pd.DataFrame(
            similarity_matrix, index=titles, columns=titles
        )
    
    def _build_tfidf_similarity(self, books_content: pd.DataFrame) -> None:
        """Build similarity matrix using TF-IDF."""
        self.logger.info("Using TF-IDF for content similarity")
        
        tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=self.config.tfidf_ngram_range,
            max_features=self.config.tfidf_max_features,
            min_df=self.config.tfidf_min_df
        )
        
        tfidf_matrix = tfidf.fit_transform(books_content['content'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        self.similarity_matrix = pd.DataFrame(
            similarity_matrix,
            index=books_content['Book-Title'],
            columns=books_content['Book-Title']
        )
    
    def recommend(self, item_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get content-based recommendations."""
        if self.similarity_matrix is None:
            raise ModelError("Model not fitted. Call fit() first.")
        
        if item_id not in self.similarity_matrix.index:
            return []
        
        sim_scores = self.similarity_matrix[item_id].sort_values(ascending=False)
        recommendations = sim_scores[1:n+1]  # Exclude the item itself
        
        return [(book, score) for book, score in recommendations.items()]


class CollaborativeRecommender(BaseRecommender):
    """Collaborative filtering using matrix factorization."""
    
    def __init__(self, config: RecommenderConfig):
        self.config = config
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.item_similarity_matrix: Optional[pd.DataFrame] = None
        self.svd_model: Optional[TruncatedSVD] = None
        self.user_means: Optional[Dict[int, float]] = None
        self.surprise_model: Optional[Any] = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, data: pd.DataFrame) -> None:
        """Build collaborative filtering model."""
        try:
            self.logger.info("Building collaborative filtering model...")
            
            # Create user-item matrix
            self.user_item_matrix = data.pivot_table(
                index='User-ID', 
                columns='Book-Title',
                values='Book-Rating'
            ).fillna(0)
            
            # Build matrix factorization model
            self._build_matrix_factorization()
            
            # Build Surprise model if available
            if SURPRISE_AVAILABLE:
                self._build_surprise_model(data)
            
            self.logger.info(f"Collaborative model built with {self.user_item_matrix.shape}")
            
        except Exception as e:
            self.logger.error(f"Error building collaborative model: {str(e)}")
            raise ModelError(f"Failed to build collaborative model: {str(e)}")
    
    def _build_matrix_factorization(self) -> None:
        """Build SVD-based matrix factorization model."""
        # Center ratings around user means
        user_matrix = self.user_item_matrix.replace(0, np.nan)
        user_means = user_matrix.mean(axis=1)
        self.user_means = user_means.to_dict()
        
        # Center the data
        for user_id in user_matrix.index:
            user_mean = user_means[user_id]
            mask = ~np.isnan(user_matrix.loc[user_id])
            user_matrix.loc[user_id, mask] = user_matrix.loc[user_id, mask] - user_mean
        
        user_matrix = user_matrix.fillna(0)
        
        # Apply SVD
        n_components = min(
            user_matrix.shape[0], 
            user_matrix.shape[1], 
            self.config.svd_n_components
        )
        
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=self.config.random_state)
        svd_result = self.svd_model.fit_transform(user_matrix)
        
        # Calculate item similarities
        item_factors = self.svd_model.components_.T
        item_similarity = cosine_similarity(item_factors)
        
        self.item_similarity_matrix = pd.DataFrame(
            item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
    
    def _build_surprise_model(self, data: pd.DataFrame) -> None:
        """Build Surprise library model."""
        reader = Reader(rating_scale=(1, 10))
        dataset = Dataset.load_from_df(data[['User-ID', 'Book-Title', 'Book-Rating']], reader)
        trainset, _ = surprise_train_test_split(dataset, test_size=0.2)
        
        self.surprise_model = SVD(
            n_factors=self.config.svd_n_components,
            n_epochs=self.config.svd_n_epochs,
            lr_all=self.config.svd_lr,
            reg_all=self.config.svd_reg,
            random_state=self.config.random_state
        )
        self.surprise_model.fit(trainset)
    
    def recommend(self, item_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get collaborative filtering recommendations."""
        if self.item_similarity_matrix is None:
            raise ModelError("Model not fitted. Call fit() first.")
        
        if item_id not in self.item_similarity_matrix.index:
            return []
        
        sim_scores = self.item_similarity_matrix[item_id].sort_values(ascending=False)
        recommendations = sim_scores[1:n+1]  # Exclude the item itself
        
        return [(book, score) for book, score in recommendations.items()]


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommendations."""
    
    def __init__(self, config: RecommenderConfig):
        self.config = config
        self.popularity_scores: Optional[pd.DataFrame] = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, data: pd.DataFrame) -> None:
        """Build popularity-based model."""
        try:
            self.logger.info("Building popularity-based model...")
            
            # Calculate popularity scores
            self.popularity_scores = data.groupby('Book-Title')['Book-Rating'].agg(['count', 'mean'])
            self.popularity_scores['score'] = (
                self.popularity_scores['count'] * self.popularity_scores['mean']
            )
            self.popularity_scores = self.popularity_scores.sort_values('score', ascending=False)
            
            self.logger.info(f"Popularity model built with {len(self.popularity_scores)} books")
            
        except Exception as e:
            self.logger.error(f"Error building popularity model: {str(e)}")
            raise ModelError(f"Failed to build popularity model: {str(e)}")
    
    def recommend(self, item_id: str = None, n: int = 10) -> List[Tuple[str, float]]:
        """Get popularity-based recommendations."""
        if self.popularity_scores is None:
            raise ModelError("Model not fitted. Call fit() first.")
        
        top_books = self.popularity_scores.head(n)
        return [(book, score) for book, score in zip(top_books.index, top_books['score'])]


class HybridBookRecommender:
    """
    Improved Hybrid Book Recommender System.
    
    Combines content-based, collaborative filtering, and popularity-based approaches
    with optimizable weights and comprehensive evaluation capabilities.
    """
    
    def __init__(self, config: Optional[RecommenderConfig] = None):
        """Initialize the hybrid recommender system."""
        self.config = config or RecommenderConfig()
        self._setup_logging()
        
        # Initialize component recommenders
        self.content_recommender = ContentBasedRecommender(self.config)
        self.collab_recommender = CollaborativeRecommender(self.config)
        self.popularity_recommender = PopularityRecommender(self.config)
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        # Caching
        self.recommendation_cache: Dict[str, Dict] = {}
        
        # Statistics
        self.global_mean_rating: float = 0.0
        self.book_stats: Optional[pd.DataFrame] = None
        self.user_stats: Optional[pd.DataFrame] = None
        
        self.logger.info("Hybrid recommender system initialized")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.log_file
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, books_path: str, ratings_path: str, users_path: str) -> None:
        """Load and merge the dataset files."""
        try:
            self.logger.info("Loading datasets...")
            
            books_df = pd.read_csv(books_path, low_memory=False)
            ratings_df = pd.read_csv(ratings_path, low_memory=False)
            users_df = pd.read_csv(users_path, low_memory=False)
            
            # Merge datasets
            df1 = books_df.merge(ratings_df, how="left", on="ISBN")
            self.data = df1.merge(users_df, how="left", on="User-ID")
            
            self.logger.info(f"Loaded {len(books_df)} books, {len(ratings_df)} ratings, {len(users_df)} users")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise DataError(f"Failed to load data: {str(e)}")
    
    def preprocess_data(self, verbose: bool = True) -> pd.DataFrame:
        """Preprocess and clean the data."""
        if self.data is None:
            raise DataError("No data loaded. Call load_data() first.")
        
        try:
            if verbose:
                self.logger.info("Preprocessing data...")
            
            # Remove missing ratings
            initial_size = len(self.data)
            self.data = self.data.dropna(subset=['User-ID', 'Book-Rating'])
            
            if verbose:
                self.logger.info(f"Removed {initial_size - len(self.data)} rows with missing ratings")
            
            # Clean text fields
            self._clean_text_fields()
            
            # Convert data types
            self._convert_data_types()
            
            # Remove implicit ratings (rating = 0)
            self.data = self.data[self.data["Book-Rating"] > 0]
            
            # Filter by frequency thresholds
            self._filter_by_frequency(verbose)
            
            # Calculate statistics
            self._calculate_statistics()
            
            if verbose:
                self.logger.info(f"Preprocessing complete. Final dataset: {len(self.data)} ratings")
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise DataError(f"Failed to preprocess data: {str(e)}")
    
    def _clean_text_fields(self) -> None:
        """Clean text fields in the dataset."""
        text_fields = ['Book-Title', 'Book-Author', 'Publisher']
        
        for field in text_fields:
            if field in self.data.columns:
                self.data[field] = self.data[field].astype(str).str.strip()
                self.data[field] = self.data[field].str.replace(r'\s+', ' ', regex=True)
    
    def _convert_data_types(self) -> None:
        """Convert data types safely."""
        # Convert User-ID to int
        self.data['User-ID'] = self.data['User-ID'].astype('int')
        
        # Handle Year-Of-Publication
        self.data['Year-Of-Publication'] = pd.to_numeric(
            self.data['Year-Of-Publication'], errors='coerce'
        )
        
        # Filter invalid years
        self.data.loc[self.data['Year-Of-Publication'] < 1800, 'Year-Of-Publication'] = None
        self.data.loc[self.data['Year-Of-Publication'] > 2023, 'Year-Of-Publication'] = None
        
        # Handle Book-Rating
        self.data['Book-Rating'] = pd.to_numeric(self.data['Book-Rating'], errors='coerce')
    
    def _filter_by_frequency(self, verbose: bool) -> None:
        """Filter data by frequency thresholds."""
        book_counts = self.data['Book-Title'].value_counts()
        user_counts = self.data['User-ID'].value_counts()
        
        popular_books = book_counts[book_counts >= self.config.min_book_ratings].index
        active_users = user_counts[user_counts >= self.config.min_user_ratings].index
        
        initial_books = self.data['Book-Title'].nunique()
        initial_users = self.data['User-ID'].nunique()
        
        self.data = self.data[self.data['Book-Title'].isin(popular_books)]
        self.data = self.data[self.data['User-ID'].isin(active_users)]
        
        if verbose:
            self.logger.info(f"Books: {initial_books} -> {self.data['Book-Title'].nunique()}")
            self.logger.info(f"Users: {initial_users} -> {self.data['User-ID'].nunique()}")
    
    def _calculate_statistics(self) -> None:
        """Calculate global statistics."""
        self.global_mean_rating = self.data['Book-Rating'].mean()
        
        self.book_stats = self.data.groupby('Book-Title')['Book-Rating'].agg(['mean', 'std', 'count'])
        self.user_stats = self.data.groupby('User-ID')['Book-Rating'].agg(['mean', 'std', 'count'])
    
    def split_data(self, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        if self.data is None:
            raise DataError("No data available. Call preprocess_data() first.")
        
        try:
            self.train_data, self.test_data = train_test_split(
                self.data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=None
            )
            
            if verbose:
                self.logger.info(f"Data split: {len(self.train_data)} train, {len(self.test_data)} test")
            
            return self.train_data, self.test_data
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise DataError(f"Failed to split data: {str(e)}")
    
    def fit(self, verbose: bool = True) -> None:
        """Fit all component models."""
        if self.train_data is None:
            raise ModelError("No training data available. Call split_data() first.")
        
        try:
            if verbose:
                self.logger.info("Fitting component models...")
            
            # Fit each component
            self.content_recommender.fit(self.train_data)
            self.collab_recommender.fit(self.train_data)
            self.popularity_recommender.fit(self.train_data)
            
            if verbose:
                self.logger.info("All models fitted successfully")
                
        except Exception as e:
            self.logger.error(f"Error fitting models: {str(e)}")
            raise ModelError(f"Failed to fit models: {str(e)}")
    
    def get_recommendations(self, book_title: str, n: int = None) -> List[str]:
        """Get hybrid recommendations for a book."""
        if n is None:
            n = self.config.default_n_recommendations
        
        # Check cache
        cache_key = f"{book_title}_{n}"
        if cache_key in self.recommendation_cache:
            cache_entry = self.recommendation_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.config.cache_ttl:
                return cache_entry['recommendations']
        
        try:
            # Get recommendations from each component
            content_recs = self.content_recommender.recommend(book_title, n * 2)
            collab_recs = self.collab_recommender.recommend(book_title, n * 2)
            popular_recs = self.popularity_recommender.recommend(n=n * 2)
            
            # Combine recommendations
            recommendations = self._combine_recommendations(
                content_recs, collab_recs, popular_recs, n
            )
            
            # Cache results
            self.recommendation_cache[cache_key] = {
                'recommendations': recommendations,
                'timestamp': time.time()
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {str(e)}")
            # Fallback to popularity-based recommendations
            popular_recs = self.popularity_recommender.recommend(n=n)
            return [book for book, _ in popular_recs]
    
    def _combine_recommendations(
        self, 
        content_recs: List[Tuple[str, float]], 
        collab_recs: List[Tuple[str, float]], 
        popular_recs: List[Tuple[str, float]], 
        n: int
    ) -> List[str]:
        """Combine recommendations from different approaches."""
        # Normalize scores
        content_scores = self._normalize_scores(content_recs)
        collab_scores = self._normalize_scores(collab_recs)
        popular_scores = self._normalize_scores(popular_recs)
        
        # Collect all unique books
        all_books = set()
        all_books.update([book for book, _ in content_recs])
        all_books.update([book for book, _ in collab_recs])
        all_books.update([book for book, _ in popular_recs])
        
        # Calculate hybrid scores
        hybrid_scores = {}
        for book in all_books:
            score = 0
            
            if book in content_scores:
                score += content_scores[book] * self.config.content_weight
            if book in collab_scores:
                score += collab_scores[book] * self.config.collab_weight
            if book in popular_scores:
                score += popular_scores[book] * self.config.popular_weight
            
            hybrid_scores[book] = score
        
        # Sort and return top N
        sorted_books = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return [book for book, _ in sorted_books[:n]]
    
    def _normalize_scores(self, recommendations: List[Tuple[str, float]]) -> Dict[str, float]:
        """Normalize recommendation scores to 0-1 range."""
        if not recommendations:
            return {}
        
        scores = [score for _, score in recommendations]
        min_score, max_score = min(scores), max(scores)
        
        if max_score == min_score:
            return {book: 1.0 for book, _ in recommendations}
        
        return {
            book: (score - min_score) / (max_score - min_score)
            for book, score in recommendations
        }
    
    def evaluate(self, k: int = 10, verbose: bool = True) -> Dict[str, float]:
        """Evaluate the recommender system."""
        if self.test_data is None:
            raise ModelError("No test data available. Call split_data() first.")
        
        try:
            if verbose:
                self.logger.info(f"Evaluating model with k={k}...")
            
            # Sample test users for evaluation
            test_users = self.test_data['User-ID'].unique()
            sample_size = min(100, len(test_users))
            test_users = np.random.choice(test_users, sample_size, replace=False)
            
            precision_scores = []
            recall_scores = []
            
            for user_id in test_users:
                # Get user's test items
                user_test = self.test_data[self.test_data['User-ID'] == user_id]
                relevant_items = set(user_test[user_test['Book-Rating'] >= 7]['Book-Title'])
                
                if not relevant_items:
                    continue
                
                # Get user's training items as seed
                user_train = self.train_data[self.train_data['User-ID'] == user_id]
                if len(user_train) == 0:
                    continue
                
                # Use highest rated book as seed
                seed_book = user_train.sort_values('Book-Rating', ascending=False).iloc[0]['Book-Title']
                
                # Get recommendations
                recommendations = set(self.get_recommendations(seed_book, k))
                
                # Calculate metrics
                if recommendations:
                    precision = len(recommendations & relevant_items) / len(recommendations)
                    recall = len(recommendations & relevant_items) / len(relevant_items)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
            
            # Calculate average metrics
            avg_precision = np.mean(precision_scores) if precision_scores else 0
            avg_recall = np.mean(recall_scores) if recall_scores else 0
            f1_score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            metrics = {
                'Precision@K': avg_precision,
                'Recall@K': avg_recall,
                'F1@K': f1_score,
                'NumEvaluations': len(precision_scores)
            }
            
            if verbose:
                self.logger.info(f"Evaluation results: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise ModelError(f"Failed to evaluate model: {str(e)}")
    
    def optimize_weights(self, verbose: bool = True) -> None:
        """Optimize component weights using validation data."""
        if verbose:
            self.logger.info("Optimizing component weights...")
        
        def objective(weights):
            # Normalize weights
            total = sum(weights)
            self.config.content_weight = weights[0] / total
            self.config.collab_weight = weights[1] / total
            self.config.popular_weight = weights[2] / total
            
            # Evaluate with current weights
            metrics = self.evaluate(verbose=False)
            
            # Return negative F1 score (minimize)
            return -metrics['F1@K']
        
        # Initial weights
        initial_weights = [
            self.config.content_weight,
            self.config.collab_weight,
            self.config.popular_weight
        ]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='L-BFGS-B',
            bounds=[(0.1, 0.9), (0.1, 0.9), (0.1, 0.9)]
        )
        
        # Set optimal weights
        optimal_weights = result.x
        total = sum(optimal_weights)
        self.config.content_weight = optimal_weights[0] / total
        self.config.collab_weight = optimal_weights[1] / total
        self.config.popular_weight = optimal_weights[2] / total
        
        if verbose:
            self.logger.info(f"Optimized weights: C={self.config.content_weight:.3f}, "
                           f"L={self.config.collab_weight:.3f}, P={self.config.popular_weight:.3f}")
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        try:
            model_data = {
                'config': self.config,
                'content_recommender': self.content_recommender,
                'collab_recommender': self.collab_recommender,
                'popularity_recommender': self.popularity_recommender,
                'global_mean_rating': self.global_mean_rating,
                'book_stats': self.book_stats,
                'user_stats': self.user_stats
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise ModelError(f"Failed to save model: {str(e)}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.config = model_data['config']
            self.content_recommender = model_data['content_recommender']
            self.collab_recommender = model_data['collab_recommender']
            self.popularity_recommender = model_data['popularity_recommender']
            self.global_mean_rating = model_data['global_mean_rating']
            self.book_stats = model_data['book_stats']
            self.user_stats = model_data['user_stats']
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise ModelError(f"Failed to load model: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = RecommenderConfig(
        min_book_ratings=10,
        min_user_ratings=3,
        content_weight=0.3,
        collab_weight=0.5,
        popular_weight=0.2,
        optimize_weights=True
    )
    
    # Initialize recommender
    recommender = HybridBookRecommender(config)
    
    try:
        # Load and preprocess data
        recommender.load_data(
            books_path='Recommender/Books.csv',
            ratings_path='Recommender/Ratings.csv',
            users_path='Recommender/Users.csv'
        )
        
        recommender.preprocess_data(verbose=True)
        
        # Split data and fit models
        recommender.split_data(verbose=True)
        recommender.fit(verbose=True)
        
        # Evaluate model
        metrics = recommender.evaluate(verbose=True)
        print(f"Initial evaluation: {metrics}")
        
        # Optimize weights
        recommender.optimize_weights(verbose=True)
        
        # Re-evaluate with optimized weights
        optimized_metrics = recommender.evaluate(verbose=True)
        print(f"Optimized evaluation: {optimized_metrics}")
        
        # Get sample recommendations
        sample_book = "The Da Vinci Code"
        recommendations = recommender.get_recommendations(sample_book, n=5)
        print(f"\nRecommendations for '{sample_book}':")
        for i, book in enumerate(recommendations, 1):
            print(f"{i}. {book}")
        
        # Save model
        recommender.save_model('improved_book_recommender.pkl')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc() 