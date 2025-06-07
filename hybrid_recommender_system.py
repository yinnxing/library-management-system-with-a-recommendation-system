import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import Dataset, Reader, SVD, NMF as SurpriseNMF, KNNBasic, KNNWithMeans, BaselineOnly
from surprise.model_selection import train_test_split as surprise_train_test_split, cross_validate
from surprise import accuracy
from collections import defaultdict
import warnings
import pickle
import time
from datetime import datetime
import json
import os

warnings.filterwarnings('ignore')

class HybridRecommenderSystem:
    """
    Comprehensive Hybrid Recommender System for Academic Research
    
    Implements multiple recommendation algorithms:
    - Content-Based Filtering (TF-IDF, SVD, NMF)
    - Collaborative Filtering (SVD, NMF, KNN variants)
    - Popularity-Based Filtering
    - Hybrid approaches with different combination strategies
    
    Provides extensive evaluation metrics for academic reporting:
    - Accuracy metrics (RMSE, MAE, Precision, Recall, F1)
    - Ranking quality metrics (NDCG@k, MAP@k, MRR)
    - Diversity and novelty metrics
    - Coverage metrics
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Data storage
        self.books_df = None
        self.ratings_df = None
        self.users_df = None
        self.processed_ratings = None
        
        # Models
        self.models = {}
        self.content_models = {}
        self.collaborative_models = {}
        
        # Evaluation results
        self.evaluation_results = {}
        self.ranking_results = {}
        
        # Hybrid weights
        self.hybrid_weights = {
            'content_based': 0.3,
            'collaborative': 0.5,
            'popularity': 0.2
        }
        
        print("Hybrid Recommender System initialized")
    
    def load_data(self, books_path, ratings_path, users_path=None):
        """Load and preprocess datasets"""
        print("Loading datasets...")
        
        # Load data
        self.books_df = pd.read_csv(books_path, sep=',', on_bad_lines='skip', encoding='latin-1')
        self.ratings_df = pd.read_csv(ratings_path, sep=',', on_bad_lines='skip')
        
        if users_path:
            self.users_df = pd.read_csv(users_path, sep=',', on_bad_lines='skip')
        
        print(f"Loaded {len(self.books_df)} books and {len(self.ratings_df)} ratings")
        
        # Preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Comprehensive data preprocessing"""
        print("Preprocessing data...")
        
        # Clean books data
        self.books_df = self.books_df.dropna(subset=['ISBN', 'Book-Title', 'Book-Author'])
        self.books_df['Book-Title'] = self.books_df['Book-Title'].str.strip()
        self.books_df['Book-Author'] = self.books_df['Book-Author'].str.strip()
        
        # Filter explicit ratings only (remove implicit feedback)
        initial_ratings = len(self.ratings_df)
        self.ratings_df = self.ratings_df[self.ratings_df['Book-Rating'] > 0]
        print(f"Removed {initial_ratings - len(self.ratings_df)} implicit ratings")
        
        # Apply filtering thresholds
        self._apply_filtering_thresholds()
        
        # Create content features
        self._create_content_features()
        
        # Create user-item matrix
        self._create_user_item_matrix()
        
        print(f"Final dataset: {len(self.books_df)} books, {len(self.ratings_df)} ratings")
        print(f"Sparsity: {1 - len(self.ratings_df) / (len(self.books_df) * self.ratings_df['User-ID'].nunique()):.4f}")
    
    def _apply_filtering_thresholds(self, min_book_ratings=10, min_user_ratings=5):
        """Apply filtering thresholds to reduce sparsity"""
        print(f"Applying filtering thresholds (min_book_ratings={min_book_ratings}, min_user_ratings={min_user_ratings})")
        
        # Iterative filtering
        prev_books, prev_users = 0, 0
        iteration = 0
        
        while True:
            iteration += 1
            current_books = self.ratings_df['ISBN'].nunique()
            current_users = self.ratings_df['User-ID'].nunique()
            
            # Filter books with insufficient ratings
            book_counts = self.ratings_df['ISBN'].value_counts()
            valid_books = book_counts[book_counts >= min_book_ratings].index
            self.ratings_df = self.ratings_df[self.ratings_df['ISBN'].isin(valid_books)]
            
            # Filter users with insufficient ratings
            user_counts = self.ratings_df['User-ID'].value_counts()
            valid_users = user_counts[user_counts >= min_user_ratings].index
            self.ratings_df = self.ratings_df[self.ratings_df['User-ID'].isin(valid_users)]
            
            new_books = self.ratings_df['ISBN'].nunique()
            new_users = self.ratings_df['User-ID'].nunique()
            
            print(f"Iteration {iteration}: {new_books} books, {new_users} users")
            
            # Check convergence
            if new_books == prev_books and new_users == prev_users:
                break
                
            prev_books, prev_users = new_books, new_users
            
            if iteration > 10:  # Safety break
                break
        
        # Update books dataframe
        self.books_df = self.books_df[self.books_df['ISBN'].isin(self.ratings_df['ISBN'])]
        
    def _create_content_features(self):
        """Create content-based features"""
        print("Creating content features...")
        
        # Combine text features
        self.books_df['content_text'] = (
            self.books_df['Book-Title'].fillna('') + ' ' +
            self.books_df['Book-Author'].fillna('') + ' ' +
            self.books_df['Publisher'].fillna('')
        )
        
        # Extract year from publication year
        self.books_df['Year-Of-Publication'] = pd.to_numeric(
            self.books_df['Year-Of-Publication'], errors='coerce'
        )
        self.books_df['Year-Of-Publication'] = self.books_df['Year-Of-Publication'].fillna(
            self.books_df['Year-Of-Publication'].median()
        )
        
        # Create decade feature
        self.books_df['decade'] = (self.books_df['Year-Of-Publication'] // 10) * 10
        
    def _create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        print("Creating user-item matrix...")
        
        # Create pivot table
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='User-ID', 
            columns='ISBN', 
            values='Book-Rating',
            fill_value=0
        )
        
        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(self.user_item_matrix.index)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.user_item_matrix.columns)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
    
    def build_content_based_models(self):
        """Build multiple content-based models"""
        print("Building content-based models...")
        
        # TF-IDF model
        self._build_tfidf_model()
        
        # SVD content model
        self._build_content_svd_model()
        
        # NMF content model
        self._build_content_nmf_model()
        
    def _build_tfidf_model(self):
        """Build TF-IDF content-based model"""
        print("Building TF-IDF model...")
        
        # Create TF-IDF vectors
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = tfidf.fit_transform(self.books_df['content_text'])
        
        # Compute similarity matrix
        content_similarity = cosine_similarity(tfidf_matrix)
        
        self.content_models['tfidf'] = {
            'vectorizer': tfidf,
            'matrix': tfidf_matrix,
            'similarity': content_similarity,
            'book_indices': {isbn: idx for idx, isbn in enumerate(self.books_df['ISBN'])}
        }
        
    def _build_content_svd_model(self):
        """Build SVD-based content model"""
        print("Building Content SVD model...")
        
        # Use TF-IDF matrix for SVD
        tfidf_matrix = self.content_models['tfidf']['matrix']
        
        # Apply SVD
        svd = TruncatedSVD(n_components=100, random_state=self.random_state)
        svd_matrix = svd.fit_transform(tfidf_matrix)
        
        # Compute similarity
        svd_similarity = cosine_similarity(svd_matrix)
        
        self.content_models['svd'] = {
            'model': svd,
            'matrix': svd_matrix,
            'similarity': svd_similarity
        }
        
    def _build_content_nmf_model(self):
        """Build NMF-based content model"""
        print("Building Content NMF model...")
        
        # Use TF-IDF matrix for NMF
        tfidf_matrix = self.content_models['tfidf']['matrix']
        
        # Apply NMF
        nmf = NMF(n_components=100, random_state=self.random_state, max_iter=200)
        nmf_matrix = nmf.fit_transform(tfidf_matrix)
        
        # Compute similarity
        nmf_similarity = cosine_similarity(nmf_matrix)
        
        self.content_models['nmf'] = {
            'model': nmf,
            'matrix': nmf_matrix,
            'similarity': nmf_similarity
        }
    
    def build_collaborative_models(self):
        """Build multiple collaborative filtering models"""
        print("Building collaborative filtering models...")
        
        # Prepare Surprise dataset
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(
            self.ratings_df[['User-ID', 'ISBN', 'Book-Rating']], 
            reader
        )
        
        # Split data
        self.trainset, self.testset = surprise_train_test_split(
            data, test_size=0.2, random_state=self.random_state
        )
        
        # Build different collaborative models
        self._build_svd_model()
        self._build_nmf_model()
        self._build_knn_models()
        self._build_baseline_model()
        
    def _build_svd_model(self):
        """Build SVD collaborative filtering model"""
        print("Building SVD model...")
        
        svd = SVD(
            n_factors=150,
            n_epochs=50,
            lr_all=0.008,
            reg_all=0.01,
            random_state=self.random_state
        )
        
        svd.fit(self.trainset)
        self.collaborative_models['svd'] = svd
        
    def _build_nmf_model(self):
        """Build NMF collaborative filtering model"""
        print("Building NMF model...")
        
        nmf = SurpriseNMF(
            n_factors=100,
            n_epochs=50,
            random_state=self.random_state
        )
        
        nmf.fit(self.trainset)
        self.collaborative_models['nmf'] = nmf
        
    def _build_knn_models(self):
        """Build KNN-based models"""
        print("Building KNN models...")
        
        # User-based KNN
        knn_user = KNNWithMeans(
            k=50,
            sim_options={'name': 'pearson_baseline', 'user_based': True},
            random_state=self.random_state
        )
        knn_user.fit(self.trainset)
        self.collaborative_models['knn_user'] = knn_user
        
        # Item-based KNN
        knn_item = KNNWithMeans(
            k=50,
            sim_options={'name': 'pearson_baseline', 'user_based': False},
            random_state=self.random_state
        )
        knn_item.fit(self.trainset)
        self.collaborative_models['knn_item'] = knn_item
        
    def _build_baseline_model(self):
        """Build baseline model"""
        print("Building baseline model...")
        
        baseline = BaselineOnly(random_state=self.random_state)
        baseline.fit(self.trainset)
        self.collaborative_models['baseline'] = baseline
    
    def build_popularity_model(self):
        """Build popularity-based model"""
        print("Building popularity model...")
        
        # Calculate book popularity metrics
        book_stats = self.ratings_df.groupby('ISBN').agg({
            'Book-Rating': ['count', 'mean', 'std'],
            'User-ID': 'nunique'
        }).round(3)
        
        book_stats.columns = ['rating_count', 'rating_mean', 'rating_std', 'user_count']
        book_stats = book_stats.reset_index()
        
        # Calculate popularity score (weighted rating)
        C = book_stats['rating_mean'].mean()  # Average rating across all books
        m = book_stats['rating_count'].quantile(0.7)  # Minimum votes required
        
        book_stats['popularity_score'] = (
            (book_stats['rating_count'] / (book_stats['rating_count'] + m)) * book_stats['rating_mean'] +
            (m / (book_stats['rating_count'] + m)) * C
        )
        
        self.models['popularity'] = book_stats.sort_values('popularity_score', ascending=False)
        
    def get_content_recommendations(self, isbn, model_type='tfidf', n=20):
        """Get content-based recommendations"""
        if model_type not in self.content_models:
            raise ValueError(f"Model type {model_type} not available")
            
        model = self.content_models[model_type]
        book_indices = model.get('book_indices', self.content_models['tfidf']['book_indices'])
        
        if isbn not in book_indices:
            return []
            
        idx = book_indices[isbn]
        similarity_scores = list(enumerate(model['similarity'][idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar books (excluding the book itself)
        book_indices_list = [i[0] for i in similarity_scores[1:n+1]]
        
        recommendations = []
        for book_idx in book_indices_list:
            book_isbn = list(book_indices.keys())[list(book_indices.values()).index(book_idx)]
            book_info = self.books_df[self.books_df['ISBN'] == book_isbn].iloc[0]
            recommendations.append({
                'isbn': book_isbn,
                'title': book_info['Book-Title'],
                'author': book_info['Book-Author'],
                'similarity_score': similarity_scores[book_idx][1]
            })
            
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, model_type='svd', n=20):
        """Get collaborative filtering recommendations"""
        if model_type not in self.collaborative_models:
            raise ValueError(f"Model type {model_type} not available")
            
        model = self.collaborative_models[model_type]
        
        # Get all books the user hasn't rated
        user_ratings = self.ratings_df[self.ratings_df['User-ID'] == user_id]['ISBN'].tolist()
        all_books = self.books_df['ISBN'].tolist()
        unrated_books = [book for book in all_books if book not in user_ratings]
        
        # Predict ratings for unrated books
        predictions = []
        for isbn in unrated_books:
            pred = model.predict(user_id, isbn)
            predictions.append((isbn, pred.est))
            
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        recommendations = []
        for isbn, predicted_rating in predictions[:n]:
            book_info = self.books_df[self.books_df['ISBN'] == isbn].iloc[0]
            recommendations.append({
                'isbn': isbn,
                'title': book_info['Book-Title'],
                'author': book_info['Book-Author'],
                'predicted_rating': predicted_rating
            })
            
        return recommendations
    
    def get_popularity_recommendations(self, n=20):
        """Get popularity-based recommendations"""
        if 'popularity' not in self.models:
            self.build_popularity_model()
            
        top_books = self.models['popularity'].head(n)
        
        recommendations = []
        for _, book in top_books.iterrows():
            book_info = self.books_df[self.books_df['ISBN'] == book['ISBN']].iloc[0]
            recommendations.append({
                'isbn': book['ISBN'],
                'title': book_info['Book-Title'],
                'author': book_info['Book-Author'],
                'popularity_score': book['popularity_score'],
                'rating_count': book['rating_count'],
                'rating_mean': book['rating_mean']
            })
            
        return recommendations
    
    def get_hybrid_recommendations(self, user_id=None, isbn=None, n=20, strategy='weighted'):
        """
        Get hybrid recommendations using different combination strategies
        
        Args:
            user_id: User ID for collaborative filtering
            isbn: Book ISBN for content-based filtering
            n: Number of recommendations
            strategy: 'weighted', 'rank_fusion', 'cascade'
        """
        if strategy == 'weighted':
            return self._get_weighted_hybrid_recommendations(user_id, isbn, n)
        elif strategy == 'rank_fusion':
            return self._get_rank_fusion_recommendations(user_id, isbn, n)
        elif strategy == 'cascade':
            return self._get_cascade_recommendations(user_id, isbn, n)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _get_weighted_hybrid_recommendations(self, user_id, isbn, n):
        """Weighted hybrid approach"""
        recommendations = {}
        
        # Get content-based recommendations if ISBN provided
        if isbn and 'tfidf' in self.content_models:
            content_recs = self.get_content_recommendations(isbn, 'tfidf', n*2)
            for rec in content_recs:
                rec_isbn = rec['isbn']
                score = rec['similarity_score'] * self.hybrid_weights['content_based']
                recommendations[rec_isbn] = recommendations.get(rec_isbn, 0) + score
        
        # Get collaborative recommendations if user_id provided
        if user_id and 'svd' in self.collaborative_models:
            collab_recs = self.get_collaborative_recommendations(user_id, 'svd', n*2)
            for rec in collab_recs:
                rec_isbn = rec['isbn']
                # Normalize predicted rating to 0-1 scale
                normalized_score = (rec['predicted_rating'] - 1) / 9
                score = normalized_score * self.hybrid_weights['collaborative']
                recommendations[rec_isbn] = recommendations.get(rec_isbn, 0) + score
        
        # Get popularity recommendations
        if 'popularity' in self.models:
            pop_recs = self.get_popularity_recommendations(n*2)
            max_pop_score = max([rec['popularity_score'] for rec in pop_recs]) if pop_recs else 1
            for rec in pop_recs:
                rec_isbn = rec['isbn']
                normalized_score = rec['popularity_score'] / max_pop_score
                score = normalized_score * self.hybrid_weights['popularity']
                recommendations[rec_isbn] = recommendations.get(rec_isbn, 0) + score
        
        # Sort by combined score
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Format final recommendations
        final_recs = []
        for isbn, score in sorted_recs[:n]:
            book_info = self.books_df[self.books_df['ISBN'] == isbn].iloc[0]
            final_recs.append({
                'isbn': isbn,
                'title': book_info['Book-Title'],
                'author': book_info['Book-Author'],
                'hybrid_score': score
            })
            
        return final_recs
    
    def _get_rank_fusion_recommendations(self, user_id, isbn, n):
        """Rank fusion hybrid approach"""
        all_recommendations = {}
        
        # Get rankings from different methods
        if isbn and 'tfidf' in self.content_models:
            content_recs = self.get_content_recommendations(isbn, 'tfidf', n*2)
            for rank, rec in enumerate(content_recs):
                rec_isbn = rec['isbn']
                # Reciprocal rank fusion
                score = 1 / (rank + 1)
                all_recommendations[rec_isbn] = all_recommendations.get(rec_isbn, 0) + score
        
        if user_id and 'svd' in self.collaborative_models:
            collab_recs = self.get_collaborative_recommendations(user_id, 'svd', n*2)
            for rank, rec in enumerate(collab_recs):
                rec_isbn = rec['isbn']
                score = 1 / (rank + 1)
                all_recommendations[rec_isbn] = all_recommendations.get(rec_isbn, 0) + score
        
        # Sort by fusion score
        sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Format final recommendations
        final_recs = []
        for isbn, score in sorted_recs[:n]:
            book_info = self.books_df[self.books_df['ISBN'] == isbn].iloc[0]
            final_recs.append({
                'isbn': isbn,
                'title': book_info['Book-Title'],
                'author': book_info['Book-Author'],
                'fusion_score': score
            })
            
        return final_recs
    
    def _get_cascade_recommendations(self, user_id, isbn, n):
        """Cascade hybrid approach"""
        recommendations = []
        
        # Start with collaborative filtering
        if user_id and 'svd' in self.collaborative_models:
            collab_recs = self.get_collaborative_recommendations(user_id, 'svd', n)
            recommendations.extend(collab_recs[:n//2])
        
        # Fill remaining with content-based
        if isbn and 'tfidf' in self.content_models and len(recommendations) < n:
            content_recs = self.get_content_recommendations(isbn, 'tfidf', n)
            # Filter out already recommended books
            recommended_isbns = [rec['isbn'] for rec in recommendations]
            for rec in content_recs:
                if rec['isbn'] not in recommended_isbns and len(recommendations) < n:
                    recommendations.append(rec)
        
        # Fill remaining with popularity
        if len(recommendations) < n:
            pop_recs = self.get_popularity_recommendations(n)
            recommended_isbns = [rec['isbn'] for rec in recommendations]
            for rec in pop_recs:
                if rec['isbn'] not in recommended_isbns and len(recommendations) < n:
                    recommendations.append(rec)
        
        return recommendations[:n] 