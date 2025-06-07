"""
Test Suite for Improved Hybrid Book Recommender System

This module contains comprehensive tests for all components of the recommender system.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the improved recommender system
from hybrid_recommender_improved import (
    HybridBookRecommender,
    ContentBasedRecommender,
    CollaborativeRecommender,
    PopularityRecommender,
    RecommenderConfig,
    DataError,
    ModelError
)


class TestRecommenderConfig(unittest.TestCase):
    """Test the configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RecommenderConfig()
        
        self.assertEqual(config.min_book_ratings, 15)
        self.assertEqual(config.min_user_ratings, 3)
        self.assertEqual(config.content_weight, 0.3)
        self.assertEqual(config.collab_weight, 0.5)
        self.assertEqual(config.popular_weight, 0.2)
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.random_state, 42)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RecommenderConfig(
            min_book_ratings=10,
            content_weight=0.4,
            collab_weight=0.4,
            popular_weight=0.2
        )
        
        self.assertEqual(config.min_book_ratings, 10)
        self.assertEqual(config.content_weight, 0.4)
        self.assertEqual(config.collab_weight, 0.4)
        self.assertEqual(config.popular_weight, 0.2)


class TestDataGeneration(unittest.TestCase):
    """Helper class to generate test data."""
    
    @staticmethod
    def create_sample_data():
        """Create sample data for testing."""
        # Create sample books
        books_data = {
            'ISBN': ['123', '456', '789', '101', '102'],
            'Book-Title': ['Book A', 'Book B', 'Book C', 'Book D', 'Book E'],
            'Book-Author': ['Author 1', 'Author 2', 'Author 1', 'Author 3', 'Author 2'],
            'Publisher': ['Pub 1', 'Pub 2', 'Pub 1', 'Pub 3', 'Pub 2'],
            'Year-Of-Publication': [2000, 2005, 2010, 2015, 2020]
        }
        books_df = pd.DataFrame(books_data)
        
        # Create sample ratings
        ratings_data = {
            'User-ID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5] * 3,
            'ISBN': ['123', '456', '789', '123', '456', '101', '456', '789', '101', '789', '101', '102', '123', '456', '102'] * 3,
            'Book-Rating': [8, 7, 9, 6, 8, 7, 9, 8, 6, 7, 9, 8, 8, 7, 9] * 3
        }
        ratings_df = pd.DataFrame(ratings_data)
        
        # Create sample users
        users_data = {
            'User-ID': [1, 2, 3, 4, 5],
            'Location': ['City A', 'City B', 'City C', 'City D', 'City E'],
            'Age': [25, 30, 35, 40, 45]
        }
        users_df = pd.DataFrame(users_data)
        
        return books_df, ratings_df, users_df
    
    @staticmethod
    def save_sample_data_to_files():
        """Save sample data to temporary files."""
        books_df, ratings_df, users_df = TestDataGeneration.create_sample_data()
        
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        books_path = os.path.join(temp_dir, 'books.csv')
        ratings_path = os.path.join(temp_dir, 'ratings.csv')
        users_path = os.path.join(temp_dir, 'users.csv')
        
        books_df.to_csv(books_path, index=False)
        ratings_df.to_csv(ratings_path, index=False)
        users_df.to_csv(users_path, index=False)
        
        return books_path, ratings_path, users_path, temp_dir


class TestContentBasedRecommender(unittest.TestCase):
    """Test the content-based recommender."""
    
    def setUp(self):
        """Set up test data."""
        self.config = RecommenderConfig(min_book_ratings=1, min_user_ratings=1)
        self.recommender = ContentBasedRecommender(self.config)
        
        # Create test data
        books_df, ratings_df, users_df = TestDataGeneration.create_sample_data()
        df1 = books_df.merge(ratings_df, on="ISBN")
        self.test_data = df1.merge(users_df, on="User-ID")
    
    def test_fit(self):
        """Test fitting the content-based model."""
        self.recommender.fit(self.test_data)
        
        self.assertIsNotNone(self.recommender.similarity_matrix)
        self.assertGreater(len(self.recommender.similarity_matrix), 0)
    
    def test_recommend(self):
        """Test getting recommendations."""
        self.recommender.fit(self.test_data)
        
        recommendations = self.recommender.recommend('Book A', n=3)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
        
        # Check that recommendations are tuples of (book, score)
        for rec in recommendations:
            self.assertIsInstance(rec, tuple)
            self.assertEqual(len(rec), 2)
            self.assertIsInstance(rec[0], str)
            self.assertIsInstance(rec[1], (int, float))
    
    def test_recommend_nonexistent_book(self):
        """Test recommending for a book that doesn't exist."""
        self.recommender.fit(self.test_data)
        
        recommendations = self.recommender.recommend('Nonexistent Book', n=3)
        
        self.assertEqual(len(recommendations), 0)
    
    def test_recommend_without_fit(self):
        """Test that recommending without fitting raises an error."""
        with self.assertRaises(ModelError):
            self.recommender.recommend('Book A', n=3)


class TestCollaborativeRecommender(unittest.TestCase):
    """Test the collaborative filtering recommender."""
    
    def setUp(self):
        """Set up test data."""
        self.config = RecommenderConfig(min_book_ratings=1, min_user_ratings=1)
        self.recommender = CollaborativeRecommender(self.config)
        
        # Create test data
        books_df, ratings_df, users_df = TestDataGeneration.create_sample_data()
        df1 = books_df.merge(ratings_df, on="ISBN")
        self.test_data = df1.merge(users_df, on="User-ID")
    
    def test_fit(self):
        """Test fitting the collaborative filtering model."""
        self.recommender.fit(self.test_data)
        
        self.assertIsNotNone(self.recommender.user_item_matrix)
        self.assertIsNotNone(self.recommender.item_similarity_matrix)
        self.assertIsNotNone(self.recommender.svd_model)
        self.assertIsNotNone(self.recommender.user_means)
    
    def test_recommend(self):
        """Test getting recommendations."""
        self.recommender.fit(self.test_data)
        
        recommendations = self.recommender.recommend('Book A', n=3)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
        
        # Check that recommendations are tuples of (book, score)
        for rec in recommendations:
            self.assertIsInstance(rec, tuple)
            self.assertEqual(len(rec), 2)
            self.assertIsInstance(rec[0], str)
            self.assertIsInstance(rec[1], (int, float))


class TestPopularityRecommender(unittest.TestCase):
    """Test the popularity-based recommender."""
    
    def setUp(self):
        """Set up test data."""
        self.config = RecommenderConfig(min_book_ratings=1, min_user_ratings=1)
        self.recommender = PopularityRecommender(self.config)
        
        # Create test data
        books_df, ratings_df, users_df = TestDataGeneration.create_sample_data()
        df1 = books_df.merge(ratings_df, on="ISBN")
        self.test_data = df1.merge(users_df, on="User-ID")
    
    def test_fit(self):
        """Test fitting the popularity model."""
        self.recommender.fit(self.test_data)
        
        self.assertIsNotNone(self.recommender.popularity_scores)
        self.assertGreater(len(self.recommender.popularity_scores), 0)
        self.assertIn('count', self.recommender.popularity_scores.columns)
        self.assertIn('mean', self.recommender.popularity_scores.columns)
        self.assertIn('score', self.recommender.popularity_scores.columns)
    
    def test_recommend(self):
        """Test getting recommendations."""
        self.recommender.fit(self.test_data)
        
        recommendations = self.recommender.recommend(n=3)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
        
        # Check that recommendations are tuples of (book, score)
        for rec in recommendations:
            self.assertIsInstance(rec, tuple)
            self.assertEqual(len(rec), 2)
            self.assertIsInstance(rec[0], str)
            self.assertIsInstance(rec[1], (int, float))


class TestHybridBookRecommender(unittest.TestCase):
    """Test the main hybrid recommender system."""
    
    def setUp(self):
        """Set up test data and recommender."""
        self.config = RecommenderConfig(
            min_book_ratings=1,
            min_user_ratings=1,
            test_size=0.3,
            log_level="ERROR"  # Reduce logging during tests
        )
        self.recommender = HybridBookRecommender(self.config)
        
        # Create temporary data files
        self.books_path, self.ratings_path, self.users_path, self.temp_dir = \
            TestDataGeneration.save_sample_data_to_files()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test recommender initialization."""
        self.assertIsNotNone(self.recommender.config)
        self.assertIsNotNone(self.recommender.content_recommender)
        self.assertIsNotNone(self.recommender.collab_recommender)
        self.assertIsNotNone(self.recommender.popularity_recommender)
        self.assertIsInstance(self.recommender.recommendation_cache, dict)
    
    def test_load_data(self):
        """Test loading data from files."""
        self.recommender.load_data(self.books_path, self.ratings_path, self.users_path)
        
        self.assertIsNotNone(self.recommender.data)
        self.assertGreater(len(self.recommender.data), 0)
        
        # Check that required columns exist
        required_columns = ['Book-Title', 'Book-Author', 'User-ID', 'Book-Rating']
        for col in required_columns:
            self.assertIn(col, self.recommender.data.columns)
    
    def test_load_data_nonexistent_files(self):
        """Test loading data with nonexistent files."""
        with self.assertRaises(DataError):
            self.recommender.load_data('nonexistent1.csv', 'nonexistent2.csv', 'nonexistent3.csv')
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        self.recommender.load_data(self.books_path, self.ratings_path, self.users_path)
        processed_data = self.recommender.preprocess_data(verbose=False)
        
        self.assertIsNotNone(processed_data)
        self.assertGreater(len(processed_data), 0)
        self.assertEqual(self.recommender.global_mean_rating, processed_data['Book-Rating'].mean())
        self.assertIsNotNone(self.recommender.book_stats)
        self.assertIsNotNone(self.recommender.user_stats)
    
    def test_preprocess_data_without_loading(self):
        """Test preprocessing without loading data first."""
        with self.assertRaises(DataError):
            self.recommender.preprocess_data()
    
    def test_split_data(self):
        """Test data splitting."""
        self.recommender.load_data(self.books_path, self.ratings_path, self.users_path)
        self.recommender.preprocess_data(verbose=False)
        
        train_data, test_data = self.recommender.split_data(verbose=False)
        
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(test_data)
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(test_data), 0)
        self.assertEqual(len(train_data) + len(test_data), len(self.recommender.data))
    
    def test_fit_models(self):
        """Test fitting all component models."""
        self.recommender.load_data(self.books_path, self.ratings_path, self.users_path)
        self.recommender.preprocess_data(verbose=False)
        self.recommender.split_data(verbose=False)
        
        self.recommender.fit(verbose=False)
        
        # Check that all models are fitted
        self.assertIsNotNone(self.recommender.content_recommender.similarity_matrix)
        self.assertIsNotNone(self.recommender.collab_recommender.item_similarity_matrix)
        self.assertIsNotNone(self.recommender.popularity_recommender.popularity_scores)
    
    def test_get_recommendations(self):
        """Test getting hybrid recommendations."""
        self.recommender.load_data(self.books_path, self.ratings_path, self.users_path)
        self.recommender.preprocess_data(verbose=False)
        self.recommender.split_data(verbose=False)
        self.recommender.fit(verbose=False)
        
        recommendations = self.recommender.get_recommendations('Book A', n=3)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
        
        # Check that all recommendations are strings (book titles)
        for rec in recommendations:
            self.assertIsInstance(rec, str)
    
    def test_get_recommendations_with_caching(self):
        """Test that caching works for recommendations."""
        self.recommender.load_data(self.books_path, self.ratings_path, self.users_path)
        self.recommender.preprocess_data(verbose=False)
        self.recommender.split_data(verbose=False)
        self.recommender.fit(verbose=False)
        
        # Get recommendations twice
        recommendations1 = self.recommender.get_recommendations('Book A', n=3)
        recommendations2 = self.recommender.get_recommendations('Book A', n=3)
        
        # Should be the same due to caching
        self.assertEqual(recommendations1, recommendations2)
        
        # Check that cache was used
        cache_key = 'Book A_3'
        self.assertIn(cache_key, self.recommender.recommendation_cache)
    
    def test_evaluate(self):
        """Test model evaluation."""
        self.recommender.load_data(self.books_path, self.ratings_path, self.users_path)
        self.recommender.preprocess_data(verbose=False)
        self.recommender.split_data(verbose=False)
        self.recommender.fit(verbose=False)
        
        metrics = self.recommender.evaluate(k=3, verbose=False)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('Precision@K', metrics)
        self.assertIn('Recall@K', metrics)
        self.assertIn('F1@K', metrics)
        self.assertIn('NumEvaluations', metrics)
        
        # Check that metrics are valid
        self.assertGreaterEqual(metrics['Precision@K'], 0)
        self.assertLessEqual(metrics['Precision@K'], 1)
        self.assertGreaterEqual(metrics['Recall@K'], 0)
        self.assertLessEqual(metrics['Recall@K'], 1)
        self.assertGreaterEqual(metrics['F1@K'], 0)
        self.assertLessEqual(metrics['F1@K'], 1)
    
    def test_optimize_weights(self):
        """Test weight optimization."""
        self.recommender.load_data(self.books_path, self.ratings_path, self.users_path)
        self.recommender.preprocess_data(verbose=False)
        self.recommender.split_data(verbose=False)
        self.recommender.fit(verbose=False)
        
        # Store original weights
        original_weights = (
            self.recommender.config.content_weight,
            self.recommender.config.collab_weight,
            self.recommender.config.popular_weight
        )
        
        self.recommender.optimize_weights(verbose=False)
        
        # Check that weights are still valid (sum to 1, all positive)
        total_weight = (
            self.recommender.config.content_weight +
            self.recommender.config.collab_weight +
            self.recommender.config.popular_weight
        )
        self.assertAlmostEqual(total_weight, 1.0, places=3)
        self.assertGreater(self.recommender.config.content_weight, 0)
        self.assertGreater(self.recommender.config.collab_weight, 0)
        self.assertGreater(self.recommender.config.popular_weight, 0)
    
    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        self.recommender.load_data(self.books_path, self.ratings_path, self.users_path)
        self.recommender.preprocess_data(verbose=False)
        self.recommender.split_data(verbose=False)
        self.recommender.fit(verbose=False)
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'test_model.pkl')
        self.recommender.save_model(model_path)
        
        self.assertTrue(os.path.exists(model_path))
        
        # Create new recommender and load model
        new_recommender = HybridBookRecommender()
        new_recommender.load_model(model_path)
        
        # Check that key attributes are loaded
        self.assertIsNotNone(new_recommender.config)
        self.assertIsNotNone(new_recommender.content_recommender)
        self.assertIsNotNone(new_recommender.collab_recommender)
        self.assertIsNotNone(new_recommender.popularity_recommender)
        
        # Test that loaded model can make recommendations
        recommendations = new_recommender.get_recommendations('Book A', n=3)
        self.assertIsInstance(recommendations, list)


class TestErrorHandling(unittest.TestCase):
    """Test error handling throughout the system."""
    
    def test_data_error_inheritance(self):
        """Test that DataError inherits from RecommenderError."""
        error = DataError("Test error")
        self.assertIsInstance(error, DataError)
        self.assertIsInstance(error, Exception)
    
    def test_model_error_inheritance(self):
        """Test that ModelError inherits from RecommenderError."""
        error = ModelError("Test error")
        self.assertIsInstance(error, ModelError)
        self.assertIsInstance(error, Exception)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in the recommender system."""
    
    def setUp(self):
        """Set up test recommender."""
        self.config = RecommenderConfig(log_level="ERROR")
        self.recommender = HybridBookRecommender(self.config)
    
    def test_normalize_scores(self):
        """Test score normalization."""
        recommendations = [('Book A', 0.8), ('Book B', 0.6), ('Book C', 0.9)]
        normalized = self.recommender._normalize_scores(recommendations)
        
        self.assertIsInstance(normalized, dict)
        self.assertEqual(len(normalized), 3)
        
        # Check that scores are normalized to 0-1 range
        for score in normalized.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
        
        # Check that the highest score is 1 and lowest is 0
        scores = list(normalized.values())
        self.assertAlmostEqual(max(scores), 1.0)
        self.assertAlmostEqual(min(scores), 0.0)
    
    def test_normalize_scores_empty(self):
        """Test normalizing empty recommendations."""
        normalized = self.recommender._normalize_scores([])
        self.assertEqual(normalized, {})
    
    def test_normalize_scores_same_values(self):
        """Test normalizing recommendations with same scores."""
        recommendations = [('Book A', 0.5), ('Book B', 0.5), ('Book C', 0.5)]
        normalized = self.recommender._normalize_scores(recommendations)
        
        # All scores should be 1.0 when they're all the same
        for score in normalized.values():
            self.assertEqual(score, 1.0)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRecommenderConfig,
        TestContentBasedRecommender,
        TestCollaborativeRecommender,
        TestPopularityRecommender,
        TestHybridBookRecommender,
        TestErrorHandling,
        TestUtilityFunctions
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}") 