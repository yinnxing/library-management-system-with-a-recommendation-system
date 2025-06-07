import pandas as pd
import numpy as np
import time
import os
from hybrid_recommender import HybridBookRecommender

def test_improved_recommender():
    """Test the improved recommender system with enhanced performance and quality"""
    print("\n===== INITIALIZING IMPROVED RECOMMENDER SYSTEM =====")
    recommender = HybridBookRecommender(
        content_weight=0.6,  # Start with content-heavy weights based on previous evaluation
        collab_weight=0.3,
        popular_weight=0.1,
        optimize_weights=True
    )
    
    print("\n===== LOADING DATASETS =====")
    # Load data
    recommender.load_data(
        books_path='Books.csv',
        ratings_path='Ratings.csv',
        users_path='Users.csv'
    )
    
    print("\n===== PREPROCESSING DATA =====")
    # Use the same preprocessing parameters
    processed_data = recommender.preprocess_data(
        min_book_ratings=10,
        min_user_ratings=3,
        verbose=True
    )
    
    print("\n===== BUILDING ENHANCED CONTENT MODEL =====")
    # Build enhanced content model with sentence transformers
    recommender.build_enhanced_content_model(verbose=True)
    
    print("\n===== BUILDING ANN INDEX =====")
    # Build approximate nearest neighbors index for faster retrieval
    recommender.build_ann_index(verbose=True)
    
    print("\n===== BUILDING ADVANCED COLLABORATIVE MODELS =====")
    # Build advanced collaborative filtering models
    recommender.build_advanced_collaborative_models(verbose=True)
    
    print("\n===== SPLITTING DATA FOR EVALUATION =====")
    train_data, test_data = recommender.split_data(test_size=0.2, random_state=42, verbose=True)
    
    print("\n===== BUILDING STANDARD MODELS FOR COMPARISON =====")
    # Build standard models for comparison
    recommender.build_content_based_model(verbose=False)
    recommender.build_collaborative_model(n_components=100, verbose=False)
    
    print("\n===== PERFORMANCE COMPARISON =====")
    # Compare performance of standard vs. optimized recommendation methods
    test_books = ["The Da Vinci Code", "Harry Potter and the Sorcerer's Stone", "To Kill a Mockingbird"]
    
    for book in test_books:
        if book in recommender.merged_df['Book-Title'].values:
            print(f"\nPerformance test for '{book}':")
            
            # Measure standard recommendation time
            start_time = time.time()
            standard_recs = recommender.get_hybrid_recommendations(book, n=10)
            standard_time = time.time() - start_time
            
            # Measure optimized recommendation time
            start_time = time.time()
            optimized_recs = recommender.get_optimized_hybrid_recommendations(book, n=10)
            optimized_time = time.time() - start_time
            
            # Calculate speedup
            speedup = standard_time / optimized_time if optimized_time > 0 else float('inf')
            
            print(f"  - Standard method: {standard_time:.4f} seconds")
            print(f"  - Optimized method: {optimized_time:.4f} seconds")
            print(f"  - Speedup: {speedup:.2f}x")
            
            # Calculate diversity
            standard_diversity = recommender.diversity_score(standard_recs)
            optimized_diversity = recommender.diversity_score(optimized_recs)
            
            print(f"  - Standard diversity: {standard_diversity:.4f}")
            print(f"  - Optimized diversity: {optimized_diversity:.4f}")
            
            # Compare recommendations
            print("\nStandard recommendations:")
            for i, rec in enumerate(standard_recs[:5], 1):
                print(f"  {i}. {rec}")
                
            print("\nOptimized recommendations:")
            for i, rec in enumerate(optimized_recs[:5], 1):
                print(f"  {i}. {rec}")
    
    print("\n===== OPTIMIZING WEIGHTS WITH BAYESIAN OPTIMIZATION =====")
    # Optimize weights using Bayesian optimization
    recommender.optimize_weights_bayesian(n_iterations=10)
    
    print("\n===== EVALUATING ENHANCED MODEL =====")
    # Evaluate with enhanced metrics
    enhanced_metrics = recommender.evaluate_enhanced_model(k=10, verbose=True)
    
    print("\n===== PREDICTION ACCURACY COMPARISON =====")
    # Compare prediction accuracy of standard vs. advanced methods
    test_users = recommender.test_data['User-ID'].unique()[:5]
    test_user_books = []
    
    for user_id in test_users:
        user_test_data = recommender.test_data[recommender.test_data['User-ID'] == user_id]
        if len(user_test_data) > 0:
            test_user_books.append((user_id, user_test_data.iloc[0]['Book-Title'], user_test_data.iloc[0]['Book-Rating']))
    
    print("\nPrediction comparison:")
    mae_standard = []
    mae_advanced = []
    
    for user_id, book_title, actual_rating in test_user_books:
        try:
            # Standard prediction
            standard_pred = recommender.predict_rating(user_id, book_title)
            standard_error = abs(standard_pred - actual_rating)
            mae_standard.append(standard_error)
            
            # Advanced prediction
            advanced_pred = recommender.predict_rating_advanced(user_id, book_title)
            advanced_error = abs(advanced_pred - actual_rating)
            mae_advanced.append(advanced_error)
            
            print(f"User {user_id}, Book '{book_title}':")
            print(f"  - Actual rating: {actual_rating}")
            print(f"  - Standard prediction: {standard_pred:.2f} (error: {standard_error:.2f})")
            print(f"  - Advanced prediction: {advanced_pred:.2f} (error: {advanced_error:.2f})")
        except Exception as e:
            print(f"Error predicting for user {user_id}, book '{book_title}': {str(e)}")
    
    # Calculate average MAE
    if mae_standard and mae_advanced:
        avg_mae_standard = sum(mae_standard) / len(mae_standard)
        avg_mae_advanced = sum(mae_advanced) / len(mae_advanced)
        improvement = (avg_mae_standard - avg_mae_advanced) / avg_mae_standard * 100
        
        print(f"\nAverage MAE (standard): {avg_mae_standard:.4f}")
        print(f"Average MAE (advanced): {avg_mae_advanced:.4f}")
        print(f"Improvement: {improvement:.2f}%")
    
    print("\n===== SAVING IMPROVED MODEL =====")
    # Save the improved model
    recommender.save_model('improved_book_recommender_model.pkl')
    
    print("\nImproved recommender system testing complete!")

if __name__ == "__main__":
    try:
        test_improved_recommender()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc() 