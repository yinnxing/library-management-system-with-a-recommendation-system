import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from hybrid_recommender import HybridBookRecommender

def run_comparison():
    """
    Compare the original and improved recommender systems side by side.
    This script evaluates both systems on the same dataset and reports
    performance metrics, recommendation quality, and efficiency.
    """
    print("\n===== RECOMMENDER SYSTEM COMPARISON =====")
    print("Comparing original vs. improved hybrid recommender systems")
    
    # Create results dictionary to store metrics
    results = {
        'original': {},
        'improved': {}
    }
    
    # ===== ORIGINAL RECOMMENDER =====
    print("\n===== SETTING UP ORIGINAL RECOMMENDER =====")
    original = HybridBookRecommender(
        content_weight=0.3,
        collab_weight=0.5,
        popular_weight=0.2,
        optimize_weights=False
    )
    
    # Load data
    print("Loading data for original recommender...")
    original.load_data(
        books_path='Books.csv',
        ratings_path='Ratings.csv',
        users_path='Users.csv'
    )
    
    # Preprocess data
    print("Preprocessing data for original recommender...")
    original.preprocess_data(
        min_book_ratings=10,
        min_user_ratings=3,
        verbose=False
    )
    
    # Split data
    print("Splitting data for original recommender...")
    original.split_data(test_size=0.2, random_state=42, verbose=False)
    
    # Build models
    print("Building models for original recommender...")
    start_time = time.time()
    original.build_content_based_model(verbose=False)
    original.build_collaborative_model(n_components=100, verbose=False)
    original_build_time = time.time() - start_time
    results['original']['build_time'] = original_build_time
    
    # ===== IMPROVED RECOMMENDER =====
    print("\n===== SETTING UP IMPROVED RECOMMENDER =====")
    improved = HybridBookRecommender(
        content_weight=0.6,
        collab_weight=0.3,
        popular_weight=0.1,
        optimize_weights=False
    )
    
    # Load data
    print("Loading data for improved recommender...")
    improved.load_data(
        books_path='Books.csv',
        ratings_path='Ratings.csv',
        users_path='Users.csv'
    )
    
    # Preprocess data
    print("Preprocessing data for improved recommender...")
    improved.preprocess_data(
        min_book_ratings=10,
        min_user_ratings=3,
        verbose=False
    )
    
    # Split data
    print("Splitting data for improved recommender...")
    improved.split_data(test_size=0.2, random_state=42, verbose=False)
    
    # Build models
    print("Building models for improved recommender...")
    start_time = time.time()
    improved.build_enhanced_content_model(verbose=False)
    improved.build_ann_index(verbose=False)
    improved.build_advanced_collaborative_models(verbose=False)
    improved_build_time = time.time() - start_time
    results['improved']['build_time'] = improved_build_time
    
    # ===== PERFORMANCE COMPARISON =====
    print("\n===== PERFORMANCE COMPARISON =====")
    
    # Test books for comparison
    test_books = ["The Da Vinci Code", "Harry Potter and the Sorcerer's Stone", "To Kill a Mockingbird"]
    valid_test_books = [book for book in test_books if book in original.merged_df['Book-Title'].values]
    
    if not valid_test_books:
        print("No valid test books found. Selecting random books...")
        valid_test_books = np.random.choice(original.merged_df['Book-Title'].unique(), 3).tolist()
    
    recommendation_times = {
        'original': [],
        'improved': []
    }
    
    diversity_scores = {
        'original': [],
        'improved': []
    }
    
    for book in valid_test_books:
        print(f"\nTesting with book: '{book}'")
        
        # Original recommender
        start_time = time.time()
        original_recs = original.get_hybrid_recommendations(book, n=10)
        original_time = time.time() - start_time
        recommendation_times['original'].append(original_time)
        
        # Improved recommender
        start_time = time.time()
        improved_recs = improved.get_optimized_hybrid_recommendations(book, n=10)
        improved_time = time.time() - start_time
        recommendation_times['improved'].append(improved_time)
        
        # Calculate diversity
        original_diversity = improved.diversity_score(original_recs)
        improved_diversity = improved.diversity_score(improved_recs)
        
        diversity_scores['original'].append(original_diversity)
        diversity_scores['improved'].append(improved_diversity)
        
        # Print results
        print(f"  - Original recommendation time: {original_time:.4f} seconds")
        print(f"  - Improved recommendation time: {improved_time:.4f} seconds")
        print(f"  - Speedup: {original_time/improved_time:.2f}x")
        print(f"  - Original diversity: {original_diversity:.4f}")
        print(f"  - Improved diversity: {improved_diversity:.4f}")
    
    # Calculate average recommendation time
    avg_original_time = sum(recommendation_times['original']) / len(recommendation_times['original'])
    avg_improved_time = sum(recommendation_times['improved']) / len(recommendation_times['improved'])
    avg_speedup = avg_original_time / avg_improved_time
    
    results['original']['avg_recommendation_time'] = avg_original_time
    results['improved']['avg_recommendation_time'] = avg_improved_time
    results['improved']['avg_speedup'] = avg_speedup
    
    # Calculate average diversity
    avg_original_diversity = sum(diversity_scores['original']) / len(diversity_scores['original'])
    avg_improved_diversity = sum(diversity_scores['improved']) / len(diversity_scores['improved'])
    diversity_improvement = (avg_improved_diversity - avg_original_diversity) / avg_original_diversity * 100
    
    results['original']['avg_diversity'] = avg_original_diversity
    results['improved']['avg_diversity'] = avg_improved_diversity
    results['improved']['diversity_improvement'] = diversity_improvement
    
    print(f"\nAverage recommendation time (original): {avg_original_time:.4f} seconds")
    print(f"Average recommendation time (improved): {avg_improved_time:.4f} seconds")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Average diversity (original): {avg_original_diversity:.4f}")
    print(f"Average diversity (improved): {avg_improved_diversity:.4f}")
    print(f"Diversity improvement: {diversity_improvement:.2f}%")
    
    # ===== RECOMMENDATION QUALITY COMPARISON =====
    print("\n===== RECOMMENDATION QUALITY COMPARISON =====")
    
    # Evaluate both models
    print("Evaluating original recommender...")
    original_metrics = original.evaluate_model(k=10, verbose=False)
    
    print("Evaluating improved recommender...")
    improved_metrics = improved.evaluate_enhanced_model(k=10, verbose=False)
    
    # Store metrics
    for metric in ['RMSE', 'MAE', 'Precision@K', 'Recall@K', 'F1@K']:
        results['original'][metric] = original_metrics[metric]
        results['improved'][metric] = improved_metrics[metric]
    
    # Additional metrics for improved recommender
    results['improved']['Diversity'] = improved_metrics.get('Diversity', 0)
    results['improved']['Coverage'] = improved_metrics.get('Coverage', 0)
    
    # Print comparison
    print("\nMetric comparison:")
    for metric in ['RMSE', 'MAE', 'Precision@K', 'Recall@K', 'F1@K']:
        original_value = original_metrics[metric]
        improved_value = improved_metrics[metric]
        
        if metric == 'RMSE' or metric == 'MAE':
            # Lower is better for error metrics
            improvement = (original_value - improved_value) / original_value * 100
            better = improved_value < original_value
        else:
            # Higher is better for other metrics
            improvement = (improved_value - original_value) / original_value * 100 if original_value > 0 else float('inf')
            better = improved_value > original_value
            
        print(f"{metric}: {original_value:.4f} → {improved_value:.4f} " + 
              f"({'better' if better else 'worse'} by {abs(improvement):.2f}%)")
    
    # Print additional metrics
    print(f"Diversity (improved): {improved_metrics.get('Diversity', 0):.4f}")
    print(f"Coverage (improved): {improved_metrics.get('Coverage', 0):.4f}")
    
    # ===== PREDICTION ACCURACY COMPARISON =====
    print("\n===== PREDICTION ACCURACY COMPARISON =====")
    
    # Sample user-book pairs from test data
    test_users = original.test_data['User-ID'].unique()[:10]
    test_user_books = []
    
    for user_id in test_users:
        user_test_data = original.test_data[original.test_data['User-ID'] == user_id]
        if len(user_test_data) > 0:
            test_user_books.append((user_id, user_test_data.iloc[0]['Book-Title'], user_test_data.iloc[0]['Book-Rating']))
    
    original_errors = []
    improved_errors = []
    
    for user_id, book_title, actual_rating in test_user_books:
        try:
            # Original prediction
            original_pred = original.predict_rating(user_id, book_title)
            original_error = abs(original_pred - actual_rating)
            original_errors.append(original_error)
            
            # Improved prediction
            improved_pred = improved.predict_rating_advanced(user_id, book_title)
            improved_error = abs(improved_pred - actual_rating)
            improved_errors.append(improved_error)
            
            print(f"User {user_id}, Book '{book_title}':")
            print(f"  - Actual: {actual_rating}, Original: {original_pred:.2f}, Improved: {improved_pred:.2f}")
        except Exception as e:
            print(f"Error predicting for user {user_id}, book '{book_title}': {str(e)}")
    
    # Calculate average errors
    if original_errors and improved_errors:
        avg_original_error = sum(original_errors) / len(original_errors)
        avg_improved_error = sum(improved_errors) / len(improved_errors)
        error_improvement = (avg_original_error - avg_improved_error) / avg_original_error * 100
        
        results['original']['avg_prediction_error'] = avg_original_error
        results['improved']['avg_prediction_error'] = avg_improved_error
        results['improved']['prediction_improvement'] = error_improvement
        
        print(f"\nAverage prediction error (original): {avg_original_error:.4f}")
        print(f"Average prediction error (improved): {avg_improved_error:.4f}")
        print(f"Prediction improvement: {error_improvement:.2f}%")
    
    # ===== SUMMARY =====
    print("\n===== SUMMARY =====")
    print("Original recommender:")
    print(f"  - Build time: {results['original']['build_time']:.2f} seconds")
    print(f"  - Recommendation time: {results['original']['avg_recommendation_time']:.4f} seconds")
    print(f"  - RMSE: {results['original']['RMSE']:.4f}")
    print(f"  - F1@K: {results['original']['F1@K']:.4f}")
    
    print("\nImproved recommender:")
    print(f"  - Build time: {results['improved']['build_time']:.2f} seconds")
    print(f"  - Recommendation time: {results['improved']['avg_recommendation_time']:.4f} seconds")
    print(f"  - RMSE: {results['improved']['RMSE']:.4f}")
    print(f"  - F1@K: {results['improved']['F1@K']:.4f}")
    print(f"  - Diversity: {results['improved']['avg_diversity']:.4f}")
    print(f"  - Coverage: {results['improved']['Coverage']:.4f}")
    
    print("\nImprovements:")
    print(f"  - Recommendation speed: {results['improved']['avg_speedup']:.2f}x faster")
    print(f"  - Diversity: {results['improved']['diversity_improvement']:.2f}% better")
    print(f"  - RMSE: {((results['original']['RMSE'] - results['improved']['RMSE']) / results['original']['RMSE'] * 100):.2f}% better")
    print(f"  - F1@K: {((results['improved']['F1@K'] - results['original']['F1@K']) / results['original']['F1@K'] * 100):.2f}% better")
    
    return results

if __name__ == "__main__":
    try:
        results = run_comparison()
        
        # Optional: Create visualizations of the results
        try:
            import matplotlib.pyplot as plt
            
            # Recommendation time comparison
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.bar(['Original', 'Improved'], 
                    [results['original']['avg_recommendation_time'], 
                     results['improved']['avg_recommendation_time']])
            plt.title('Recommendation Time (seconds)')
            plt.ylabel('Time (s)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Metrics comparison
            plt.subplot(1, 2, 2)
            metrics = ['RMSE', 'F1@K']
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, 
                   [results['original']['RMSE'], results['original']['F1@K']], 
                   width, label='Original')
            plt.bar(x + width/2, 
                   [results['improved']['RMSE'], results['improved']['F1@K']], 
                   width, label='Improved')
            
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.title('Recommendation Quality')
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('recommender_comparison.png')
            print("\nVisualization saved as 'recommender_comparison.png'")
            
        except ImportError:
            print("\nMatplotlib not available. Skipping visualization.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc() 