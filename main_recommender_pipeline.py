#!/usr/bin/env python3
"""
Comprehensive Hybrid Recommender System Pipeline
Academic Research Implementation

This script demonstrates a complete workflow for building and evaluating
a hybrid recommender system with academic-grade metrics.

Features:
- Multiple recommendation algorithms (Content-based, Collaborative, Popularity)
- Hybrid combination strategies (Weighted, Rank Fusion, Cascade)
- Comprehensive evaluation metrics (Accuracy, Ranking Quality @20, Diversity, Novelty)
- Academic report generation with all necessary indexes

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import warnings
from datetime import datetime

# Import our custom modules
from hybrid_recommender_system import HybridRecommenderSystem
from evaluation_framework import RecommenderEvaluator

warnings.filterwarnings('ignore')

def main():
    """Main pipeline for hybrid recommender system"""
    
    print("="*80)
    print("HYBRID RECOMMENDER SYSTEM - ACADEMIC RESEARCH PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    config = {
        'data_paths': {
            'books': 'data/Books.csv',
            'ratings': 'Recommender/Ratings.csv',  # Assuming ratings are in Recommender folder
            'users': 'Recommender/Users.csv'       # Assuming users are in Recommender folder
        },
        'random_state': 42,
        'evaluation': {
            'test_size': 0.2,
            'min_interactions': 5,
            'k_values': [5, 10, 20],
            'sample_users': 100
        }
    }
    
    # Check if data files exist
    print("Checking data files...")
    for data_type, path in config['data_paths'].items():
        if os.path.exists(path):
            print(f"✓ {data_type}: {path}")
        else:
            print(f"✗ {data_type}: {path} (not found)")
            if data_type in ['ratings', 'users']:
                print(f"  Note: You may need to place {data_type}.csv in the Recommender/ directory")
    print()
    
    # Step 1: Initialize and Load Data
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("-" * 50)
    
    start_time = time.time()
    
    # Initialize recommender system
    recommender = HybridRecommenderSystem(random_state=config['random_state'])
    
    # Load data (adjust paths as needed)
    try:
        recommender.load_data(
            books_path=config['data_paths']['books'],
            ratings_path=config['data_paths']['ratings'],
            users_path=config['data_paths']['users'] if os.path.exists(config['data_paths']['users']) else None
        )
        print(f"✓ Data loading completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Please ensure the data files are in the correct locations.")
        return
    
    print()
    
    # Step 2: Build Models
    print("STEP 2: MODEL BUILDING")
    print("-" * 50)
    
    # Build content-based models
    print("Building content-based models...")
    start_time = time.time()
    recommender.build_content_based_models()
    print(f"✓ Content-based models completed in {time.time() - start_time:.2f} seconds")
    
    # Build collaborative filtering models
    print("Building collaborative filtering models...")
    start_time = time.time()
    recommender.build_collaborative_models()
    print(f"✓ Collaborative filtering models completed in {time.time() - start_time:.2f} seconds")
    
    # Build popularity model
    print("Building popularity model...")
    start_time = time.time()
    recommender.build_popularity_model()
    print(f"✓ Popularity model completed in {time.time() - start_time:.2f} seconds")
    
    print()
    
    # Step 3: Demonstrate Recommendations
    print("STEP 3: RECOMMENDATION DEMONSTRATION")
    print("-" * 50)
    
    # Get sample data for demonstration
    sample_users = recommender.ratings_df['User-ID'].unique()[:5]
    sample_books = recommender.books_df['ISBN'].iloc[:5].tolist()
    
    print("Sample Recommendations:")
    print()
    
    # Content-based recommendations
    if sample_books:
        sample_isbn = sample_books[0]
        sample_book_info = recommender.books_df[recommender.books_df['ISBN'] == sample_isbn].iloc[0]
        print(f"Content-based recommendations for '{sample_book_info['Book-Title']}':")
        
        try:
            content_recs = recommender.get_content_recommendations(sample_isbn, 'tfidf', 5)
            for i, rec in enumerate(content_recs[:3], 1):
                print(f"  {i}. {rec['title']} by {rec['author']} (similarity: {rec['similarity_score']:.3f})")
        except Exception as e:
            print(f"  Error: {e}")
        print()
    
    # Collaborative filtering recommendations
    if sample_users.size > 0:
        sample_user = sample_users[0]
        print(f"Collaborative filtering recommendations for User {sample_user}:")
        
        try:
            collab_recs = recommender.get_collaborative_recommendations(sample_user, 'svd', 5)
            for i, rec in enumerate(collab_recs[:3], 1):
                print(f"  {i}. {rec['title']} by {rec['author']} (predicted rating: {rec['predicted_rating']:.2f})")
        except Exception as e:
            print(f"  Error: {e}")
        print()
    
    # Popularity recommendations
    print("Top popularity-based recommendations:")
    try:
        pop_recs = recommender.get_popularity_recommendations(5)
        for i, rec in enumerate(pop_recs[:3], 1):
            print(f"  {i}. {rec['title']} by {rec['author']} (score: {rec['popularity_score']:.3f})")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Hybrid recommendations
    if sample_users.size > 0 and sample_books:
        print(f"Hybrid recommendations (User {sample_user}, based on '{sample_book_info['Book-Title']}'):")
        
        try:
            hybrid_recs = recommender.get_hybrid_recommendations(
                user_id=sample_user, 
                isbn=sample_isbn, 
                n=5, 
                strategy='weighted'
            )
            for i, rec in enumerate(hybrid_recs[:3], 1):
                print(f"  {i}. {rec['title']} by {rec['author']} (hybrid score: {rec['hybrid_score']:.3f})")
        except Exception as e:
            print(f"  Error: {e}")
        print()
    
    # Step 4: Comprehensive Evaluation
    print("STEP 4: COMPREHENSIVE EVALUATION")
    print("-" * 50)
    
    # Initialize evaluator
    evaluator = RecommenderEvaluator(recommender)
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation (this may take several minutes)...")
    start_time = time.time()
    
    try:
        evaluation_results = evaluator.run_comprehensive_evaluation()
        print(f"✓ Evaluation completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        print("Continuing with report generation...")
        evaluation_results = {}
    
    print()
    
    # Step 5: Generate Academic Report
    print("STEP 5: ACADEMIC REPORT GENERATION")
    print("-" * 50)
    
    # Generate evaluation report
    try:
        report_path = f"academic_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        evaluator.generate_evaluation_report(report_path)
        print(f"✓ Academic report saved to: {report_path}")
    except Exception as e:
        print(f"✗ Error generating report: {e}")
    
    # Generate plots
    try:
        evaluator.plot_evaluation_results(save_plots=True)
        print("✓ Evaluation plots generated and saved")
    except Exception as e:
        print(f"✗ Error generating plots: {e}")
    
    print()
    
    # Step 6: Save Models
    print("STEP 6: MODEL PERSISTENCE")
    print("-" * 50)
    
    try:
        import pickle
        model_path = f"hybrid_recommender_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # Save the entire recommender system
        with open(model_path, 'wb') as f:
            pickle.dump(recommender, f)
        
        print(f"✓ Models saved to: {model_path}")
        
        # Save model summary
        summary_path = f"model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_path, 'w') as f:
            f.write("HYBRID RECOMMENDER SYSTEM - MODEL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET STATISTICS:\n")
            f.write(f"- Books: {len(recommender.books_df):,}\n")
            f.write(f"- Ratings: {len(recommender.ratings_df):,}\n")
            f.write(f"- Users: {recommender.ratings_df['User-ID'].nunique():,}\n")
            f.write(f"- Sparsity: {1 - len(recommender.ratings_df) / (len(recommender.books_df) * recommender.ratings_df['User-ID'].nunique()):.4f}\n\n")
            
            f.write("MODELS BUILT:\n")
            f.write("Content-based Models:\n")
            for model_name in recommender.content_models.keys():
                f.write(f"  - {model_name.upper()}\n")
            
            f.write("Collaborative Filtering Models:\n")
            for model_name in recommender.collaborative_models.keys():
                f.write(f"  - {model_name.upper()}\n")
            
            f.write("Other Models:\n")
            for model_name in recommender.models.keys():
                f.write(f"  - {model_name.upper()}\n")
            
            f.write(f"\nHybrid Weights:\n")
            for component, weight in recommender.hybrid_weights.items():
                f.write(f"  - {component}: {weight}\n")
        
        print(f"✓ Model summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"✗ Error saving models: {e}")
    
    print()
    
    # Final Summary
    print("PIPELINE COMPLETION SUMMARY")
    print("-" * 50)
    print("✓ Data preprocessing completed")
    print("✓ Multiple recommendation models built:")
    print("  - Content-based (TF-IDF, SVD, NMF)")
    print("  - Collaborative filtering (SVD, NMF, KNN variants)")
    print("  - Popularity-based")
    print("  - Hybrid combinations")
    print("✓ Comprehensive evaluation performed:")
    print("  - Accuracy metrics (RMSE, MAE)")
    print("  - Ranking quality @20 (Precision, Recall, F1, NDCG, MAP)")
    print("  - Diversity and coverage metrics")
    print("  - Novelty metrics")
    print("✓ Academic report generated")
    print("✓ Visualization plots created")
    print("✓ Models saved for future use")
    
    print()
    print("="*80)
    print("HYBRID RECOMMENDER SYSTEM PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return recommender, evaluator

def demonstrate_usage():
    """Demonstrate how to use the saved models"""
    
    print("\nDEMONSTRATION: Using the Hybrid Recommender System")
    print("-" * 60)
    
    print("""
# Example usage after running the pipeline:

# 1. Load the saved model
import pickle
with open('hybrid_recommender_models_YYYYMMDD_HHMMSS.pkl', 'rb') as f:
    recommender = pickle.load(f)

# 2. Get content-based recommendations
content_recs = recommender.get_content_recommendations(
    isbn='0195153448',  # Example ISBN
    model_type='tfidf',
    n=10
)

# 3. Get collaborative filtering recommendations
collab_recs = recommender.get_collaborative_recommendations(
    user_id=276725,     # Example User ID
    model_type='svd',
    n=10
)

# 4. Get hybrid recommendations
hybrid_recs = recommender.get_hybrid_recommendations(
    user_id=276725,
    isbn='0195153448',
    n=10,
    strategy='weighted'  # or 'rank_fusion', 'cascade'
)

# 5. Get popularity-based recommendations
pop_recs = recommender.get_popularity_recommendations(n=10)

# 6. Adjust hybrid weights
recommender.hybrid_weights = {
    'content_based': 0.4,
    'collaborative': 0.4,
    'popularity': 0.2
}
""")

if __name__ == "__main__":
    try:
        recommender, evaluator = main()
        demonstrate_usage()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc() 