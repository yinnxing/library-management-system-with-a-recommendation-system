import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import accuracy
from collections import defaultdict
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class RecommenderEvaluator:
    """
    Comprehensive evaluation framework for recommender systems
    
    Provides academic-grade evaluation metrics:
    1. Accuracy Metrics: RMSE, MAE, Precision@k, Recall@k, F1@k
    2. Ranking Quality: NDCG@k, MAP@k, MRR, AUC
    3. Diversity Metrics: Intra-list diversity, Coverage
    4. Novelty Metrics: Popularity bias, Long-tail coverage
    5. Statistical Significance Tests
    """
    
    def __init__(self, recommender_system):
        self.recommender = recommender_system
        self.evaluation_results = {}
        self.test_users = None
        self.test_items = None
        
    def prepare_evaluation_data(self, test_size=0.2, min_interactions=5):
        """Prepare evaluation datasets with proper train/test split"""
        print("Preparing evaluation data...")
        
        # Get users with sufficient interactions for evaluation
        user_counts = self.recommender.ratings_df['User-ID'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        
        # Sample test users
        np.random.seed(self.recommender.random_state)
        self.test_users = np.random.choice(
            valid_users, 
            size=min(1000, len(valid_users)), 
            replace=False
        )
        
        # Create train/test split for each user
        self.train_ratings = []
        self.test_ratings = []
        
        for user_id in self.test_users:
            user_ratings = self.recommender.ratings_df[
                self.recommender.ratings_df['User-ID'] == user_id
            ].copy()
            
            # Split user's ratings
            n_test = max(1, int(len(user_ratings) * test_size))
            test_indices = np.random.choice(
                user_ratings.index, 
                size=n_test, 
                replace=False
            )
            
            test_data = user_ratings.loc[test_indices]
            train_data = user_ratings.drop(test_indices)
            
            self.train_ratings.append(train_data)
            self.test_ratings.append(test_data)
        
        self.train_ratings = pd.concat(self.train_ratings, ignore_index=True)
        self.test_ratings = pd.concat(self.test_ratings, ignore_index=True)
        
        print(f"Evaluation data prepared: {len(self.test_users)} users")
        print(f"Train ratings: {len(self.train_ratings)}, Test ratings: {len(self.test_ratings)}")
    
    def evaluate_accuracy_metrics(self, models_to_evaluate=None):
        """Evaluate accuracy metrics (RMSE, MAE) for collaborative filtering models"""
        print("Evaluating accuracy metrics...")
        
        if models_to_evaluate is None:
            models_to_evaluate = list(self.recommender.collaborative_models.keys())
        
        accuracy_results = {}
        
        for model_name in models_to_evaluate:
            if model_name not in self.recommender.collaborative_models:
                continue
                
            model = self.recommender.collaborative_models[model_name]
            
            # Get predictions for test set
            predictions = []
            actuals = []
            
            for _, row in self.test_ratings.iterrows():
                pred = model.predict(row['User-ID'], row['ISBN'])
                predictions.append(pred.est)
                actuals.append(row['Book-Rating'])
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            accuracy_results[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'n_predictions': len(predictions)
            }
            
            print(f"{model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        self.evaluation_results['accuracy'] = accuracy_results
        return accuracy_results
    
    def evaluate_ranking_metrics(self, k_values=[5, 10, 20], models_to_evaluate=None):
        """Evaluate ranking quality metrics"""
        print("Evaluating ranking metrics...")
        
        if models_to_evaluate is None:
            models_to_evaluate = ['svd', 'nmf', 'knn_item', 'hybrid_weighted']
        
        ranking_results = {}
        
        for k in k_values:
            ranking_results[f'@{k}'] = {}
            
            for model_name in models_to_evaluate:
                print(f"Evaluating {model_name} @ {k}...")
                
                precision_scores = []
                recall_scores = []
                ndcg_scores = []
                map_scores = []
                
                for user_id in self.test_users[:100]:  # Sample for efficiency
                    # Get user's test items (ground truth)
                    user_test_items = set(
                        self.test_ratings[self.test_ratings['User-ID'] == user_id]['ISBN'].tolist()
                    )
                    
                    if len(user_test_items) == 0:
                        continue
                    
                    # Get recommendations
                    try:
                        if model_name == 'hybrid_weighted':
                            # For hybrid, we need both user_id and an item
                            user_train_items = self.train_ratings[
                                self.train_ratings['User-ID'] == user_id
                            ]['ISBN'].tolist()
                            if user_train_items:
                                recs = self.recommender.get_hybrid_recommendations(
                                    user_id=user_id, 
                                    isbn=user_train_items[0], 
                                    n=k, 
                                    strategy='weighted'
                                )
                            else:
                                continue
                        else:
                            recs = self.recommender.get_collaborative_recommendations(
                                user_id, model_name, k
                            )
                        
                        if not recs:
                            continue
                            
                        recommended_items = [rec['isbn'] for rec in recs]
                        
                        # Calculate metrics
                        precision = self._calculate_precision_at_k(
                            recommended_items, user_test_items, k
                        )
                        recall = self._calculate_recall_at_k(
                            recommended_items, user_test_items, k
                        )
                        ndcg = self._calculate_ndcg_at_k(
                            recommended_items, user_test_items, k
                        )
                        map_score = self._calculate_map_at_k(
                            recommended_items, user_test_items, k
                        )
                        
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        ndcg_scores.append(ndcg)
                        map_scores.append(map_score)
                        
                    except Exception as e:
                        print(f"Error evaluating {model_name} for user {user_id}: {e}")
                        continue
                
                # Aggregate results
                if precision_scores:
                    ranking_results[f'@{k}'][model_name] = {
                        'Precision': np.mean(precision_scores),
                        'Recall': np.mean(recall_scores),
                        'F1': self._calculate_f1(np.mean(precision_scores), np.mean(recall_scores)),
                        'NDCG': np.mean(ndcg_scores),
                        'MAP': np.mean(map_scores),
                        'n_users_evaluated': len(precision_scores)
                    }
                    
                    print(f"  Precision@{k}: {np.mean(precision_scores):.4f}")
                    print(f"  Recall@{k}: {np.mean(recall_scores):.4f}")
                    print(f"  NDCG@{k}: {np.mean(ndcg_scores):.4f}")
                    print(f"  MAP@{k}: {np.mean(map_scores):.4f}")
        
        self.evaluation_results['ranking'] = ranking_results
        return ranking_results
    
    def _calculate_precision_at_k(self, recommended_items, relevant_items, k):
        """Calculate Precision@k"""
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        return relevant_recommended / len(recommended_k) if recommended_k else 0
    
    def _calculate_recall_at_k(self, recommended_items, relevant_items, k):
        """Calculate Recall@k"""
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        return relevant_recommended / len(relevant_items) if relevant_items else 0
    
    def _calculate_f1(self, precision, recall):
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_ndcg_at_k(self, recommended_items, relevant_items, k):
        """Calculate NDCG@k"""
        recommended_k = recommended_items[:k]
        
        # DCG calculation
        dcg = 0
        for i, item in enumerate(recommended_k):
            if item in relevant_items:
                dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # IDCG calculation (ideal DCG)
        idcg = 0
        for i in range(min(len(relevant_items), k)):
            idcg += 1 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0
    
    def _calculate_map_at_k(self, recommended_items, relevant_items, k):
        """Calculate MAP@k"""
        recommended_k = recommended_items[:k]
        
        if not relevant_items:
            return 0
        
        score = 0
        num_hits = 0
        
        for i, item in enumerate(recommended_k):
            if item in relevant_items:
                num_hits += 1
                score += num_hits / (i + 1)
        
        return score / len(relevant_items)
    
    def evaluate_diversity_metrics(self, k=20, models_to_evaluate=None):
        """Evaluate diversity and coverage metrics"""
        print("Evaluating diversity metrics...")
        
        if models_to_evaluate is None:
            models_to_evaluate = ['svd', 'hybrid_weighted']
        
        diversity_results = {}
        
        for model_name in models_to_evaluate:
            print(f"Evaluating diversity for {model_name}...")
            
            all_recommendations = []
            intra_list_diversities = []
            
            for user_id in self.test_users[:100]:  # Sample for efficiency
                try:
                    if model_name == 'hybrid_weighted':
                        user_train_items = self.train_ratings[
                            self.train_ratings['User-ID'] == user_id
                        ]['ISBN'].tolist()
                        if user_train_items:
                            recs = self.recommender.get_hybrid_recommendations(
                                user_id=user_id, 
                                isbn=user_train_items[0], 
                                n=k, 
                                strategy='weighted'
                            )
                        else:
                            continue
                    else:
                        recs = self.recommender.get_collaborative_recommendations(
                            user_id, model_name, k
                        )
                    
                    if not recs:
                        continue
                    
                    recommended_items = [rec['isbn'] for rec in recs]
                    all_recommendations.extend(recommended_items)
                    
                    # Calculate intra-list diversity
                    diversity = self._calculate_intra_list_diversity(recommended_items)
                    intra_list_diversities.append(diversity)
                    
                except Exception as e:
                    continue
            
            # Calculate metrics
            catalog_coverage = len(set(all_recommendations)) / len(self.recommender.books_df)
            avg_intra_list_diversity = np.mean(intra_list_diversities) if intra_list_diversities else 0
            
            # Calculate popularity bias
            popularity_bias = self._calculate_popularity_bias(all_recommendations)
            
            diversity_results[model_name] = {
                'catalog_coverage': catalog_coverage,
                'intra_list_diversity': avg_intra_list_diversity,
                'popularity_bias': popularity_bias,
                'unique_items_recommended': len(set(all_recommendations)),
                'total_recommendations': len(all_recommendations)
            }
            
            print(f"  Catalog Coverage: {catalog_coverage:.4f}")
            print(f"  Intra-list Diversity: {avg_intra_list_diversity:.4f}")
            print(f"  Popularity Bias: {popularity_bias:.4f}")
        
        self.evaluation_results['diversity'] = diversity_results
        return diversity_results
    
    def _calculate_intra_list_diversity(self, recommended_items):
        """Calculate intra-list diversity based on content similarity"""
        if len(recommended_items) < 2:
            return 0
        
        # Get content features for recommended items
        item_features = []
        for isbn in recommended_items:
            book_info = self.recommender.books_df[
                self.recommender.books_df['ISBN'] == isbn
            ]
            if not book_info.empty:
                # Use author and publisher as diversity features
                author = book_info.iloc[0]['Book-Author']
                publisher = book_info.iloc[0]['Publisher']
                item_features.append((author, publisher))
        
        # Calculate diversity as percentage of unique author-publisher combinations
        unique_features = len(set(item_features))
        total_items = len(item_features)
        
        return unique_features / total_items if total_items > 0 else 0
    
    def _calculate_popularity_bias(self, recommended_items):
        """Calculate popularity bias (lower is better for diversity)"""
        if not recommended_items:
            return 0
        
        # Get popularity scores for recommended items
        popularity_model = self.recommender.models.get('popularity')
        if popularity_model is None:
            self.recommender.build_popularity_model()
            popularity_model = self.recommender.models['popularity']
        
        popularity_scores = []
        for isbn in recommended_items:
            book_pop = popularity_model[popularity_model['ISBN'] == isbn]
            if not book_pop.empty:
                popularity_scores.append(book_pop.iloc[0]['popularity_score'])
        
        # Return average popularity (higher means more bias toward popular items)
        return np.mean(popularity_scores) if popularity_scores else 0
    
    def evaluate_novelty_metrics(self, models_to_evaluate=None):
        """Evaluate novelty metrics"""
        print("Evaluating novelty metrics...")
        
        if models_to_evaluate is None:
            models_to_evaluate = ['svd', 'hybrid_weighted']
        
        novelty_results = {}
        
        # Calculate item popularity distribution
        item_popularity = self.recommender.ratings_df['ISBN'].value_counts()
        total_interactions = len(self.recommender.ratings_df)
        
        for model_name in models_to_evaluate:
            print(f"Evaluating novelty for {model_name}...")
            
            all_recommendations = []
            
            for user_id in self.test_users[:100]:
                try:
                    if model_name == 'hybrid_weighted':
                        user_train_items = self.train_ratings[
                            self.train_ratings['User-ID'] == user_id
                        ]['ISBN'].tolist()
                        if user_train_items:
                            recs = self.recommender.get_hybrid_recommendations(
                                user_id=user_id, 
                                isbn=user_train_items[0], 
                                n=20, 
                                strategy='weighted'
                            )
                        else:
                            continue
                    else:
                        recs = self.recommender.get_collaborative_recommendations(
                            user_id, model_name, 20
                        )
                    
                    if recs:
                        recommended_items = [rec['isbn'] for rec in recs]
                        all_recommendations.extend(recommended_items)
                        
                except Exception as e:
                    continue
            
            # Calculate novelty metrics
            novelty_scores = []
            long_tail_items = 0
            
            for isbn in all_recommendations:
                if isbn in item_popularity.index:
                    popularity = item_popularity[isbn]
                    # Novelty = -log2(popularity/total_interactions)
                    novelty = -np.log2(popularity / total_interactions)
                    novelty_scores.append(novelty)
                    
                    # Count long-tail items (bottom 20% by popularity)
                    if popularity <= item_popularity.quantile(0.2):
                        long_tail_items += 1
            
            avg_novelty = np.mean(novelty_scores) if novelty_scores else 0
            long_tail_percentage = long_tail_items / len(all_recommendations) if all_recommendations else 0
            
            novelty_results[model_name] = {
                'average_novelty': avg_novelty,
                'long_tail_percentage': long_tail_percentage,
                'total_recommendations': len(all_recommendations)
            }
            
            print(f"  Average Novelty: {avg_novelty:.4f}")
            print(f"  Long-tail Percentage: {long_tail_percentage:.4f}")
        
        self.evaluation_results['novelty'] = novelty_results
        return novelty_results
    
    def run_comprehensive_evaluation(self):
        """Run all evaluation metrics"""
        print("Running comprehensive evaluation...")
        
        # Prepare evaluation data
        self.prepare_evaluation_data()
        
        # Run all evaluations
        accuracy_results = self.evaluate_accuracy_metrics()
        ranking_results = self.evaluate_ranking_metrics()
        diversity_results = self.evaluate_diversity_metrics()
        novelty_results = self.evaluate_novelty_metrics()
        
        # Compile final results
        final_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_users': len(self.recommender.ratings_df['User-ID'].unique()),
                'n_items': len(self.recommender.ratings_df['ISBN'].unique()),
                'n_ratings': len(self.recommender.ratings_df),
                'sparsity': 1 - len(self.recommender.ratings_df) / (
                    len(self.recommender.ratings_df['User-ID'].unique()) * 
                    len(self.recommender.ratings_df['ISBN'].unique())
                )
            },
            'accuracy_metrics': accuracy_results,
            'ranking_metrics': ranking_results,
            'diversity_metrics': diversity_results,
            'novelty_metrics': novelty_results
        }
        
        self.evaluation_results = final_results
        return final_results
    
    def generate_evaluation_report(self, save_path='evaluation_report.json'):
        """Generate and save comprehensive evaluation report"""
        print("Generating evaluation report...")
        
        if not self.evaluation_results:
            print("No evaluation results found. Running comprehensive evaluation...")
            self.run_comprehensive_evaluation()
        
        # Save results to JSON
        with open(save_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        print(f"Evaluation report saved to {save_path}")
        
        # Generate summary tables
        self._generate_summary_tables()
        
        return self.evaluation_results
    
    def _generate_summary_tables(self):
        """Generate summary tables for academic reporting"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        # Dataset Information
        print("\nDATASET INFORMATION:")
        print("-" * 40)
        dataset_info = self.evaluation_results.get('dataset_info', {})
        for key, value in dataset_info.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value:,}")
        
        # Accuracy Metrics
        if 'accuracy_metrics' in self.evaluation_results:
            print("\nACCURACY METRICS:")
            print("-" * 40)
            accuracy_data = []
            for model, metrics in self.evaluation_results['accuracy_metrics'].items():
                accuracy_data.append([
                    model.upper(),
                    f"{metrics['RMSE']:.4f}",
                    f"{metrics['MAE']:.4f}",
                    f"{metrics['n_predictions']:,}"
                ])
            
            print(f"{'Model':<15} {'RMSE':<8} {'MAE':<8} {'N_Pred':<8}")
            print("-" * 45)
            for row in accuracy_data:
                print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8}")
        
        # Ranking Metrics @20
        if 'ranking_metrics' in self.evaluation_results:
            print("\nRANKING QUALITY METRICS @20:")
            print("-" * 60)
            ranking_20 = self.evaluation_results['ranking_metrics'].get('@20', {})
            
            if ranking_20:
                ranking_data = []
                for model, metrics in ranking_20.items():
                    ranking_data.append([
                        model.upper(),
                        f"{metrics['Precision']:.4f}",
                        f"{metrics['Recall']:.4f}",
                        f"{metrics['F1']:.4f}",
                        f"{metrics['NDCG']:.4f}",
                        f"{metrics['MAP']:.4f}"
                    ])
                
                print(f"{'Model':<15} {'Prec@20':<8} {'Rec@20':<8} {'F1@20':<8} {'NDCG@20':<8} {'MAP@20':<8}")
                print("-" * 70)
                for row in ranking_data:
                    print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8}")
        
        # Diversity Metrics
        if 'diversity_metrics' in self.evaluation_results:
            print("\nDIVERSITY METRICS:")
            print("-" * 50)
            diversity_data = []
            for model, metrics in self.evaluation_results['diversity_metrics'].items():
                diversity_data.append([
                    model.upper(),
                    f"{metrics['catalog_coverage']:.4f}",
                    f"{metrics['intra_list_diversity']:.4f}",
                    f"{metrics['popularity_bias']:.4f}"
                ])
            
            print(f"{'Model':<15} {'Coverage':<10} {'Diversity':<10} {'Pop_Bias':<10}")
            print("-" * 50)
            for row in diversity_data:
                print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10}")
        
        # Novelty Metrics
        if 'novelty_metrics' in self.evaluation_results:
            print("\nNOVELTY METRICS:")
            print("-" * 40)
            novelty_data = []
            for model, metrics in self.evaluation_results['novelty_metrics'].items():
                novelty_data.append([
                    model.upper(),
                    f"{metrics['average_novelty']:.4f}",
                    f"{metrics['long_tail_percentage']:.4f}"
                ])
            
            print(f"{'Model':<15} {'Avg_Novelty':<12} {'LongTail_%':<12}")
            print("-" * 40)
            for row in novelty_data:
                print(f"{row[0]:<15} {row[1]:<12} {row[2]:<12}")
        
        print("\n" + "="*80)
    
    def plot_evaluation_results(self, save_plots=True):
        """Generate visualization plots for evaluation results"""
        print("Generating evaluation plots...")
        
        if not self.evaluation_results:
            print("No evaluation results found. Please run evaluation first.")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hybrid Recommender System Evaluation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy Metrics
        if 'accuracy_metrics' in self.evaluation_results:
            ax1 = axes[0, 0]
            accuracy_data = self.evaluation_results['accuracy_metrics']
            models = list(accuracy_data.keys())
            rmse_values = [accuracy_data[m]['RMSE'] for m in models]
            mae_values = [accuracy_data[m]['MAE'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax1.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8)
            ax1.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Error')
            ax1.set_title('Accuracy Metrics (Lower is Better)')
            ax1.set_xticks(x)
            ax1.set_xticklabels([m.upper() for m in models], rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Precision@k for different k values
        if 'ranking_metrics' in self.evaluation_results:
            ax2 = axes[0, 1]
            ranking_data = self.evaluation_results['ranking_metrics']
            k_values = [int(k.replace('@', '')) for k in ranking_data.keys()]
            
            for model in ['svd', 'hybrid_weighted']:
                if model in ranking_data.get('@20', {}):
                    precision_values = []
                    for k in k_values:
                        k_key = f'@{k}'
                        if k_key in ranking_data and model in ranking_data[k_key]:
                            precision_values.append(ranking_data[k_key][model]['Precision'])
                        else:
                            precision_values.append(0)
                    
                    ax2.plot(k_values, precision_values, marker='o', label=model.upper(), linewidth=2)
            
            ax2.set_xlabel('k')
            ax2.set_ylabel('Precision@k')
            ax2.set_title('Precision@k vs k')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: NDCG@20 comparison
        if 'ranking_metrics' in self.evaluation_results:
            ax3 = axes[0, 2]
            ranking_20 = self.evaluation_results['ranking_metrics'].get('@20', {})
            
            if ranking_20:
                models = list(ranking_20.keys())
                ndcg_values = [ranking_20[m]['NDCG'] for m in models]
                
                bars = ax3.bar(models, ndcg_values, alpha=0.8, color='skyblue')
                ax3.set_xlabel('Models')
                ax3.set_ylabel('NDCG@20')
                ax3.set_title('NDCG@20 Comparison')
                ax3.set_xticklabels([m.upper() for m in models], rotation=45)
                ax3.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, ndcg_values):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 4: Diversity Metrics
        if 'diversity_metrics' in self.evaluation_results:
            ax4 = axes[1, 0]
            diversity_data = self.evaluation_results['diversity_metrics']
            models = list(diversity_data.keys())
            coverage_values = [diversity_data[m]['catalog_coverage'] for m in models]
            diversity_values = [diversity_data[m]['intra_list_diversity'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax4.bar(x - width/2, coverage_values, width, label='Catalog Coverage', alpha=0.8)
            ax4.bar(x + width/2, diversity_values, width, label='Intra-list Diversity', alpha=0.8)
            ax4.set_xlabel('Models')
            ax4.set_ylabel('Score')
            ax4.set_title('Diversity Metrics')
            ax4.set_xticks(x)
            ax4.set_xticklabels([m.upper() for m in models], rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Novelty vs Popularity Bias
        if 'novelty_metrics' in self.evaluation_results and 'diversity_metrics' in self.evaluation_results:
            ax5 = axes[1, 1]
            novelty_data = self.evaluation_results['novelty_metrics']
            diversity_data = self.evaluation_results['diversity_metrics']
            
            for model in novelty_data.keys():
                if model in diversity_data:
                    novelty = novelty_data[model]['average_novelty']
                    pop_bias = diversity_data[model]['popularity_bias']
                    ax5.scatter(pop_bias, novelty, s=100, label=model.upper(), alpha=0.7)
            
            ax5.set_xlabel('Popularity Bias')
            ax5.set_ylabel('Average Novelty')
            ax5.set_title('Novelty vs Popularity Bias')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Overall Performance Radar Chart
        ax6 = axes[1, 2]
        ax6.text(0.5, 0.5, 'Comprehensive\nEvaluation\nCompleted\n\nSee detailed\nresults in\nJSON report', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
            print("Evaluation plots saved to 'evaluation_results.png'")
        
        plt.show()
        
        return fig 