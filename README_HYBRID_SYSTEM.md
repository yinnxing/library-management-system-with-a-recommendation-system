# Comprehensive Hybrid Recommender System for Academic Research

A state-of-the-art hybrid recommender system implementation with extensive evaluation metrics designed for academic research and publication.

## 🎯 Overview

This project implements a comprehensive hybrid recommender system that combines multiple recommendation approaches and provides extensive evaluation metrics suitable for academic research. The system is designed to generate all necessary indexes and metrics for writing academic reports on recommender systems.

## 🚀 Features

### Recommendation Algorithms
- **Content-Based Filtering**
  - TF-IDF vectorization with cosine similarity
  - SVD-based dimensionality reduction
  - Non-negative Matrix Factorization (NMF)
  
- **Collaborative Filtering**
  - Matrix Factorization (SVD)
  - Non-negative Matrix Factorization (NMF)
  - User-based and Item-based KNN
  - Baseline algorithms
  
- **Popularity-Based Filtering**
  - Weighted rating approach
  - Bayesian average implementation
  
- **Hybrid Approaches**
  - Weighted combination
  - Rank fusion
  - Cascade strategy

### Evaluation Metrics

#### Accuracy Metrics
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)

#### Ranking Quality Metrics @k (k=5,10,20)
- **Precision@k**
- **Recall@k**
- **F1-Score@k**
- **NDCG@k** (Normalized Discounted Cumulative Gain)
- **MAP@k** (Mean Average Precision)

#### Diversity and Coverage Metrics
- **Catalog Coverage**
- **Intra-list Diversity**
- **Popularity Bias**

#### Novelty Metrics
- **Average Novelty**
- **Long-tail Coverage**

## 📁 Project Structure

```
hybrid_recommender_system/
├── hybrid_recommender_system.py    # Main recommender system implementation
├── evaluation_framework.py         # Comprehensive evaluation metrics
├── main_recommender_pipeline.py    # Complete pipeline execution
├── requirements.txt                # Python dependencies
├── README_HYBRID_SYSTEM.md        # This file
└── data/
    ├── Books.csv                   # Book metadata
    └── Recommender/
        ├── Ratings.csv             # User ratings
        └── Users.csv               # User metadata (optional)
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pandas
- numpy
- scikit-learn
- scikit-surprise
- matplotlib
- seaborn

### Setup

1. **Clone or download the project files**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your dataset**
   - Place `Books.csv` in the `data/` directory
   - Place `Ratings.csv` and `Users.csv` in the `Recommender/` directory
   - Ensure the CSV files follow the Book-Crossing dataset format:
     - Books.csv: ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, Image-URL-L
     - Ratings.csv: User-ID, ISBN, Book-Rating
     - Users.csv: User-ID, Location, Age (optional)

## 🚀 Quick Start

### Running the Complete Pipeline

```bash
python main_recommender_pipeline.py
```

This will execute the complete workflow:
1. Data loading and preprocessing
2. Model building (all algorithms)
3. Recommendation demonstration
4. Comprehensive evaluation
5. Academic report generation
6. Model persistence

### Using Individual Components

```python
from hybrid_recommender_system import HybridRecommenderSystem
from evaluation_framework import RecommenderEvaluator

# Initialize and load data
recommender = HybridRecommenderSystem(random_state=42)
recommender.load_data(
    books_path='data/Books.csv',
    ratings_path='Recommender/Ratings.csv'
)

# Build models
recommender.build_content_based_models()
recommender.build_collaborative_models()
recommender.build_popularity_model()

# Get recommendations
content_recs = recommender.get_content_recommendations('0195153448', 'tfidf', 10)
collab_recs = recommender.get_collaborative_recommendations(276725, 'svd', 10)
hybrid_recs = recommender.get_hybrid_recommendations(276725, '0195153448', 10, 'weighted')

# Evaluate system
evaluator = RecommenderEvaluator(recommender)
results = evaluator.run_comprehensive_evaluation()
evaluator.generate_evaluation_report('evaluation_report.json')
```

## 📊 Academic Evaluation Output

The system generates comprehensive evaluation results suitable for academic papers:

### 1. Accuracy Metrics Table
```
Model           RMSE     MAE      N_Pred
SVD             2.1234   1.6789   12,345
NMF             2.2456   1.7234   12,345
KNN_ITEM        2.3567   1.8456   12,345
```

### 2. Ranking Quality @20 Table
```
Model           Prec@20  Rec@20   F1@20    NDCG@20  MAP@20
SVD             0.0234   0.0456   0.0312   0.0567   0.0234
HYBRID_WEIGHTED 0.0267   0.0523   0.0356   0.0634   0.0278
```

### 3. Diversity Metrics Table
```
Model           Coverage Diversity Pop_Bias
SVD             0.1234   0.7890    4.5678
HYBRID_WEIGHTED 0.1456   0.8234    4.2345
```

### 4. Novelty Metrics Table
```
Model           Avg_Novelty  LongTail_%
SVD             8.9012       0.1234
HYBRID_WEIGHTED 9.2345       0.1456
```

## 📈 Visualization

The system automatically generates publication-ready plots:
- Accuracy metrics comparison
- Precision@k curves
- NDCG@20 comparison
- Diversity metrics
- Novelty vs. Popularity bias scatter plot

## 🔧 Customization

### Adjusting Hybrid Weights
```python
recommender.hybrid_weights = {
    'content_based': 0.4,
    'collaborative': 0.4,
    'popularity': 0.2
}
```

### Modifying Evaluation Parameters
```python
evaluator.evaluate_ranking_metrics(
    k_values=[5, 10, 15, 20, 25],
    models_to_evaluate=['svd', 'nmf', 'hybrid_weighted']
)
```

### Custom Filtering Thresholds
```python
recommender._apply_filtering_thresholds(
    min_book_ratings=15,
    min_user_ratings=10
)
```

## 📝 Academic Usage

This system is designed to support academic research. The evaluation framework provides:

1. **Standardized Metrics**: All metrics follow academic standards and best practices
2. **Statistical Significance**: Proper train/test splits and evaluation protocols
3. **Reproducibility**: Fixed random seeds and deterministic algorithms
4. **Comprehensive Coverage**: Multiple evaluation dimensions (accuracy, ranking, diversity, novelty)
5. **Publication-Ready Output**: Tables and plots formatted for academic papers

### Citing This Work

If you use this system in your research, please cite:

```bibtex
@software{hybrid_recommender_system,
  title={Comprehensive Hybrid Recommender System for Academic Research},
  author={AI Assistant},
  year={2024},
  url={https://github.com/your-repo/hybrid-recommender-system}
}
```

## 🔍 Dataset Requirements

The system works with the Book-Crossing dataset format but can be adapted for other domains:

- **Minimum Requirements**:
  - 1,000+ users
  - 1,000+ items
  - 10,000+ ratings
  - Explicit ratings (1-10 scale)

- **Recommended**:
  - 10,000+ users
  - 10,000+ items
  - 100,000+ ratings
  - Rich item metadata

## 🚨 Performance Considerations

- **Memory Usage**: Content-based models require significant memory for large datasets
- **Computation Time**: Full evaluation can take 30-60 minutes for large datasets
- **Scalability**: System is optimized for research datasets (up to 1M ratings)

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error during TF-IDF**
   - Reduce `max_features` in TF-IDF vectorizer
   - Use smaller book subset for content models

2. **Slow Evaluation**
   - Reduce number of test users in evaluation
   - Use smaller k values for ranking metrics

3. **Missing Data Files**
   - Ensure CSV files are in correct directories
   - Check file encoding (use UTF-8 or latin-1)

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Please focus on:
- Additional evaluation metrics
- New hybrid combination strategies
- Performance optimizations
- Documentation improvements

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 📞 Support

For academic research support or questions:
- Create an issue in the repository
- Provide detailed error messages and dataset characteristics
- Include system specifications for performance issues

---

**Note**: This system is designed for academic research and educational purposes. For production use, additional optimizations and scalability considerations may be required. 