# CS116--Hybrid Item Recommendation System

## Overview

This project implements a hybrid recommendation system for item recommendation using:

* Frequency-based recommendation
* ALS (Alternating Least Squares) collaborative filtering
* Learning-to-Rank models

  * LightGBM Ranker
  * XGBoost Ranker
  * CatBoost Ranker
* Trend-aware feature engineering for cold-start items

The pipeline is designed for large-scale transaction data and focuses on ranking relevant items for each customer.

---

## Problem Statement

Given historical transaction data, the system predicts the most relevant items for each customer.

The project combines:

* Collaborative filtering signals
* User-item interaction frequency
* Temporal popularity trends
* Learning-to-rank optimization

This hybrid approach improves recommendation quality compared to using a single recommendation technique.

---

## Main Features

### 1. Data Processing

* Load and merge transaction datasets
* Ground truth preparation
* Sparse matrix construction
* Memory optimization utilities

### 2. Feature Engineering

* Global item popularity
* Trending item scores
* Customer-item interaction frequency
* Temporal statistics
* Cold-start item handling

### 3. Recommendation Models

#### Frequency-Based Recommendation

Uses historical interaction counts between customers and items.

#### ALS Collaborative Filtering

Uses matrix factorization on sparse user-item matrices.

Library:

* `implicit`

#### Learning-to-Rank Models

Three ranking models are trained:

* LightGBM Ranker
* XGBoost Ranker
* CatBoost Ranker

These models rank candidate items for each customer.

### 4. Prediction Pipeline

* Batch prediction
* Resume prediction from checkpoint
* Separate handling for:

  * Users with interaction history
  * Cold-start users

### 5. Evaluation

The system evaluates recommendations using:

* Precision@K

---

## Project Pipeline

```text
Transaction Data
       │
       ▼
Data Cleaning & Merge
       │
       ▼
Feature Engineering
       │
       ├── Frequency Features
       ├── Trend Features
       └── ALS Embeddings
       │
       ▼
Candidate Generation
       │
       ▼
Learning-to-Rank Training
       │
       ├── LightGBM
       ├── XGBoost
       └── CatBoost
       │
       ▼
Prediction
       │
       ▼
Evaluation (Precision@K)
```

---

## Technologies Used

### Programming Language

* Python 3.x

### Libraries

* pandas
* polars
* numpy
* scipy
* scikit-learn
* implicit
* lightgbm
* xgboost
* catboost
* tqdm

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

Install dependencies:

```bash
pip install -q implicit lightgbm xgboost catboost
pip install pandas polars numpy scipy scikit-learn tqdm
```

---

## Dataset Structure

Expected dataset structure:

```text
/kaggle/input/
│
├── datasetcs116/
│   └── transaction_2024
│
└── 2025gt/
    └── final_groundtruth.pkl
```

Main components:

* Transaction history
* Customer IDs
* Item IDs
* Ground truth recommendation labels

---

## Running the Notebook

Open the notebook:

```bash
jupyter notebook item-recommendation.ipynb
```

Or run in Kaggle/Google Colab.

Execute cells sequentially:

1. Install dependencies
2. Load data
3. Feature engineering
4. Build ALS model
5. Generate training data
6. Train ranking models
7. Predict recommendations
8. Evaluate Precision@K
9. Save submission/checkpoints

---

## Model Training

### LightGBM Ranker

Gradient boosting ranking model optimized for ranking tasks.

### XGBoost Ranker

Tree-based ranking model with strong performance on structured data.

### CatBoost Ranker

Handles categorical patterns effectively and improves ranking robustness.

---

## Evaluation Metric

### Precision@K

Precision@K measures how many recommended items are relevant among the top-K predictions.

Formula:

```text
Precision@K = Relevant Recommended Items / K
```

---

## Checkpoint System

The notebook supports checkpoint saving for long prediction runs.

Saved files:

```text
prediction_checkpoint_410k.pkl
prediction_checkpoint_no_hist.pkl
```

This allows resuming prediction without restarting the entire pipeline.

---

## Output Files

Generated outputs may include:

```text
submission.csv
prediction_checkpoint_410k.pkl
prediction_checkpoint_no_hist.pkl
```

---

## Future Improvements

Potential extensions:

* Deep learning recommendation models
* Transformer-based sequential recommendation
* Session-based recommendation
* Graph Neural Networks (GNN)
* Real-time recommendation serving
* Feature store integration
* Hyperparameter optimization

---

## Example Use Cases

* E-commerce recommendation
* Retail transaction analysis
* Personalized product ranking
* Customer behavior modeling

---

## Repository Structure

```text
.
├── item-recommendation.ipynb
├── README.md
├── submission.csv
├── prediction_checkpoint_410k.pkl
└── prediction_checkpoint_no_hist.pkl
```

---

## Notes

* The notebook is optimized for large transaction datasets.
* Sparse matrices are used for memory efficiency.
* Batch prediction is implemented to reduce RAM usage.
* Resume prediction is supported for long-running inference.

---

## Author

Developed for recommendation system experimentation and ranking-based item recommendation research.
