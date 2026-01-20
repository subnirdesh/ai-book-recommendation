# Book Recommendation System - AI Module

##  Project Overview

This project implements a book recommendation system using three machine learning algorithms:
1. **Naive Bayes** - Probabilistic classifier
2. **K-Nearest Neighbors (KNN)** - Distance-based classifier
3. **Logistic Regression** - Linear classification model

The system is designed with modularity in mind, making it easy to integrate with the Final Year Project (FYP) that uses Spring Boot backend and React frontend.

## Project Structure

```
book-recommendation-ai/
├── notebooks/              # Jupyter notebooks for experimentation
│   └── main_training.ipynb
├── src/                   # Core application code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── utils.py
│   └── models/
│       ├── naive_bayes_model.py
│       ├── knn_model.py
│       └── logistic_regression_model.py
├── data/                  # Data storage
│   ├── raw/              # Original datasets
│   └── processed/        # Processed datasets
├── trained_models/        # Saved model files (.pkl)
├── api/                   # Flask REST API
│   ├── app.py
│   └── routes.py
├── tests/                 # Unit tests
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Conda (Miniconda or Anaconda)
- Git
- VS Code (recommended)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/book-recommendation-ai.git
cd book-recommendation-ai
```

### 2. Create Conda Environment
```bash
# Create environment
conda create -n book-rec-ai python=3.10 -y

# Activate environment
conda activate book-rec-ai

# Install Jupyter
conda install jupyter -y
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Your Dataset
Place your book dataset in the `data/raw/` directory. The dataset should include:
- Book titles
- Authors
- Categories/genres
- Ratings (optional)
- Descriptions (optional)

Example CSV format:
```csv
title,authors,categories,average_rating,description
"Harry Potter","J.K. Rowling","Fantasy",4.5,"A young wizard's journey"
```

## Running the Project

### Option 1: Using Jupyter Notebook (Recommended for Development)
```bash
# Start Jupyter
jupyter notebook

# Open notebooks/main_training.ipynb
# Follow the cells to train and evaluate models
```

### Option 2: Using Python Scripts
```bash
# Train all models
python -m notebooks.main_training

# Or train individual models
python -m src.models.naive_bayes_model
```

### Option 3: Using Flask API
```bash
# Start the Flask server
python api/app.py

# API will be available at http://localhost:5000
```

## Dependencies

Main libraries used:
- `scikit-learn` - Machine learning algorithms
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `flask` - REST API framework
- `flask-cors` - CORS support
- `joblib` - Model serialization
- `matplotlib` & `seaborn` - Visualization

Full list in `requirements.txt`


##  License

This project is for academic purposes as part of AI Module Coursework.

---

