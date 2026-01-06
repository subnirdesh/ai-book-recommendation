import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model directory
MODEL_DIR = BASE_DIR / 'trained_models'

# Creating directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# GOOGLE API KEY CONFIG
GOOGLE_BOOKS_API_KEY = os.environ.get('GOOGLE_BOOKS_API_KEY')
if not GOOGLE_BOOKS_API_KEY:
    raise ValueError("Google Books API key not set. Please set GOOGLE_BOOKS_API_KEY environment variable.")


# Model configurations
MODEL_CONFIG = {
    'naive_bayes': {
        'name': 'Naive Bayes Classifier',
        'model_path': MODEL_DIR / 'naive_bayes.pkl'
    },
    'knn': {
        'name': 'K-Nearest Neighbors',
        'model_path': MODEL_DIR / 'knn.pkl',
        'n_neighbors': 5
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'model_path': MODEL_DIR / 'logistic_regression.pkl',
        'max_iter': 1000
    }
}

# Flask configurations
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = True
    
    # CORS settings for FYP integration
    CORS_ORIGINS = [
        'http://localhost:5173',  # React frontend
        'http://localhost:8080',  # Spring Boot backend
    ]

# Feature engineering settings
FEATURE_CONFIG = {
    'text_features': ['title', 'authors', 'categories'],
    'numerical_features': ['average_rating', 'ratings_count'],
    'max_features': 5000  # for TF-IDF
}

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42