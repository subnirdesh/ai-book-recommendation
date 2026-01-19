from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from pathlib import Path
import numpy as np

# Import models
from src.models.naive_bayes_model import NaiveBayesRecommender
from src.models.knn_model import KNNRecommender
from src.models.logistic_regression_model import LogisticRegressionRecommender
from src.mood_mapper import MoodMapper
from src.hybrid_model import HybridEnsemble
from src.book_retrieval import BookRetrieval

app = Flask(__name__)
CORS(app)

# Global variables for models
models = {}
category_mapping = {}
reverse_mapping = {}
mood_mapper = MoodMapper()
hybrid_model = None
book_retrieval = None

def load_all_models():
    """Load all trained models on startup"""
    global models, category_mapping, reverse_mapping, hybrid_model, book_retrieval
    
    print("Loading models...")
    
    # Load category mapping
    try:
        with open('data/processed/category_mapping.json', 'r') as f:
            category_mapping = json.load(f)
            reverse_mapping = {v: k for k, v in category_mapping.items()}
        print(f"✓ Loaded {len(category_mapping)} categories")
    except:
        print("⚠️ Category mapping not found")
    
    # Load individual models
    model_info = [
        ('naive_bayes', NaiveBayesRecommender),
        ('knn', KNNRecommender),
        ('logistic_regression', LogisticRegressionRecommender)
    ]
    
    for key, ModelClass in model_info:
        try:
            model = ModelClass()
            model.load_model()
            models[key] = model
            print(f"✓ {key} loaded")
        except Exception as e:
            print(f"✗ {key} failed: {e}")
    
    # Create hybrid ensemble
    if len(models) >= 2:
        hybrid_model = HybridEnsemble(models, method='weighted_average')
        models['hybrid'] = hybrid_model
        print("✓ Hybrid ensemble created")
    
    # Initialize book retrieval
    book_retrieval = BookRetrieval()
    
    print(f"✓ {len(models)} models ready!")
    print("✓ Mood-aware system initialized")
    print("✓ Book retrieval system ready")

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', 
                         categories=list(category_mapping.keys()),
                         models_available=list(models.keys()))

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get mood-aware book recommendations"""
    try:
        data = request.json
        user_input = data.get('text', '')
        model_choice = data.get('model', 'hybrid')  # Default to hybrid
        top_n = int(data.get('top_n', 10))
        
        if not user_input:
            return jsonify({'error': 'Please provide input text'}), 400
        
        if model_choice not in models:
            return jsonify({'error': f'Model {model_choice} not available'}), 400
        
        # Process query through mood mapper
        query_analysis = mood_mapper.process_query(user_input)
        enhanced_query = query_analysis['enhanced_query']
        
        # Get predictions using selected model
        model = models[model_choice]
        predictions = model.predict_proba([enhanced_query])[0]
        
        # Adjust predictions based on mood
        if query_analysis['avoid_categories']:
            for cat in query_analysis['avoid_categories']:
                if cat in category_mapping:
                    cat_idx = category_mapping[cat]
                    predictions[cat_idx] *= 0.1
        
        if query_analysis['recommended_categories']:
            for cat in query_analysis['recommended_categories']:
                if cat in category_mapping:
                    cat_idx = category_mapping[cat]
                    predictions[cat_idx] *= 1.5
        
        # Renormalize
        predictions = predictions / predictions.sum()
        
        # Get top categories
        top_indices = predictions.argsort()[-5:][::-1]
        top_categories = [
            (reverse_mapping[idx], float(predictions[idx]))
            for idx in top_indices
        ]
        
        # Get actual books
        books = book_retrieval.get_diverse_recommendations(
            top_categories, 
            total_books=top_n
        )
        
        # Also show category predictions for transparency
        category_predictions = []
        for cat, score in top_categories[:5]:
            category_predictions.append({
                'category': cat,
                'confidence': float(score),
                'confidence_pct': float(score * 100)
            })
        
        return jsonify({
            'success': True,
            'query': user_input,
            'use_case': query_analysis['use_case'],
            'detected_moods': query_analysis['detected_moods'],
            'explanation': mood_mapper.explain_recommendation(query_analysis),
            'model': model_choice,
            'category_predictions': category_predictions,
            'book_recommendations': books,
            'total_books': len(books)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'categories': len(category_mapping)
    })

if __name__ == '__main__':
    load_all_models()
    print("\n" + "="*70)
    print("🚀 Starting Book Recommendation Web App")
    print("="*70)
    print("\n✓ Server running at: http://localhost:5001")
    print("✓ Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)