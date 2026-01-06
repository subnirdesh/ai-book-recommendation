from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from pathlib import Path

# Import models
from src.models.naive_bayes_model import NaiveBayesRecommender
from src.models.knn_model import KNNRecommender
from src.models.logistic_regression_model import LogisticRegressionRecommender

app = Flask(__name__)
CORS(app)

# Global variables for models
models = {}
category_mapping = {}
reverse_mapping = {}

def load_all_models():

    """Load all trained models on startup"""

    global models, category_mapping, reverse_mapping

    print("Loading models...")

    # Loading category mapping
    try:
        with open('data/processed/category_mapping.json', 'r') as f:
            category_mapping = json.load(f)
            reverse_mapping = {v: k for k, v in category_mapping.items()}
        print(f"Loaded {len(category_mapping)} categories")
    except:
        print("Category mapping not found")

    # Load models
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
            print(f"{key} loaded")
        except Exception as e:
            print(f"{key} failed: {e}")
    
    print(f" {len(models)} models ready!")


@app.route("/")
def home():

    return render_template('index.html', 
                         models=list(models.keys()),
                         categories=list(category_mapping.keys()))


@app.route("/recommend", methods=['POST'])
def recommend():
    
    """Get recommendations"""

    try:
        data = request.json
        user_input = data.get('text', '')
        model_choice = data.get('model', 'naive_bayes')
        top_n = int(data.get('top_n', 5))
        
        if not user_input:
            return jsonify({'error': 'Please provide input text'}), 400
        
        if model_choice not in models:
            return jsonify({'error': f'Model {model_choice} not available'}), 400
        
        # Getting recommendations
        model = models[model_choice]
        predictions = model.predict_proba([user_input])[0]
        top_indices = predictions.argsort()[-top_n:][::-1]
        top_scores = predictions[top_indices]

        recommendations = []
        for idx, score in zip(top_indices, top_scores):
            category = reverse_mapping.get(idx, f"Category {idx}")
            recommendations.append({
                'category': category,
                'confidence': float(score),
                'confidence_pct': float(score * 100)
            })

        return jsonify({
            'success': True,
            'query': user_input,
            'model': model_choice,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/compare',methods=['POST'])
def compare_models():

    """Compare all models"""

    try:
        data = request.json
        user_input = data.get('text', '')
        top_n = int(data.get('top_n', 3))

        if not user_input:
            return jsonify({'error': 'Please provide input text'}), 400
        
        results = {}

        for model_key, model in models.items():
            predictions = model.predict_proba([user_input])[0]
            top_indices = predictions.argsort()[-top_n:][::-1]
            top_scores = predictions[top_indices]

            recommendations = []
            for idx, score in zip(top_indices, top_scores):
                category = reverse_mapping.get(idx, f"Category {idx}")
                recommendations.append({
                    'category': category,
                    'confidence': float(score),
                    'confidence_pct': float(score * 100)
                })
            results[model_key] = recommendations

        return jsonify({
            'success': True,
            'query': user_input,
            'results': results
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
    print("Starting Book Recommendation Web App")
    print("="*70)
    print("\n✓ Server running at: http://localhost:5001")
    print("✓ Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

        

        


    






    

