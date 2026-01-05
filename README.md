# Book Recommendation System - AI Module

**Course:** AI Module Coursework 2  
**Student:** [Your Name]  
**Submission Date:** January 7, 2025

## 📋 Project Overview

This project implements a book recommendation system using three machine learning algorithms:
1. **Naive Bayes** - Probabilistic classifier
2. **K-Nearest Neighbors (KNN)** - Distance-based classifier
3. **Logistic Regression** - Linear classification model

The system is designed with modularity in mind, making it easy to integrate with the Final Year Project (FYP) that uses Spring Boot backend and React frontend.

## 🏗️ Project Structure

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

## 🚀 Setup Instructions

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

## 📊 Running the Project

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

## 🔌 API Endpoints

### GET `/`
Returns API information and available endpoints.

### GET `/health`
Health check endpoint showing model status.

### GET `/models`
Lists all available models and their loading status.

### POST `/recommend`
Get book recommendations.

**Request:**
```json
{
  "text": "fantasy adventure magic",
  "model": "naive_bayes",
  "top_n": 5
}
```

**Response:**
```json
{
  "model_used": "naive_bayes",
  "recommendations": [
    {
      "book_index": 42,
      "confidence_score": 0.85
    }
  ],
  "count": 5
}
```

### POST `/predict`
Predict book category.

**Request:**
```json
{
  "text": "A story about wizards and magic",
  "model": "logistic_regression"
}
```

**Response:**
```json
{
  "model_used": "logistic_regression",
  "prediction": 2,
  "confidence": 0.87,
  "all_probabilities": [0.05, 0.08, 0.87]
}
```

### POST `/compare`
Compare predictions from all models.

**Request:**
```json
{
  "text": "science fiction space exploration",
  "top_n": 3
}
```

## 🧪 Testing the API

### Using cURL
```bash
# Get recommendations
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "fantasy adventure", "model": "naive_bayes", "top_n": 5}'

# Check health
curl http://localhost:5000/health
```

### Using Python
```python
import requests

# Get recommendations
response = requests.post(
    'http://localhost:5000/recommend',
    json={
        'text': 'fantasy adventure magic',
        'model': 'knn',
        'top_n': 5
    }
)
print(response.json())
```

## 📈 Model Performance

After training, compare the three algorithms:
- **Accuracy**: Overall correctness
- **Precision**: Quality of positive predictions
- **Recall**: Coverage of positive cases
- **F1-Score**: Harmonic mean of precision and recall

Results will be saved in `trained_models/model_comparison.csv` and visualized in `trained_models/comparison.png`.

## 🔄 Integration with FYP

### For Spring Boot Integration

1. **Keep Flask API Running**
   ```bash
   python api/app.py
   ```

2. **Spring Boot Controller Example**
   ```java
   @RestController
   @RequestMapping("/api/books")
   public class BookController {
       
       private final RestTemplate restTemplate = new RestTemplate();
       
       @PostMapping("/recommend")
       public ResponseEntity<?> getRecommendations(@RequestBody BookRequest request) {
           String flaskUrl = "http://localhost:5000/recommend";
           
           Map<String, Object> requestBody = new HashMap<>();
           requestBody.put("text", request.getText());
           requestBody.put("model", "naive_bayes");
           requestBody.put("top_n", 5);
           
           return restTemplate.postForEntity(flaskUrl, requestBody, Map.class);
       }
   }
   ```

3. **Enable CORS** (already configured in Flask)

### Project Architecture for FYP
```
┌─────────────────┐
│  React Frontend │
│   (Port 3000)   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│ Spring Boot API │
│   (Port 8080)   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   Flask AI API  │
│   (Port 5000)   │
└─────────────────┘
```

## 📝 Submission Checklist

- [x] Source code in organized structure
- [x] Jupyter notebooks with experimentation
- [x] Three trained models (Naive Bayes, KNN, Logistic Regression)
- [x] Flask API for model serving
- [x] Requirements.txt with all dependencies
- [x] README with setup instructions
- [x] Model comparison and evaluation
- [x] Documentation and comments

## 🔧 Troubleshooting

### Issue: Models not loading in Flask
**Solution:** Make sure you've trained and saved the models first by running the Jupyter notebook.

### Issue: Import errors
**Solution:** Make sure you're in the correct conda environment:
```bash
conda activate book-rec-ai
```

### Issue: Dataset not found
**Solution:** Place your dataset in `data/raw/` and update the path in the notebook.

### Issue: CORS errors from React
**Solution:** The Flask API is already configured with CORS. Ensure React is running on port 3000.

## 📚 Dependencies

Main libraries used:
- `scikit-learn` - Machine learning algorithms
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `flask` - REST API framework
- `flask-cors` - CORS support
- `joblib` - Model serialization
- `matplotlib` & `seaborn` - Visualization

Full list in `requirements.txt`

## 👨‍💻 Development

### Running Tests
```bash
pytest tests/
```

### Adding New Features
1. Create feature branch
2. Implement changes
3. Test thoroughly
4. Commit and push

### VS Code Setup
Recommended extensions:
- Python
- Jupyter
- Pylance
- GitLens

## 📧 Contact

For questions or issues:
- Email: [your.email@example.com]
- GitHub Issues: [repo-url/issues]

## 📄 License

This project is for academic purposes as part of AI Module Coursework.

---

**Note:** This project is designed to be easily integrated into your FYP. Keep the Flask API as a microservice and connect it to your Spring Boot backend.