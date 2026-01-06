from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import numpy as np
import config

class KNNRecommender:
    """
    K-Nearest Neighbors model for book recommendation
    """

    def __init__(self,n_neighbors=5,metric='cosine',algorithm='brute'):

        self.n_neighbors=n_neighbors
        self.metric=metric
        self.model=KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm=algorithm,
            n_jobs=-1  # Using all available CPU cores
        )

        self.vectorizer=TfidfVectorizer(max_features=config.FEATURE_CONFIG['max_features'])
        self.is_trained=False

    
    def prepare_text_features(self,X):
        """Converts text to TF-IDF features"""

        if not self.is_trained:
            X_vectorized=self.vectorizer.fit_transform(X)
        
        else:
            X_vectorized=self.vectorizer.transform(X)

        return X_vectorized
    

    def train(self, X_train, y_train):
        """
        Trains the KNN model
        """

        print(f"Training KNN model with {self.n_neighbors} neighbors...")

        # Preparing features 
        X_train_processed=self.prepare_text_features(X_train)

        # Training Model
        self.model.fit(X_train_processed,y_train)
        self.is_trained=True

        print("Training completed!")
        
        return self
    


    def predict(self, X):
        """
        Makes predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_processed = self.prepare_text_features(X)
        predictions = self.model.predict(X_processed)
        return predictions
    

    def predict_proba(self, X):
        """
        Gets prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            Probability estimates
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_processed = self.prepare_text_features(X)
        probabilities = self.model.predict_proba(X_processed)
        return probabilities
    
    def get_nearest_neighbors(self, X, n_neighbors=None):
        """
        Gets nearest neighbors for given samples
        
        Args:
            X: Query samples
            n_neighbors: Number of neighbors (if None, uses model's n_neighbors)
            
        Returns:
            Distances and indices of nearest neighbors
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        X_processed = self.prepare_text_features(X)
        distances, indices = self.model.kneighbors(X_processed, n_neighbors=n_neighbors)
        return distances, indices
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_neighbors': self.n_neighbors
        }
        
        print(f"\nKNN Evaluation (k={self.n_neighbors}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return metrics
    

    def get_top_recommendations(self, X, top_n=5):
        """
        Get top N recommendations based on nearest neighbors
        
        Args:
            X: User features
            top_n: Number of recommendations
            
        Returns:
            Top N book indices with distances
        """
        distances, indices = self.get_nearest_neighbors(X, n_neighbors=top_n)
        return indices, distances
    

    def tune_k(self, X_train, y_train, X_val, y_val, k_range=range(1, 21)):
        """
        Finds optimal k value
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            k_range: Range of k values to test
            
        Returns:
            Best k value and scores
        """
        scores = []
        
        X_train_processed = self.prepare_text_features(X_train)
        X_val_processed = self.vectorizer.transform(X_val)

        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k, metric=self.metric, n_jobs=-1)
            model.fit(X_train_processed, y_train)
            score = model.score(X_val_processed, y_val)
            scores.append(score)
            print(f"k={k}, accuracy={score:.4f}")
        
        best_k = list(k_range)[np.argmax(scores)]
        print(f"\nBest k: {best_k} with accuracy: {max(scores):.4f}")
        
        return best_k, scores
    
    def save_model(self, filename=None):
        """Saves trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if filename is None:
            filename = config.MODEL_CONFIG['knn']['model_path']
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")


    def load_model(self, filename=None):
        """Loads trained model from disk"""
        if filename is None:
            filename = config.MODEL_CONFIG['knn']['model_path']
        
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.n_neighbors = model_data['n_neighbors']
        self.metric = model_data['metric']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filename}")
        return self
    
    



        