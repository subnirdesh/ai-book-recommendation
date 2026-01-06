from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
import numpy as np
import config

class NaiveBayesRecommender:

    """
    Naive Bayes model for book recommendation
    """

    def __init__(self, model_type='multinomial'):
        self.model_type = model_type

        if model_type != 'multinomial':
            raise ValueError("Only MultinomialNB is supported for text data")

        self.model = MultinomialNB()
        self.vectorizer = TfidfVectorizer(
            max_features=config.FEATURE_CONFIG['max_features'],
            ngram_range=config.FEATURE_CONFIG['ngram_range'],
            min_df=config.FEATURE_CONFIG['min_df'],
            max_df=config.FEATURE_CONFIG['max_df'],
            sublinear_tf=True  # Using log scaling
        )

        self.is_trained = False


    def prepare_text_features(self, X, fit=False):
        if fit:
            return self.vectorizer.fit_transform(X)
        return self.vectorizer.transform(X)
    

    def train(self,X_train,y_train):
        """
        Trains the Naive Bayes model
        
        Args:
            X_train: Training features (text or numerical)
            y_train: Training labels
        """

        print(f"Training Naive Bayes model...")

        # Preparing features 
        X_train_processed=self.prepare_text_features(X_train,fit=True)

        # Training the model 
        self.model.fit(X_train_processed,y_train)
        self.is_trained=True

        print("Training completed!")
        return self
    

    def predict(self,X):
        """
        Make predictions
        
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
    

    def predict_proba(self,X):
        """
        Gets prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            Probability estimates
        """

        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_processed=self.prepare_text_features(X)
        probabilities=self.model.predict_proba(X_processed)

        return probabilities
    

    def evaluate(self,X_test,y_test):
        """
        Evaluates model performance
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """

        predictions=self.predict(X_test)

        accuracy=accuracy_score(y_test,predictions)
        precision,recall,f1,_=precision_recall_fscore_support(y_test, predictions, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        print(f"\n{self.model_type.upper()} Naive Bayes Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        return metrics
    

    def get_top_recommendations(self,X,top_n=5):
        """
        Gets top N recommendations based on prediction probabilities
        
        Args:
            X: User features
            top_n: Number of recommendations
            
        Returns:
            Top N book indices with probabilities
        """

        probabilities=self.predict_proba(X)
        top_indices=np.argsort(probabilities, axis=1)[:, -top_n:][:, ::-1]
        top_probs = np.sort(probabilities, axis=1)[:, -top_n:][:, ::-1]

        return top_indices,top_probs
    

    def save_model(self, filename=None):
        """Saves trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if filename is None:
            filename = config.MODEL_CONFIG['naive_bayes']['model_path']
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename=None):
        """Loads trained model from disk"""
        if filename is None:
            filename = config.MODEL_CONFIG['naive_bayes']['model_path']
        
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filename}")
        return self
    

    



