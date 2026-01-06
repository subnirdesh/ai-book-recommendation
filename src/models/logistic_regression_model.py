from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import numpy as np
import config

class LogisticRegressionRecommender:
    """
    Logistic Regression model for book recommendation
    """

    def __init__(self, max_iter=1000, solver='lbfgs', multi_class='multinomial'):
        
        self.max_iter = max_iter
        self.solver = solver
        self.multi_class = multi_class
        self.model = LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            multi_class=multi_class,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,  # Using all CPU cores
            C=1.0,  # Regularization strength
            class_weight='balanced'  # Handling class imbalance
        )
        self.vectorizer = TfidfVectorizer(
            max_features=config.FEATURE_CONFIG['max_features'],
            ngram_range=config.FEATURE_CONFIG['ngram_range'],
            min_df=config.FEATURE_CONFIG['min_df'],
            max_df=config.FEATURE_CONFIG['max_df'],
            sublinear_tf=True
        )
        self.is_trained = False


    def prepare_text_features(self, X):
        """Converting text to TF-IDF features"""
        if not self.is_trained:
            X_vectorized = self.vectorizer.fit_transform(X)
        else:
            X_vectorized = self.vectorizer.transform(X)
        return X_vectorized
    
    def train(self, X_train, y_train):
        """
        Trains the Logistic Regression model
        
        Args:
            X_train: Training features (text or numerical)
            y_train: Training labels
        """
        print(f"Training Logistic Regression model...")
        
        # Preparing features
        X_train_processed = self.prepare_text_features(X_train)
        
        # Training model
        self.model.fit(X_train_processed, y_train)
        self.is_trained = True
        
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
    
    def get_feature_importance(self, top_n=20):
        """
        Gets most important features (words) for each class
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of class -> top features
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        importance_dict = {}
        
        for idx, class_coef in enumerate(self.model.coef_):
            top_indices = np.argsort(class_coef)[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            importance_dict[f'class_{idx}'] = top_features
        
        return importance_dict
    

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
            'f1_score': f1
        }
        
        print(f"\nLogistic Regression Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return metrics
    
    def get_top_recommendations(self, X, top_n=5):
        """
        Gets top N recommendations based on prediction probabilities
        
        Args:
            X: User features
            top_n: Number of recommendations
            
        Returns:
            Top N book indices with probabilities
        """
        probabilities = self.predict_proba(X)
        top_indices = np.argsort(probabilities, axis=1)[:, -top_n:][:, ::-1]
        top_probs = np.sort(probabilities, axis=1)[:, -top_n:][:, ::-1]
        
        return top_indices, top_probs
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, 
                           C_range=[0.001, 0.01, 0.1, 1, 10, 100]):
        """
        Tunes regularization parameter C
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            C_range: Range of C values to test
            
        Returns:
            Best C value and scores
        """
        scores = []
        
        X_train_processed = self.prepare_text_features(X_train)
        X_val_processed = self.vectorizer.transform(X_val)
        
        for C in C_range:
            model = LogisticRegression(
                C=C,
                max_iter=self.max_iter,
                solver=self.solver,
                multi_class=self.multi_class,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
            model.fit(X_train_processed, y_train)
            score = model.score(X_val_processed, y_val)
            scores.append(score)
            print(f"C={C}, accuracy={score:.4f}")
        
        best_C = C_range[np.argmax(scores)]
        print(f"\nBest C: {best_C} with accuracy: {max(scores):.4f}")
        
        return best_C, scores
    
    def save_model(self, filename=None):
        """Saves trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if filename is None:
            filename = config.MODEL_CONFIG['logistic_regression']['model_path']
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'max_iter': self.max_iter,
            'solver': self.solver,
            'multi_class': self.multi_class,
            'is_trained': self.is_trained
        }


    def load_model(self, filename=None):
        """Loads trained model from disk"""
        if filename is None:
            filename = config.MODEL_CONFIG['logistic_regression']['model_path']
        
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.max_iter = model_data['max_iter']
        self.solver = model_data['solver']
        self.multi_class = model_data['multi_class']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filename}")
        return self
    

    