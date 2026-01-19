"""
Hybrid Ensemble Model
Combining Naive Bayes, KNN, and Logistic Regression
"""

import numpy as np
from sklearn.metrics import accuracy_score

class HybridEnsemble:
    """
    Creating a hybrid ensemble that is combining multiple classifiers
    Supporting weighted averaging, voting, and max-confidence strategies
    """

    def __init__(self, models, weights=None, method='weighted_average'):
        self.models = models
        self.method = method
        self.n_classes = None

        # Setting default weights
        if weights is None:
            weights = {
                'naive_bayes': 0.30,
                'knn': 0.30,
                'logistic_regression': 0.40
            }

        # Normalizing weights
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

        # Checking models
        self._check_models()

    def _check_models(self):
        """Checking model availability and predict_proba support"""
        print("\nChecking ensemble models...")

        for name, model in self.models.items():
            if model is None:
                print(f"  {name}: not loaded")
            elif hasattr(model, "predict_proba"):
                print(f"  {name}: usable (supports predict_proba)")
            else:
                print(f"  {name}: usable (hard prediction only)")

    def _safe_predict_proba(self, model, X):
        """Calling predict_proba safely and returning None when failing"""

        try:
            if hasattr(model, "predict_proba"):
                out = model.predict_proba(X)
                if out is None:
                    return None
                return np.asarray(out)
            else:
                preds = model.predict(X)
                preds = np.asarray(preds)

                # Creating one-hot encodings
                if self.n_classes is None:
                    self.n_classes = int(np.max(preds)) + 1

                probs = np.zeros((len(preds), self.n_classes))
                for i, p in enumerate(preds):
                    probs[i, int(p)] = 1.0
                return probs

        except Exception as e:
            print(f"Warning: {model.__class__.__name__} failing: {e}")
            return None

    def _weighted_average(self, X):
        """Combining model probabilities using weighted averaging"""

        combined = None
        total_weight = 0.0

        for name, model in self.models.items():
            if model is None:
                continue

            probs = self._safe_predict_proba(model, X)
            if probs is None:
                continue

            weight = self.weights.get(name, 0)

            if combined is None:
                combined = probs * weight
                self.n_classes = probs.shape[1]
            else:
                combined = combined + probs * weight

            total_weight += weight

        if combined is None:
            raise ValueError("All models failed to generate probabilities")

        if total_weight > 0:
            combined = combined / total_weight

        return combined

    def _voting(self, X):
        """Combining predictions using majority voting"""

        votes = []

        for name, model in self.models.items():
            if model is None:
                continue

            try:
                preds = model.predict(X)
                votes.append(np.asarray(preds))
            except Exception as e:
                print(f"Warning: {name} failing in voting: {e}")

        if len(votes) == 0:
            raise ValueError("No models produced predictions")

        votes = np.array(votes)  # shape = (n_models, n_samples)

        # Counting votes
        n_models, n_samples = votes.shape
        n_classes = int(np.max(votes)) + 1
        probs = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for v in votes[:, i]:
                probs[i, int(v)] += 1

        return probs / n_models

    def _max_confidence(self, X):
        """Taking the maximum confidence across classifiers"""

        prob_list = []

        for name, model in self.models.items():
            probs = self._safe_predict_proba(model, X)
            if probs is not None:
                prob_list.append(probs)

        if not prob_list:
            raise ValueError("No model produced probability outputs")

        stacked = np.stack(prob_list)  # shape: (n_models, n_samples, n_classes)
        combined = np.max(stacked, axis=0)

        return combined / combined.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        """Dispatching to combination strategy"""

        if self.method == 'weighted_average':
            return self._weighted_average(X)
        elif self.method == 'voting':
            return self._voting(X)
        elif self.method == 'max':
            return self._max_confidence(X)
        else:
            return self._weighted_average(X)

    def predict(self, X):
        """Selecting highest scoring class"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_top_predictions(self, X, top_k=3):
        """Extracting top-k classes"""
        probs = self.predict_proba(X)[0]
        idx = np.argsort(probs)[::-1][:top_k]
        return [(int(i), float(probs[i])) for i in idx]

    def get_model_contributions(self, X):
        """Displaying each model's prediction and weighted impact"""

        output = {}

        for name, model in self.models.items():
            probs = self._safe_predict_proba(model, X)
            if probs is None:
                continue

            class_id = int(np.argmax(probs[0]))
            conf = float(probs[0][class_id])
            w = self.weights.get(name, 0.0)

            output[name] = {
                "top_class": class_id,
                "confidence": conf,
                "weight": w,
                "weighted_contribution": conf * w
            }

        return output
