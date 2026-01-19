"""
Book Recommendation System - CLI Prototype
Updated with hybrid ensemble and actual book recommendations

Usage:
    python book_recommender_cli.py
"""

import sys
import json
from pathlib import Path

# Importing model classes
from src.models.naive_bayes_model import NaiveBayesRecommender
from src.models.knn_model import KNNRecommender
from src.models.logistic_regression_model import LogisticRegressionRecommender
from src.mood_mapper import MoodMapper
from src.hybrid_model import HybridEnsemble
from src.book_retrieval import BookRetrieval


class BookRecommenderCLI:
    """CLI for generating book recommendations using multiple models"""

    def __init__(self):
        self.models = {}
        self.category_mapping = {}
        self.reverse_mapping = {}
        self.mood_mapper = MoodMapper()
        self.hybrid_model = None
        self.book_retrieval = None

    def load_models(self):
        """Loading all trained models and resources"""
        print("=" * 70)
        print("MOOD-AWARE BOOK RECOMMENDATION SYSTEM")
        print("=" * 70)
        print("\nLoading AI models...")

        # Loading category mapping file
        try:
            with open('data/processed/category_mapping.json', 'r') as f:
                self.category_mapping = json.load(f)
                self.reverse_mapping = {v: k for k, v in self.category_mapping.items()}
            print(f"Loaded {len(self.category_mapping)} categories")
        except:
            print("Category mapping not found")

        # Loading individual models
        model_info = [
            ('naive_bayes', NaiveBayesRecommender, 'Naive Bayes'),
            ('knn', KNNRecommender, 'K-Nearest Neighbors'),
            ('logistic_regression', LogisticRegressionRecommender, 'Logistic Regression')
        ]

        for key, ModelClass, name in model_info:
            try:
                model = ModelClass()
                model.load_model()
                self.models[key] = model
                print(f"{name} loaded")
            except Exception as e:
                print(f"{name} failed to load: {e}")

        # Checking if at least one model is loaded
        if not self.models:
            print("\nNo models loaded. Please train your models first.")
            sys.exit(1)

        # Creating hybrid ensemble when at least two models are available
        if len(self.models) >= 2:
            self.hybrid_model = HybridEnsemble(self.models, method='weighted_average')
            self.models['hybrid'] = self.hybrid_model
            print("Hybrid ensemble created")

        # Loading book database
        self.book_retrieval = BookRetrieval()
        print("Book database loaded")

        print(f"\n{len(self.models)} models ready\n")

    def get_user_input(self):
        """Requesting user to enter mood or preference text"""
        print("=" * 70)
        print("TELL US YOUR MOOD OR PREFERENCES")
        print("=" * 70)

        print("\nDescribe how you are feeling or what type of story you want.")
        print("\nYour input: ", end="")
        user_input = input().strip()

        if not user_input:
            print("Please enter something.")
            return self.get_user_input()

        return user_input

    def select_model(self):
        """Letting user choose which model to apply"""
        print("\n" + "=" * 70)
        print("SELECT AI MODEL")
        print("=" * 70)

        available_models = list(self.models.keys())

        model_names = {
            'hybrid': 'Hybrid Ensemble (Recommended)',
            'naive_bayes': 'Naive Bayes',
            'knn': 'K-Nearest Neighbors',
            'logistic_regression': 'Logistic Regression'
        }

        print("\nAvailable models:")
        for i, key in enumerate(available_models, 1):
            print(f"  {i}. {model_names.get(key, key)}")

        print("\nSelect (1-{}): ".format(len(available_models)), end="")

        try:
            choice = int(input().strip())
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]
        except:
            pass

        # Defaulting safely
        if "hybrid" in available_models:
            return "hybrid"
        return available_models[0]

    def get_recommendations(self, user_input, model_key):
        """Generating mood-aware category predictions and retrieving books"""
        model = self.models[model_key]

        try:
            # Processing user query through mood mapper
            query_analysis = self.mood_mapper.process_query(user_input)
            enhanced_query = query_analysis['enhanced_query']

            # Getting prediction scores
            predictions = model.predict_proba([enhanced_query])[0]

            # Applying mood-based adjustments
            for cat in query_analysis['avoid_categories']:
                if cat in self.category_mapping:
                    predictions[self.category_mapping[cat]] *= 0.1

            for cat in query_analysis['recommended_categories']:
                if cat in self.category_mapping:
                    predictions[self.category_mapping[cat]] *= 1.5

            predictions = predictions / predictions.sum()

            # Selecting top categories
            top_indices = predictions.argsort()[-5:][::-1]
            top_categories = [
                (self.reverse_mapping[idx], float(predictions[idx]))
                for idx in top_indices
            ]

            # Retrieving books
            books = self.book_retrieval.get_diverse_recommendations(
                top_categories,
                total_books=10
            )

            return {
                'query_analysis': query_analysis,
                'top_categories': top_categories,
                'books': books
            }

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None

    def display_recommendations(self, user_input, model_key, results):
        """Displaying category analysis and final book recommendations"""
        print("\n" + "=" * 70)
        print("BOOK RECOMMENDATIONS")
        print("=" * 70)

        query_analysis = results['query_analysis']

        # Showing detected mood
        print(f"\nYour query: '{user_input}'")

        if query_analysis['detected_moods']:
            print(f"Detected mood: {', '.join(query_analysis['detected_moods'])}")

        use_case_names = {
            'USE_CASE_1_MOOD': 'Mood-driven Discovery',
            'USE_CASE_2_PREFERENCE': 'Preference-based Recommendation',
            'USE_CASE_3_COMBINED': 'Combined Mood and Preference',
            'GENERAL_QUERY': 'General Query'
        }
        print(f"Type: {use_case_names.get(query_analysis['use_case'], 'General')}")

        # Showing model used
        model_names = {
            'hybrid': 'Hybrid Ensemble',
            'naive_bayes': 'Naive Bayes',
            'knn': 'K-Nearest Neighbors',
            'logistic_regression': 'Logistic Regression'
        }
        print(f"Model: {model_names.get(model_key, model_key)}")

        # Showing explanation
        explanation = self.mood_mapper.explain_recommendation(query_analysis)
        print("\n" + explanation)

        # Showing top predicted categories
        print("\n" + "-" * 70)
        print("Top Categories:")
        print("-" * 70)

        for i, (cat, conf) in enumerate(results['top_categories'][:3], 1):
            bar = "█" * int(conf * 50)
            print(f"{i}. {cat:20s} {conf * 100:5.1f}% {bar}")

        # Showing book recommendations
        print("\n" + "-" * 70)
        print("RECOMMENDED BOOKS:")
        print("-" * 70)

        if not results['books']:
            print("\nNo books found")
            return

        for i, book in enumerate(results['books'][:10], 1):
            print(f"\n{i}. {book['title']}")
            print(f"   by {book['authors']}")
            print(f"   Category: {book['category']}")

            if book['rating'] > 0:
                print(f"   Rating: {book['rating']:.1f} ({book['ratings_count']} ratings)")

            print(f"   Match Score: {book['category_confidence'] * 100:.1f}%")

            desc = book['description'][:100] + "..." if len(book['description']) > 100 else book['description']
            print(f"   {desc}")

        print("\n" + "=" * 70)

    def run(self):
        """Running main CLI loop"""
        self.load_models()

        while True:
            user_input = self.get_user_input()
            model_choice = self.select_model()

            results = self.get_recommendations(user_input, model_choice)
            if results:
                self.display_recommendations(user_input, model_choice, results)

            print("\n" + "=" * 70)
            print("Would you like another recommendation? (yes/no): ", end="")
            again = input().strip().lower()

            if again not in ['yes', 'y']:
                print("\nThank you for using the Book Recommendation System")
                print("=" * 70)
                break


def main():
    """Starting CLI application"""
    try:
        app = BookRecommenderCLI()
        app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        print("Ensure that all models are trained before running.")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
