import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Importing models
from src.models.naive_bayes_model import NaiveBayesRecommender
from src.models.knn_model import KNNRecommender
from src.models.logistic_regression_model import LogisticRegressionRecommender


class BookRecommenderCLI:

    def __init__(self):
        self.models={}
        self.category_mapping={}
        self.reverse_mapping={}


    def load_models(self):
        """Load all trained models"""

        print("="*70)
        print("BOOK RECOMMENDATION SYSTEM")
        print("="*70)
        print("\nLoading AI models...")

        # Loading category mapping
        try:
            with open('data/processed/category_mapping.json', 'r') as f:
                self.category_mapping = json.load(f)
                self.reverse_mapping = {v: k for k, v in self.category_mapping.items()}
            print(f"Loaded {len(self.category_mapping)} categories")
        except:
            print("Category mapping not found")


        # Trying  loading each model
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
                print(f"{name} failed: {e}")


        if not self.models:
            print("\n No models loaded! Please train models first.")
            sys.exit(1)

        print(f"\n {len(self.models)} models ready!\n")

    
    def get_user_input(self):

        """Gets book preferences from user"""

        print("="*70)
        print("TELL US ABOUT YOUR BOOK PREFERENCES")
        print("="*70)

        print("\nDescribe the kind of book you're looking for.")
        print("Examples:")
        print("  - 'science fiction space adventure with aliens'")
        print("  - 'mystery detective murder investigation'")
        print("  - 'business leadership and management'")
        print("  - 'fantasy magic wizards and dragons'")
        print("\nYour input: ", end="")

        user_input = input().strip()

        if not user_input:
            print("⚠️ Please enter something!")
            return self.get_user_input()
        
        return user_input
    
    def select_model(self):

        """Lets user choose which model to use"""

        print("\n" + "="*70)
        print("SELECT AI MODEL")
        print("="*70)

        available_models=list(self.models.keys())

        print("\nAvailable models:")
        for i, key in enumerate(available_models, 1):
            model_names = {
                'naive_bayes': 'Naive Bayes',
                'knn': 'K-Nearest Neighbors',
                'logistic_regression': 'Logistic Regression'
            }
            print(f"  {i}. {model_names.get(key, key)}")
        print(f"  {len(available_models)+1}. Compare all models")
        print("\nSelect (1-{}): ".format(len(available_models)+1), end="")

        try:
            choice = int(input().strip())
            if 1 <= choice <= len(available_models):
                return available_models[choice-1]
            elif choice == len(available_models)+1:
                return 'all'
            else:
                print("Invalid choice!")
                return self.select_model()
        except:
            print("Please enter a number!")
            return self.select_model()
        

    def get_recommendations(self,user_input,model_key,top_n=5):

        """Get recommendations from a specific model"""

        model=self.models[model_key]

        try:
            # Getting top predictions
            predictions = model.predict_proba([user_input])[0]
            top_indices = predictions.argsort()[-top_n:][::-1]
            top_scores = predictions[top_indices]
            
            recommendations = []
            for idx, score in zip(top_indices, top_scores):
                category = self.reverse_mapping.get(idx, f"Category {idx}")
                recommendations.append({
                    'category': category,
                    'confidence': score,
                    'rank': len(recommendations) + 1
                })
            
            return recommendations
        
        except:
            print(f"Error getting recommendations: {e}")
            return []
        

    def display_recommendations(self,user_input,model_key,recommendations):
             
         """Displays recommendations nicely"""

         print("\n" + "="*70)
         print("BOOK RECOMMENDATIONS")
         print("="*70)

         print(f"\nYour query: '{user_input}'")

         model_names = {
            'naive_bayes': 'Naive Bayes',
            'knn': 'K-Nearest Neighbors',
            'logistic_regression': 'Logistic Regression'
        }
         
         print(f"Model used: {model_names.get(model_key, model_key)}")
        
         print("\nTop book categories for you:")
         print("-"*70)


         for rec in recommendations:
            confidence_pct = rec['confidence'] * 100
            bar_length = int(confidence_pct / 5)
            bar = "█" * bar_length
            
            print(f"\n{rec['rank']}. {rec['category']}")
            print(f"   Confidence: {confidence_pct:5.1f}% {bar}")
        
         print("\n" + "="*70)
        

    def compare_all_models(self,user_input,top_n=3):

        """Compares recommendations from all models"""

        print("\n" + "="*70)
        print("COMPARING ALL AI MODELS")
        print("="*70)
        print(f"\nYour query: '{user_input}'")

        for model_key, model in self.models.items():
            model_names = {
                'naive_bayes': 'Naive Bayes',
                'knn': 'K-Nearest Neighbors (KNN)',
                'logistic_regression': 'Logistic Regression'
            }

            print(f"\n{'-'*70}")
            print(f"{model_names.get(model_key, model_key)}")
            print(f"{'-'*70}")

            recommendations = self.get_recommendations(user_input, model_key, top_n)

            for rec in recommendations:
                confidence_pct = rec['confidence'] * 100
                print(f"  {rec['rank']}. {rec['category']:20s} - {confidence_pct:5.1f}%")

        print("\n" + "="*70)


    def run(self):

        """Main application loop"""

        self.load_models()

        while True:

            # Getting user input
            user_input = self.get_user_input()

            # Selecting model
            model_choice = self.select_model()

            # Getting and displaying recommendations
            if model_choice == 'all':
                self.compare_all_models(user_input)
            else:
                recommendations = self.get_recommendations(user_input, model_choice)
                self.display_recommendations(user_input, model_choice, recommendations)

            # Asking to continue
            print("\n" + "="*70)
            print("Would you like another recommendation? (yes/no): ", end="")
            again = input().strip().lower()
            
            if again not in ['yes', 'y']:
                print("\n✓ Thank you for using Book Recommendation System!")
                print("="*70)
                break

def main():

    try:
        app = BookRecommenderCLI()
        app.run()
    except KeyboardInterrupt:
        print("\n\nExiting... Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please make sure you've trained the models first!")

if __name__ == '__main__':
    main()
        



    
        