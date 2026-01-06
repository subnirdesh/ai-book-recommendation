"""
Diagnose low accuracy and improve model performance
Run this to understand WHY accuracy is low and fix it
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path

def diagnose_problems():
    """Diagnose why accuracy is low"""
    
    print("="*70)
    print("ACCURACY DIAGNOSIS")
    print("="*70)
    
    # Load data
    train = pd.read_csv('data/processed/processed_train.csv')
    test = pd.read_csv('data/processed/processed_test.csv')
    
    with open('data/processed/category_mapping.json', 'r') as f:
        category_mapping = json.load(f)
    
    print("\n1️⃣ CLASS IMBALANCE CHECK")
    print("-"*70)
    
    train_dist = train['main_category'].value_counts()
    test_dist = test['main_category'].value_counts()
    
    print("\nTraining set distribution:")
    for cat, count in train_dist.items():
        percentage = count / len(train) * 100
        print(f"  {cat:20s}: {count:3d} ({percentage:5.1f}%)")
    
    # Check imbalance ratio
    max_count = train_dist.max()
    min_count = train_dist.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\n⚠️ Imbalance Ratio: {imbalance_ratio:.2f}x")
    print(f"   Largest class: {train_dist.idxmax()} ({max_count} books)")
    print(f"   Smallest class: {train_dist.idxmin()} ({min_count} books)")
    
    if imbalance_ratio > 5:
        print("   ❌ SEVERE IMBALANCE - This is likely hurting accuracy!")
    elif imbalance_ratio > 3:
        print("   ⚠️ MODERATE IMBALANCE - Could affect accuracy")
    else:
        print("   ✓ Acceptable balance")
    
    print("\n2️⃣ 'GENERAL' CATEGORY PROBLEM")
    print("-"*70)
    
    if 'General' in train_dist.index:
        general_count = train_dist['General']
        general_pct = general_count / len(train) * 100
        print(f"\n'General' category has {general_count} books ({general_pct:.1f}%)")
        
        # Sample some "General" books
        general_books = train[train['main_category'] == 'General'].head(5)
        print("\nSample 'General' books:")
        for idx, book in general_books.iterrows():
            print(f"  - {book['title'][:60]}")
            print(f"    Original: {book.get('all_categories', 'N/A')[:60]}")
        
        print("\n⚠️ 'General' is too vague - models can't learn patterns!")
        print("   RECOMMENDATION: Remove or redistribute 'General' books")
    
    print("\n3️⃣ TEXT QUALITY CHECK")
    print("-"*70)
    
    print(f"\nAverage text length: {train['text_length'].mean():.0f} characters")
    print(f"Average word count: {train['word_count'].mean():.0f} words")
    
    # Check for very short texts
    short_texts = train[train['text_length'] < 100]
    if len(short_texts) > 0:
        print(f"\n⚠️ {len(short_texts)} books have < 100 characters")
        print("   Very short texts don't have enough information for classification")
    
    # Check per category
    print("\nText length by category:")
    for cat in train_dist.index:
        cat_texts = train[train['main_category'] == cat]
        avg_len = cat_texts['text_length'].mean()
        print(f"  {cat:20s}: {avg_len:6.0f} chars average")
    
    print("\n4️⃣ FEATURE OVERLAP ANALYSIS")
    print("-"*70)
    
    # Check if categories have distinct vocabularies
    print("\nChecking vocabulary overlap between categories...")
    
    # Sample descriptions from different categories
    categories_to_check = ['Fiction', 'Science', 'Business']
    available_cats = [c for c in categories_to_check if c in train_dist.index]
    
    if len(available_cats) >= 2:
        for cat in available_cats[:3]:
            cat_sample = train[train['main_category'] == cat]['combined_text'].head(3)
            print(f"\n{cat} sample texts:")
            for text in cat_sample:
                words = text.split()[:15]
                print(f"  {' '.join(words)}...")
    
    print("\n" + "="*70)
    print("📊 DIAGNOSIS SUMMARY")
    print("="*70)
    
    problems_found = []
    
    if imbalance_ratio > 3:
        problems_found.append("❌ Class imbalance")
    
    if 'General' in train_dist.index and train_dist['General'] > 50:
        problems_found.append("❌ Large 'General' category")
    
    if train['text_length'].mean() < 300:
        problems_found.append("⚠️ Short text descriptions")
    
    if len(problems_found) > 0:
        print("\n🔴 PROBLEMS FOUND:")
        for p in problems_found:
            print(f"  {p}")
    else:
        print("\n✓ No major problems detected")
    
    return train, test, category_mapping

def create_confusion_matrix(model, X_test, y_test, category_mapping, model_name):
    """Create confusion matrix visualization"""
    
    print(f"\n{'='*70}")
    print(f"CONFUSION MATRIX - {model_name}")
    print("="*70)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get category names in order
    categories = [k for k, v in sorted(category_mapping.items(), key=lambda x: x[1])]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    output_path = Path('trained_models') / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {output_path}")
    plt.show()
    
    # Analyze errors
    print("\n📊 Error Analysis:")
    
    # Find most confused pairs
    np.fill_diagonal(cm, 0)  # Remove diagonal for analysis
    
    most_confused_idx = np.unravel_index(cm.argmax(), cm.shape)
    true_cat = categories[most_confused_idx[0]]
    pred_cat = categories[most_confused_idx[1]]
    confusion_count = cm[most_confused_idx]
    
    print(f"\nMost confused pair:")
    print(f"  {true_cat} → {pred_cat}: {confusion_count} misclassifications")
    
    # Per-class accuracy
    print("\nPer-category accuracy:")
    cm_with_diag = confusion_matrix(y_test, y_pred)
    for i, cat in enumerate(categories):
        correct = cm_with_diag[i, i]
        total = cm_with_diag[i, :].sum()
        accuracy = correct / total if total > 0 else 0
        print(f"  {cat:20s}: {accuracy*100:5.1f}% ({correct}/{total})")

def show_misclassified_examples(model, X_test, y_test, test_df, category_mapping, model_name):
    """Show examples of misclassified books"""
    
    print(f"\n{'='*70}")
    print(f"MISCLASSIFIED EXAMPLES - {model_name}")
    print("="*70)
    
    # Get predictions
    y_pred = model.predict(X_test.values)
    
    # Find misclassified
    misclassified_mask = y_pred != y_test.values
    misclassified_df = test_df[misclassified_mask].copy()
    misclassified_df['predicted'] = y_pred[misclassified_mask]
    
    print(f"\nTotal misclassifications: {len(misclassified_df)} / {len(test_df)}")
    
    # Reverse category mapping
    id_to_cat = {v: k for k, v in category_mapping.items()}
    
    # Show some examples
    print("\nExample misclassifications:")
    for idx, book in misclassified_df.head(5).iterrows():
        true_cat = id_to_cat[book['category_encoded']]
        pred_cat = id_to_cat[book['predicted']]
        
        print(f"\n{'-'*70}")
        print(f"Title: {book['title'][:70]}")
        print(f"True category: {true_cat}")
        print(f"Predicted: {pred_cat}")
        print(f"Text preview: {book['combined_text'][:150]}...")

def suggest_improvements():
    """Suggest specific improvements"""
    
    print("\n" + "="*70)
    print("🔧 RECOMMENDED FIXES")
    print("="*70)
    
    print("""
1. REBALANCE DATASET
   ✓ Remove or redistribute 'General' category
   ✓ Undersample Fiction to max 150 books
   ✓ Remove categories with < 40 books
   ✓ Target: All categories between 40-150 books

2. IMPROVE TEXT FEATURES
   ✓ Filter books with description < 100 characters
   ✓ Add book title weight (titles are important!)
   ✓ Use bigrams in TF-IDF (not just single words)
   ✓ Increase max_features to 10000

3. TUNE MODEL HYPERPARAMETERS
   ✓ KNN: Try k=3, k=7, k=10
   ✓ TF-IDF: Use bigrams (ngram_range=(1,2))
   ✓ Logistic Regression: Try different C values

4. TRY ENSEMBLE
   ✓ Combine all 3 models (voting)
   ✓ Use best 2 models

Expected improvement: 43% → 65-75% accuracy
    """)

def main():
    """Main diagnostic function"""
    
    # Run diagnosis
    train, test, category_mapping = diagnose_problems()
    
    # Load models if they exist
    from src.models.naive_bayes_model import NaiveBayesRecommender
    from src.models.knn_model import KNNRecommender
    from src.models.logistic_regression_model import LogisticRegressionRecommender
    
    X_test = test['combined_text']
    y_test = test['category_encoded']
    
    models_to_test = []
    
    # Try loading models
    try:
        nb = NaiveBayesRecommender()
        nb.load_model()
        models_to_test.append(('Naive Bayes', nb))
    except:
        print("⚠️ Naive Bayes model not found")
    
    try:
        knn = KNNRecommender()
        knn.load_model()
        models_to_test.append(('KNN', knn))
    except:
        print("⚠️ KNN model not found")
    
    try:
        lr = LogisticRegressionRecommender()
        lr.load_model()
        models_to_test.append(('Logistic Regression', lr))
    except:
        print("⚠️ Logistic Regression model not found")
    
    # Create confusion matrices for loaded models
    for model_name, model in models_to_test:
        create_confusion_matrix(model, X_test, y_test, category_mapping, model_name)
        show_misclassified_examples(model, X_test, y_test, test, category_mapping, model_name)
    
    # Suggest improvements
    suggest_improvements()

if __name__ == '__main__':
    main()