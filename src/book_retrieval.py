"""
Book Retrieval System
Maps predicted categories to actual books from dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

class BookRetrieval:
    """Retrieving actual books based on category predictions"""
    
    def __init__(self):
        self.books_df = None
        self.load_books()
    
    def load_books(self):
        """Loading book dataset from processed CSV files"""
        try:
            # Trying to load processed data with category info
            train = pd.read_csv('data/processed/processed_train.csv')
            val = pd.read_csv('data/processed/processed_val.csv')
            test = pd.read_csv('data/processed/processed_test.csv')
            
            # Combining all data into single dataframe
            self.books_df = pd.concat([train, val, test], ignore_index=True)
            
            # Removing duplicate books based on title and authors
            original_count = len(self.books_df)
            self.books_df = self.books_df.drop_duplicates(
                subset=['title', 'authors'], 
                keep='first'
            )
           
            
            print(f"Loaded {len(self.books_df)} books from dataset")
            
            
        except FileNotFoundError as e:
            print(f"Error: Dataset files not found - {e}")
            print("Please ensure processed data exists in data/processed/")
            self.books_df = pd.DataFrame()
        except Exception as e:
            print(f"Error loading books: {e}")
            self.books_df = pd.DataFrame()
    
    def get_books_by_category(self, category, limit=5, min_rating=0):
        """
        Getting books from a specific category
        
        Args:
            category: Category name
            limit: Number of books to return
            min_rating: Minimum rating threshold
            
        Returns:
            List of book dictionaries
        """
        # Checking if books are loaded
        if self.books_df is None or len(self.books_df) == 0:
            print("No books available in dataset")
            return []
        
        # Checking if category exists in dataset
        if category not in self.books_df['main_category'].values:
            print(f"Category '{category}' not found in dataset")
            return []
        
        # Filtering books by category
        category_books = self.books_df[
            self.books_df['main_category'] == category
        ].copy()
        
        if len(category_books) == 0:
            return []
        
        # Filtering by rating if minimum rating is specified
        if min_rating > 0 and 'average_rating' in category_books.columns:
            category_books = category_books[
                category_books['average_rating'] >= min_rating
            ]
        
        # Sorting by rating and ratings count to get quality books
        if 'average_rating' in category_books.columns:
            category_books['quality_score'] = (
                category_books['average_rating'] * 
                np.log1p(category_books['ratings_count'])
            )
            category_books = category_books.sort_values(
                'quality_score', ascending=False
            )
        
        # Getting top books based on limit
        top_books = category_books.head(limit)
        
        # Formatting book information into dictionary format
        books = []
        for idx, book in top_books.iterrows():
            books.append({
                'title': book['title'],
                'authors': book['authors'],
                'category': book['main_category'],
                'rating': float(book.get('average_rating', 0)),
                'ratings_count': int(book.get('ratings_count', 0)),
                'description': str(book.get('description', 'No description available'))[:200] + '...'
            })
        
        return books
    
    def get_books_for_predictions(self, category_predictions, books_per_category=3):
        """
        Getting actual books based on category predictions
        
        Args:
            category_predictions: List of (category, confidence) tuples
            books_per_category: Number of books per category
            
        Returns:
            List of books with recommendation scores
        """
        all_recommendations = []
        
        # Looping through each predicted category
        for category, confidence in category_predictions:
            # Getting books from this category
            books = self.get_books_by_category(
                category, 
                limit=books_per_category
            )
            
            # Adding confidence score to each book
            for book in books:
                book['category_confidence'] = confidence
                book['recommendation_score'] = confidence
                all_recommendations.append(book)
        
        # Sorting by recommendation score in descending order
        all_recommendations.sort(
            key=lambda x: x['recommendation_score'], 
            reverse=True
        )
        
        return all_recommendations
    
    def search_similar_books(self, book_title, category, limit=5):
        """
        Finding books similar to a given title
        
        Args:
            book_title: Reference book title
            category: Category to search in
            limit: Number of results
            
        Returns:
            List of similar books
        """
        # Checking if books are loaded
        if self.books_df is None or len(self.books_df) == 0:
            print("No books available in dataset")
            return []
        
        # Getting books from same category excluding the input book
        category_books = self.books_df[
            (self.books_df['main_category'] == category) &
            (self.books_df['title'] != book_title)
        ].copy()
        
        # Sorting by average rating to get best books
        if 'average_rating' in category_books.columns:
            category_books = category_books.sort_values(
                'average_rating', 
                ascending=False
            )
        
        # Formatting and returning top similar books
        books = []
        for idx, book in category_books.head(limit).iterrows():
            books.append({
                'title': book['title'],
                'authors': book['authors'],
                'category': book['main_category'],
                'rating': float(book.get('average_rating', 0)),
                'ratings_count': int(book.get('ratings_count', 0)),
                'description': str(book.get('description', 'No description available'))[:200] + '...'
            })
        
        return books
    
    def get_diverse_recommendations(self, category_predictions, total_books=10):
        """
        Getting diverse book recommendations across multiple categories
        
        Args:
            category_predictions: List of (category, confidence) tuples
            total_books: Total number of books to recommend
            
        Returns:
            List of diverse book recommendations
        """
        # Checking if predictions are provided
        if not category_predictions:
            print("No category predictions provided")
            return []
        
        recommendations = []
        
        # Calculating books per category for even distribution
        num_categories = min(3, len(category_predictions))
        books_per_category = max(2, total_books // num_categories)
        
        # Getting books from top 3 categories
        for category, confidence in category_predictions[:num_categories]:
            books = self.get_books_by_category(
                category, 
                limit=books_per_category
            )
            
            # Adding scores to each book
            for book in books:
                book['category_confidence'] = confidence
                book['recommendation_score'] = confidence
                recommendations.append(book)
        
        # Sorting by score and limiting to total_books
        recommendations.sort(
            key=lambda x: x['recommendation_score'], 
            reverse=True
        )
        
        return recommendations[:total_books]
    
    def print_recommendations(self, recommendations):
        """Printing book recommendations in formatted way"""
        if not recommendations:
            print("No recommendations to display")
            return
        
        print("\n" + "="*50)
        print("BOOK RECOMMENDATIONS")
        print("="*50)
        
        for i, book in enumerate(recommendations, 1):
            print(f"\n{i}. {book['title']}")
            print(f"   Author: {book['authors']}")
            print(f"   Category: {book['category']}")
            print(f"   Rating: {book['rating']:.2f}/5.0")
            print(f"   Ratings Count: {book['ratings_count']}")
            if 'recommendation_score' in book:
                print(f"   Confidence Score: {book['recommendation_score']:.2f}")
        
        print("\n" + "="*50)
    
    def get_category_list(self):
        """Getting list of all available categories"""
        if self.books_df is None or len(self.books_df) == 0:
            return []
        
        return sorted(self.books_df['main_category'].unique().tolist())
    
    def get_total_books(self):
        """Getting total number of books in dataset"""
        if self.books_df is None or len(self.books_df) == 0:
            return 0
        
        return len(self.books_df)


# Testing the BookRetrieval class
if __name__ == "__main__":
    print("Testing Book Retrieval System...")
    print("-" * 50)
    
    # Creating instance of BookRetrieval
    retrieval = BookRetrieval()
    
    # Testing if books loaded successfully
    print(f"\nTotal books in dataset: {retrieval.get_total_books()}")
    
    # Testing category list
    categories = retrieval.get_category_list()
    print(f"Available categories: {len(categories)}")
    if categories:
        print(f"Sample categories: {categories[:5]}")
    
    # Testing getting books by category
    if categories:
        test_category = categories[0]
        print(f"\n\nTesting get_books_by_category with '{test_category}'...")
        books = retrieval.get_books_by_category(test_category, limit=3)
        retrieval.print_recommendations(books)
    
    # Testing predictions with actual categories from dataset
    print("\n\nTesting get_books_for_predictions...")
    if len(categories) >= 3:
        sample_predictions = [
            (categories[4], 0.85),  # Fiction
            (categories[2], 0.60),  # Computers
            (categories[3], 0.45)   # Education
        ]
        print(f"Using categories: {[cat for cat, _ in sample_predictions]}")
    else:
        sample_predictions = [(categories[0], 0.85)]
    
    recommendations = retrieval.get_books_for_predictions(
        sample_predictions, 
        books_per_category=2
    )
    retrieval.print_recommendations(recommendations)
    
    # Testing diverse recommendations
    print("\n\nTesting get_diverse_recommendations...")
    diverse_recs = retrieval.get_diverse_recommendations(
        sample_predictions, 
        total_books=6
    )
    retrieval.print_recommendations(diverse_recs)
    
    print("\n\nAll tests completed!")