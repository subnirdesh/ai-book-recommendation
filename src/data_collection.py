"""
Google Books API Data Collector
Collects book data from Google Books API for recommendation system
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime
import config

class GoogleBooksCollector:
    """
    Collects book data from Google Books API
    """
    
    def __init__(self, api_key=None):
        """
        Initialize collector
        
        Args:
            api_key: Google Books API key (optional but recommended for higher rate limits)
        """
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/books/v1/volumes"
        self.collected_books = []
        
    def search_books(self, query, max_results=40, start_index=0):
        """
        Search books using Google Books API
        
        Args:
            query: Search query
            max_results: Number of results per request (max 40)
            start_index: Starting index for pagination
            
        Returns:
            List of book items
        """
        params = {
            'q': query,
            'maxResults': min(max_results, 40),
            'startIndex': start_index,
            'printType': 'books',
            'langRestrict': 'en'
        }
        
        if self.api_key:
            params['key'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('items', [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return []
    
    def extract_book_info(self, item):
        """
        Extracts relevant information from a book item
        
        Args:
            item: Book item from API response
            
        Returns:
            Dictionary with book information
        """
        volume_info = item.get('volumeInfo', {})
        
        # Extracting basic info
        book_id = item.get('id', '')
        title = volume_info.get('title', '')
        authors = ', '.join(volume_info.get('authors', []))
        
        # Extracting categories
        categories = volume_info.get('categories', [])
        if categories:
            main_category = categories[0]
        else:
            main_category = 'Unknown'
        
        # Extracting description
        description = volume_info.get('description', '')
        
        # Extracting ratings
        average_rating = volume_info.get('averageRating', 0)
        ratings_count = volume_info.get('ratingsCount', 0)
        
        # Extracting other metadata
        published_date = volume_info.get('publishedDate', '')
        publisher = volume_info.get('publisher', '')
        page_count = volume_info.get('pageCount', 0)
        language = volume_info.get('language', 'en')
        
        # Extracting thumbnail
        image_links = volume_info.get('imageLinks', {})
        thumbnail = image_links.get('thumbnail', '')
        
        return {
            'book_id': book_id,
            'title': title,
            'authors': authors,
            'main_category': main_category,
            'all_categories': ', '.join(categories),
            'description': description,
            'average_rating': average_rating,
            'ratings_count': ratings_count,
            'published_date': published_date,
            'publisher': publisher,
            'page_count': page_count,
            'language': language,
            'thumbnail': thumbnail
        }
    
    def collect_by_category(self, category, books_per_category=100):
        """
        Collects books from a specific category
        
        Args:
            category: Category to search for
            books_per_category: Number of books to collect
            
        Returns:
            List of book dictionaries
        """
        print(f"\nCollecting books from category: {category}")
        books = []
        start_index = 0
        
        while len(books) < books_per_category:
            # Searching with category
            query = f"subject:{category}"
            items = self.search_books(query, max_results=40, start_index=start_index)
            
            if not items:
                print(f"No more results for {category}")
                break
            
            # Extracting book info
            for item in items:
                if len(books) >= books_per_category:
                    break
                    
                book_info = self.extract_book_info(item)
                
                # Filtering out books without essential data
                if (book_info['title'] and 
                    book_info['authors'] and 
                    book_info['description'] and
                    len(book_info['description']) > 50):
                    books.append(book_info)
            
            print(f"Collected {len(books)}/{books_per_category} books from {category}")
            
            start_index += 40
            time.sleep(1)  # Rate limiting
            
            # Safety break
            if start_index > 400:  # Max ~10 requests per category
                break
        
        return books
    
    def collect_balanced_dataset(self, categories=None, books_per_category=100):
        """
        Collect a balanced dataset across multiple categories
        
        Args:
            categories: List of categories to collect (None for default list)
            books_per_category: Number of books per category
            
        Returns:
            DataFrame with collected books
        """
        if categories is None:
            # Default balanced categories
            categories = [
                'fiction',
                'science fiction',
                'fantasy',
                'mystery',
                'thriller',
                'romance',
                'horror',
                'biography',
                'history',
                'science',
                'technology',
                'self-help',
                'business',
                'young adult',
                'children',
                'poetry'
            ]
        
        print(f"Starting balanced data collection...")
        print(f"Categories: {len(categories)}")
        print(f"Target books per category: {books_per_category}")
        print(f"Total target: {len(categories) * books_per_category}")
        print("=" * 50)
        
        all_books = []
        
        for category in categories:
            books = self.collect_by_category(category, books_per_category)
            all_books.extend(books)
            print(f"Total collected so far: {len(all_books)}")
            time.sleep(2)  # Rate limiting between categories
        
        # Creating DataFrame
        df = pd.DataFrame(all_books)
        
        # Removing duplicates based on book_id
        df = df.drop_duplicates(subset=['book_id'], keep='first')
        
        print("\n" + "=" * 50)
        print("Data Collection Complete!")
        print(f"Total unique books: {len(df)}")
        print("\nBooks per category:")
        print(df['main_category'].value_counts())
        
        return df
    
    def save_dataset(self, df, filename='google_books_dataset.csv'):
        """
        Save collected dataset to CSV
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        filepath = config.RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"\n✓ Dataset saved to: {filepath}")
        
        # Also save as JSON for backup
        json_filepath = config.RAW_DATA_DIR / filename.replace('.csv', '.json')
        df.to_json(json_filepath, orient='records', indent=2)
        print(f"✓ Backup saved to: {json_filepath}")
    
    def get_collection_stats(self, df):
        """
        Get statistics about the collected dataset
        
        Args:
            df: DataFrame with collected data
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_books': len(df),
            'unique_books': df['book_id'].nunique(),
            'categories': df['main_category'].nunique(),
            'avg_rating': df['average_rating'].mean(),
            'books_with_ratings': (df['ratings_count'] > 0).sum(),
            'avg_description_length': df['description'].str.len().mean(),
            'date_range': f"{df['published_date'].min()} to {df['published_date'].max()}"
        }
        
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return stats

def main():
    """
    Main function to collect data
    """
    # Initializing collector
    collector = GoogleBooksCollector(api_key='AIzaSyBfBufhUID3sJ80UGfAL810_gs1EzjoyGk')

    print("Starting data collection...")
    
    # Defining categories for balanced dataset
    categories = [
        'fiction',
        'science fiction',
        'fantasy',
        'mystery',
        'thriller',
        'romance',
        'horror',
        'biography',
        'history',
        'science',
        'technology',
        'self-help',
        'business',
        'young adult'
    ]
    
    
    df = collector.collect_balanced_dataset(
        categories=categories,
        books_per_category=150  # Adjusting number as per our need
    )
    
    # Getting statistics
    collector.get_collection_stats(df)
    
    # Saving dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'google_books_dataset_{timestamp}.csv'
    collector.save_dataset(df, filename)
    
    print("\n✓ Data collection complete!")
    print("\nNext steps:")
    print("1. Check the data in data/raw/")
    print("2. Run data_preprocessing.py to clean the data")
    print("3. Start training your models!")

if __name__ == '__main__':
    main()