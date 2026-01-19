"""
Google Books API Data Collector
Collects LARGE, BALANCED datasets from Google Books API.
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
    Collects book data from Google Books API (large-scale version)
    """

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/books/v1/volumes"
        self.collected_books = []

    
    def search_books(self, query, max_results=40, start_index=0):
        params = {
            "q": query,
            "maxResults": min(max_results, 40),
            "startIndex": start_index,
            "printType": "books",
            "langRestrict": "en"
        }

        if self.api_key:
            params["key"] = self.api_key

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get("items", [])
        except Exception as e:
            print(f" API Error: {e}")
            return []

    # ============================================================
    # EXTRACT BOOK INFO
    # ============================================================
    def extract_book_info(self, item):
        info = item.get("volumeInfo", {})

        categories = info.get("categories", [])
        main_category = categories[0] if categories else "Unknown"

        return {
            "book_id": item.get("id", ""),
            "title": info.get("title", ""),
            "authors": ", ".join(info.get("authors", [])),
            "main_category": main_category,
            "all_categories": ", ".join(categories),
            "description": info.get("description", ""),
            "average_rating": info.get("averageRating", 0),
            "ratings_count": info.get("ratingsCount", 0),
            "published_date": info.get("publishedDate", ""),
            "publisher": info.get("publisher", ""),
            "page_count": info.get("pageCount", 0),
            "language": info.get("language", ""),
            "thumbnail": info.get("imageLinks", {}).get("thumbnail", "")
        }

    # ============================================================
    # COLLECT BOOKS FROM ONE CATEGORY (IMPROVED)
    # ============================================================
    def collect_by_category(self, category, target_amount=600):
        print(f"\nCollecting category: {category} (target: {target_amount})")
        books = []
        seen_ids = set()

        # 20+ query variations for DEEPEST coverage
        queries = [
            f"subject:{category}",
            f"intitle:{category}",
            f"inauthor:{category}",
            f"{category} fiction",
            f"{category} nonfiction",
            f"best {category} books",
            f"award winning {category}",
            f"{category} classics",
            f"{category} literature",
            f"{category} novel",
            f"{category} stories",
            f"{category} popular",
            f"{category} top rated",
            f"{category} new releases",
            f"{category} bestseller",
            f"genre:{category}",
            f"topic:{category}",
            f"category:{category}"
        ]

        for q in queries:
            print(f" → Query: {q}")
            start_index = 0

            while len(books) < target_amount:
                items = self.search_books(q, start_index=start_index)

                if not items:
                    break  # No more pages

                for item in items:
                    book = self.extract_book_info(item)
                    if (
                        book["book_id"] not in seen_ids
                        and book["title"]
                        and book["authors"]
                    ):
                        books.append(book)
                        seen_ids.add(book["book_id"])

                        if len(books) >= target_amount:
                            break

                start_index += 40
                time.sleep(0.4)

                # Extend pagination beyond Google's recommended depth
                if start_index > 2000:
                    break

            print(f"   Collected so far: {len(books)}/{target_amount}")

            if len(books) >= target_amount:
                break

        return books

    # ============================================================
    # BALANCED DATASET COLLECTION (IMPROVED)
    # ============================================================
    def collect_balanced_dataset(self, categories, books_per_category=600):
        print("\nStarting LARGE balanced dataset collection")
        print("================================================")
        print(f"Categories: {len(categories)}")
        print(f"Books per category: {books_per_category}")
        print(f"Expected dataset size: ~{len(categories) * books_per_category}")
        print("================================================\n")

        all_books = []
        seen_ids = set()

        for cat in categories:
            cat_books = self.collect_by_category(cat, books_per_category)
            all_books.extend(cat_books)
            seen_ids.update([b["book_id"] for b in cat_books])
            print(f"📌 Total collected so far: {len(all_books)}\n")
            time.sleep(1)

        df = pd.DataFrame(all_books).drop_duplicates(subset=["book_id"])

        print("\n==============================================")
        print("✔ Dataset Collection Complete")
        print("==============================================")
        print("Unique books:", len(df))
        print("\nBooks per category:")
        print(df["main_category"].value_counts())

        return df

    # ============================================================
    # SAVE + STATS (unchanged from your original)
    # ============================================================
    def save_dataset(self, df, filename="google_books_dataset.csv"):
        filepath = config.RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"\n✓ Dataset saved to: {filepath}")

        json_filepath = config.RAW_DATA_DIR / filename.replace(".csv", ".json")
        df.to_json(json_filepath, orient="records", indent=2)
        print(f"✓ Backup saved to: {json_filepath}")

    def get_collection_stats(self, df):
        stats = {
            "total_books": len(df),
            "unique_books": df["book_id"].nunique(),
            "categories": df["main_category"].nunique(),
            "avg_rating": df["average_rating"].mean(),
            "books_with_ratings": (df["ratings_count"] > 0).sum(),
            "avg_description_length": df["description"].str.len().mean(),
            "date_range": f"{df['published_date'].min()} - {df['published_date'].max()}"
        }

        print("\n============== DATASET STATISTICS ==============")
        for k, v in stats.items():
            print(f"{k}: {v}")

        return stats


# ============================================================
# MAIN (unchanged except higher book_per_category)
# ============================================================
def main():
    collector = GoogleBooksCollector(api_key=config.GOOGLE_BOOKS_API_KEY)

    categories = [
        "fiction", "fantasy", "science fiction", "mystery", "thriller",
        "romance", "horror", "history", "biography", "business",
        "science", "self-help", "philosophy", "young adult"
    ]

    df = collector.collect_balanced_dataset(
        categories=categories,
        books_per_category=600   # increased massively
    )

    collector.get_collection_stats(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"google_books_dataset_{timestamp}.csv"
    collector.save_dataset(df, filename)


if __name__ == "__main__":
    main()
