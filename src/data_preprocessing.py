"""
FINAL BOOK DATA PREPROCESSOR
Balanced to 650 samples per each of the 11 super-genres
"""

import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import config


class BookPreProcessor:
    """Complete preprocessing pipeline for Google Books dataset"""

    # ---- Super-genre mapping (11 genres) ----
    TARGET_GENRES = {
        "Fiction": ["fiction", "novel", "literature", "story"],
        "History": ["history", "historical", "war", "civilization"],
        "Business & Economics": ["business", "economics", "finance", "management"],
        "Biography & Autobiography": ["biography", "autobiography", "memoir"],
        "Science": ["science", "physics", "chemistry", "biology"],
        "Juvenile Fiction": ["juvenile", "young adult", "children", "ya"],
        "Language Arts & Disciplines": ["language", "linguistics", "grammar"],
        "Literary Criticism": ["criticism", "literary analysis"],
        "Education": ["education", "teaching", "learning", "pedagogy"],
        "Computers": ["computers", "technology", "programming", "software"],
        "Philosophy": ["philosophy", "ethics", "metaphysics"],
    }

    TARGET_GENRE_LIST = list(TARGET_GENRES.keys())

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.category_mapping = {}


    # LOAD DATA
    def load_data(self, filepath):
        try:
            df = pd.read_csv(filepath)
            print("✓ Data loaded")
            print("  Shape:", df.shape)
            return df
        except Exception as e:
            print("ERROR loading data:", e)
            return None


    # TEXT CLEANING
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # HANDLE MISSING VALUES
    def handle_missing(self, df):
        print("\nHandling missing values...")

        df = df.copy()

        text_cols = ["title", "authors", "description", "publisher"]
        for c in text_cols:
            df[c] = df[c].fillna("Unknown")

        df["main_category"] = df["main_category"].fillna("Unknown")
        df["average_rating"] = df["average_rating"].fillna(0)
        df["ratings_count"] = df["ratings_count"].fillna(0)

        print("✓ Missing values handled")
        return df

    
    # MAP 963 categories → 11 super-genres
    def map_to_super_genres(self, df):
        print("\nMapping categories to 11 super-genres...")

        df = df.copy()

        def infer(cat_text):
            t = str(cat_text).lower()
            for genre, keywords in self.TARGET_GENRES.items():
                if any(k in t for k in keywords):
                    return genre
            return None

        df["mapped_category"] = df["main_category"].apply(infer)

        # fallback: use all_categories field
        mask_missing = df["mapped_category"].isna()
        df.loc[mask_missing, "mapped_category"] = (
            df.loc[mask_missing, "all_categories"].apply(infer)
        )

        # final fallback
        df["mapped_category"] = df["mapped_category"].fillna("Fiction")

        print("✓ Category mapping complete")
        print(df["mapped_category"].value_counts())

        df["main_category"] = df["mapped_category"]
        df = df.drop(columns=["mapped_category"])
        return df


    # FILTER BAD ENTRIES
    def filter_quality(self, df):
        print("\nFiltering low-quality books...")

        before = len(df)
        df = df[
            (df["title"].str.len() > 2)
            & (df["authors"].str.len() > 2)
            & (df["description"].str.len() > 30)
        ]

        print(f"✓ Removed {before - len(df)} low-quality entries")
        return df

    
    # BALANCE DATASET → 650 per genre
    def balance(self, df, target_per_genre=650):
        print("\nBalancing dataset to 650 per genre...")

        df_bal = []

        for genre in self.TARGET_GENRE_LIST:
            subset = df[df["main_category"] == genre]

            count = len(subset)
            print(f"- {genre}: {count} books")

            if count == 0:
                continue

            # Oversample if too few
            if count < target_per_genre:
                subset = subset.sample(
                    n=target_per_genre, replace=True, random_state=42
                )
            # Undersample if too many
            elif count > target_per_genre:
                subset = subset.sample(
                    n=target_per_genre, replace=False, random_state=42
                )

            df_bal.append(subset)

        df_final = pd.concat(df_bal, ignore_index=True)

        print("\nFinal balanced counts:")
        print(df_final["main_category"].value_counts())

        print(f"TOTAL: {len(df_final)} books (11 × {target_per_genre})")

        return df_final


    # BUILD TEXT FEATURES
    def build_text_features(self, df):
        df = df.copy()

        print("\nBuilding text features...")

        df["title_clean"] = df["title"].apply(self.clean_text)
        df["authors_clean"] = df["authors"].apply(self.clean_text)
        df["description_clean"] = df["description"].apply(self.clean_text)

        df["combined_text"] = (
            df["title_clean"]
            + " "
            + df["authors_clean"]
            + " "
            + df["description_clean"]
        )

        df["text_length"] = df["combined_text"].str.len()
        df["word_count"] = df["combined_text"].str.split().str.len()

        print("✓ Text features created")
        return df

   
    # LABEL ENCODING
    def encode_labels(self, df):
        print("\nEncoding category labels...")

        df = df.copy()
        df["category_encoded"] = self.label_encoder.fit_transform(
            df["main_category"]
        )

        self.category_mapping = {
            c: int(i) for c, i in zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_))
        }

        print("✓ Label encoding complete")
        print(self.category_mapping)
        return df

    
    # SPLIT TRAIN / VAL / TEST
    def split(self, df):
        print("\nSplitting into train/val/test...")

        train_val, test = train_test_split(
            df,
            test_size=0.2,
            stratify=df["category_encoded"],
            random_state=42,
        )

        val_size_adjusted = 0.1 / 0.8

        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val["category_encoded"],
            random_state=42,
        )

        print(f"Train: {len(train)}")
        print(f"Val:   {len(val)}")
        print(f"Test:  {len(test)}")

        return train, val, test

    
    # SAVE DATA
    def save(self, train, val, test, prefix="processed"):
        print("\nSaving processed CSV files...")

        config.PROCESSED_DATA_DIR.mkdir(exist_ok=True)

        train.to_csv(config.PROCESSED_DATA_DIR / f"{prefix}_train.csv", index=False)
        val.to_csv(config.PROCESSED_DATA_DIR / f"{prefix}_val.csv", index=False)
        test.to_csv(config.PROCESSED_DATA_DIR / f"{prefix}_test.csv", index=False)

        with open(config.PROCESSED_DATA_DIR / "category_mapping.json", "w") as f:
            json.dump(self.category_mapping, f, indent=2)

        print("✓ Files saved")

    
    # MAIN PIPELINE
    def run(self, filepath):
        print("\n========== BOOK PREPROCESSING ==========")

        df = self.load_data(filepath)
        df = self.handle_missing(df)
        df = self.map_to_super_genres(df)
        df = self.filter_quality(df)
        df = self.balance(df, target_per_genre=650)
        df = self.build_text_features(df)
        df = self.encode_labels(df)

        train, val, test = self.split(df)
        self.save(train, val, test)

        print("\n✓ Preprocessing COMPLETE")
        print("========================================\n")



# MAIN
def main():
    processor = BookPreProcessor()
    processor.run(config.RAW_DATA_DIR / "final_raw.csv")


if __name__ == "__main__":
    main()
