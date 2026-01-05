"""
Data Preprocessing for Google Books Dataset
Cleans and prepares data for ML models
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import config


class BookPreProcessor:
    """
    Handles all data preprocessing for Google Books dataset
    """

    def __init__(self):
        self.label_encoder=LabelEncoder()
        self.category_mapping={}

    
    def load_data(self,filepath):
        """Load dataset from CSV file"""

        try:
            df=pd.read_csv(filepath)
            print(f" Data loaded successfully!")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            return df
        
        except Exception as e:
            print(f" Error while loading Data: {e}")
            return None


    def clean_text(self,text):
       """
        Cleans and normalizes text data
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """ 
       
       if pd.isna(text) or text == '':
           return ""
       
       text =str(text).lower()

       # Removing HTML filter
       text=re.sub(r'<[^>]+>','',text)

       # Removing URLs
       text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Removing special characters but keep spaces
       text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
       # Removing extra whitespace
       text = re.sub(r'\s+', ' ', text).strip()

       return text
    

    def handle_missing_values(self,df):

        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
      """  
        
        print("/n Handling missing value ....")

        df=df.copy()

        # Showing missoing values before
        print("\nMissing values before: ")
        missing=df.isnull().sum()
        print(missing[missing>0])

        # Filling missing text fields with empty string
        text_columns=['title','authors','description','publisher']
        for col in text_columns:
            if col in df.columns:
                df[col].fillna('Unknown',inplace=True)

        # Filling missing categories with 'Unknown'
        if 'main_category' in df.columns:
         df['main_category'].fillna('Unknown',inplace=True)

         # Fill missing numeric fiels with 0
         numeric_column=['average_rating','ratings_count',"page_count"]
         for col in numeric_column:
             if col in df.columns:
                 df[col].fillna(0,inplace=True)

        print("\n Missing value handled")
        return df
    
    def filter_quality_books(self,df):
        """
        Filter out low-quality book entries
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """

        print("\n Filtering qiality books ... ")
        original_count=len(df)

        # Removing books without essentail information 
        df=df[
             (df['title'].str.len()> 0)&
             (df['authors'].str.len()> 0)&
             (df['description'].str.len()> 50)&
             (df['main_category'] !='Unknown')
        ].copy()

        removed= original_count-len(df)
        print(f" Removed {removed} low-qulaity entries")
        print(f" Remianing {len(df)} books")

        return df
    
    def clean_categories(self,df):

        """
        Clean and standardize category names - ENHANCED for Google Books
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned categories
        """

        print("\nCleaning and consolidating categories...")
        print(f"Original categories: {df['main_category'].nunique()}")

        df=df.copy()

        # Converting to lowercase for matching 
        df['main_category'] = df['main_category'].str.lower()


        def categorize_book(category):
            category =str(category).lower()


            # Fiction generes 
        
            if any(word in category for word in ['fiction', 'novel']):
                if any(word in category for word in ['science fiction', 'sci-fi', 'scifi']):
                    return 'Science Fiction'
                elif any(word in category for word in ['fantasy', 'magic', 'wizard']):
                    return 'Fantasy'
                elif any(word in category for word in ['mystery', 'detective', 'crime']):
                    return 'Mystery'
                elif any(word in category for word in ['thriller', 'suspense']):
                    return 'Thriller'
                elif any(word in category for word in ['romance', 'love']):
                    return 'Romance'
                elif any(word in category for word in ['horror', 'ghost']):
                    return 'Horror'
                elif any(word in category for word in ['juvenile', 'young adult', 'ya', 'teen']):
                    return 'Young Adult'
                else:
                    return 'Fiction'
                

            # Non-fiction categories
            elif any(word in category for word in ['biography', 'memoir', 'autobiography']):
                    return 'Biography'
            elif any(word in category for word in ['history', 'historical']):
                    return 'History'
            elif any(word in category for word in ['science', 'scientific']):
                    return 'Science'
            elif any(word in category for word in ['technology', 'computer', 'programming']):
                    return 'Technology'
            elif any(word in category for word in ['self-help', 'self help', 'personal development']):
                    return 'Self-Help'
            elif any(word in category for word in ['business', 'economics', 'finance', 'management']):
                    return 'Business'
            elif any(word in category for word in ['children', 'juvenile nonfiction', 'kids']):
                    return 'Children'
            elif any(word in category for word in ['poetry', 'poems']):
                    return 'Poetry'
            elif any(word in category for word in ['cooking', 'food', 'recipes']):
                    return 'Cooking'
            elif any(word in category for word in ['travel', 'tourism']):
                    return 'Travel'
            elif any(word in category for word in ['art', 'music', 'design']):
                    return 'Arts'
            elif any(word in category for word in ['religion', 'spiritual', 'philosophy']):
                    return 'Religion & Philosophy'
            else:
                return 'General'
         
        # Applying categorization 
        df['main_category']= df['main_category'].apply(categorize_book)

        # Show category distribution
        print(f"\nConsolidated to: {df['main_category'].nunique()} categories")
        print("\nCategory distribution:")
        print(df['main_category'].value_counts())

        
        return df
    

    def balance_dataset(self,df,min_samples_per_category=100 , max_sample_per_category=None):
         
         """
        Balance dataset by filtering categories and optionally limiting samples
        
        Args:
            df: Input DataFrame
            min_samples_per_category: Minimum books needed per category
            max_samples_per_category: Maximum books per category (None for no limit)
            
        Returns:
            Balanced DataFrame
        """
         
         print(f"\nBalancing dataset...")
         print(f"Minimum samples per category: {min_samples_per_category} ")
         if max_sample_per_category:
              print(f"Maximum samples per category : {max_samples_per_category}")

        #  Counting samples per category 
         category_counts=df['main_category'].value_counts()


        # Removing categories with too few samples 
         valid_categories= category_counts[category_counts >=min_samples_per_category].index
         df_balanced= df[df['main_category'].isin(valid_categories)].copy()

         removed_categories = set(category_counts.index) - set(valid_categories)
         if removed_categories:
            print(f"\nRemoved categories with < {min_samples_per_category} books:")
            for cat in removed_categories:
                print(f"  - {cat}: {category_counts[cat]} books")


         #Optionally limit maximum samples per category(undersample)
         if max_sample_per_category:
              balanced_dfs= []
              for category in valid_categories:
                   cat_df = df_balanced[df_balanced['main_category']==category]
                   if len(cat_df) > max_sample_per_category:
                        cat_df=cat_df.sample(n=max_sample_per_category,random_state=42)
                   balanced_dfs.append(cat_df)
              df_balanced=pd.concat(balanced_dfs, ignore_index=True)

        
         print(f"\nFinal dataset:")
         print(f"  Categories: {len(valid_categories)}")
         print(f"  Total books: {len(df_balanced)}")
         print("\nBooks per category:")
         print(df_balanced['main_category'].value_counts().sort_index())
        
         return df_balanced
    

    def create_text_features(self,df):
         
         """
        Create combined text features for ML models
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new text features
        """
         
         print("\n Creating text features")

         df=df.copy()

         # Cleaning individual text fields
         df['title_clean']=df['title'].apply(self.clean_text)
         df['authors_clean']=df['authors'].apply(self.clean_text)
         df['description_clean']=df['description'].apply(self.clean_text)

         # Combining  all text features
         df['combined_text'] = (
            df['title_clean'] + ' ' + 
            df['authors_clean'] + ' ' + 
            df['description_clean']
        )
         
         # Adding text length features
         df['text_length'] = df['combined_text'].str.len()
         df['word_count'] = df['combined_text'].str.split().str.len()
        
         print(f"  Average text length: {df['text_length'].mean():.0f} characters")
         print(f"  Average word count: {df['word_count'].mean():.0f} words")
        
         return df
    


    def encode_label(self,df):
         """
        Encodes category labels to numeric values
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded labels and label mapping
        """
         
         print("\n Encoding labels")
         df=df.copy()

         # Encoding categories 
         df['category_encoded'] = self.label_encoder.fit_transform(df['main_category'])

         # Create mapping for reference
         self.category_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
         
         print("\nCategory encoding:")
         for category, code in sorted(self.category_mapping.items()):
            count = (df['main_category'] == category).sum()
            print(f"  {code}: {category} ({count} books)")
        
         return df
    

    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Splits data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            Train, validation, and test DataFrames
        """

        # First splitting : train+val and test
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['category_encoded']
        )
        
        # Second splitting: train and val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val['category_encoded']
        )


        print(f"  Training set: {len(train)} ({len(train)/len(df)*100:.1f}%)")
        print(f"  Validation set: {len(val)} ({len(val)/len(df)*100:.1f}%)")
        print(f"  Test set: {len(test)} ({len(test)/len(df)*100:.1f}%)")
        
        return train, val, test
    
    def save_processed_data(self, train, val, test, prefix='processed'):
        """
        Saves processed datasets
        
        Args:
            train: Training DataFrame
            val: Validation DataFrame
            test: Test DataFrame
            prefix: Filename prefix
        """
        print("\nSaving processed data...")
        
        train_path = config.PROCESSED_DATA_DIR / f'{prefix}_train.csv'
        val_path = config.PROCESSED_DATA_DIR / f'{prefix}_val.csv'
        test_path = config.PROCESSED_DATA_DIR / f'{prefix}_test.csv'
        
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)
        
        print(f"  Training data: {train_path}")
        print(f"  Validation data: {val_path}")
        print(f"  Test data: {test_path}")
        
         # Saving label mapping (convert numpy types to native Python)
        import json
        mapping_path = config.PROCESSED_DATA_DIR / 'category_mapping.json'
    
         # Convert all values to int
        category_mapping_python = {k: int(v) for k, v in self.category_mapping.items()}
    
        with open(mapping_path, 'w') as f:
            json.dump(category_mapping_python, f, indent=2)
    
        print(f"  Category mapping: {mapping_path}")

    
    def get_preprocessing_summary(self, df):
        """
        Generates preprocessing summary report
        
        Args:
            df: Final processed DataFrame
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_books': len(df),
            'num_categories': df['main_category'].nunique(),
            'avg_rating': df['average_rating'].mean(),
            'avg_text_length': df['combined_text'].str.len().mean(),
            'categories': df['main_category'].value_counts().to_dict()
        }
        
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Total Books: {summary['total_books']}")
        print(f"Number of Categories: {summary['num_categories']}")
        print(f"Average Rating: {summary['avg_rating']:.2f}")
        print(f"Average Text Length: {summary['avg_text_length']:.0f} characters")
        
        return summary
    

def main():
        """
        Main preprocessing pipeline
        """
        print("="*50)
        print("BOOK DATA PREPROCESSING PIPELINE")
        print("="*50)
        
        # Initialize=ing preprocessor
        preprocessor = BookPreProcessor()
        
        # Step 1: Loading data
        data_file = config.RAW_DATA_DIR / 'final_raw.csv'

        
        df = preprocessor.load_data(data_file)
        if df is None:
            return
        
        # Step 2: Handling missing values
        df = preprocessor.handle_missing_values(df)
        
        # Step 3: Filtering quality books
        df = preprocessor.filter_quality_books(df)
        
        # Step 4: Cleaning categories
        df = preprocessor.clean_categories(df)
        
        # Step 5: Balancing dataset
        df = preprocessor.balance_dataset(df, min_samples_per_category=30)
        
        # Step 6: Creating text features
        df = preprocessor.create_text_features(df)
        
        # Step 7: Encoding labels
        df = preprocessor.encode_label(df)
        
        # Step 8: Splitting data
        train, val, test = preprocessor.split_data(df)
        
        # Step 9: Saving processed data
        preprocessor.save_processed_data(train, val, test)
        
        # Step 10: Generateingsummary
        preprocessor.get_preprocessing_summary(df)
        
        print("\n" + "="*50)
        print("✓ PREPROCESSING COMPLETE!")
        print("="*50)


if __name__ == '__main__':
    main()
  
         

         
         
         
    



         


      



         

         
    

    









        

        

        



      
        


