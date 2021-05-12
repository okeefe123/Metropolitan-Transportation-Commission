from sklearn.model_selection import StratifiedShuffleSplit
import json
import re
import numpy as np
import pandas as pd
import spacy
from sklearn.pipeline                import Pipeline, FeatureUnion
from sklearn.svm                     import LinearSVC      # baseline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from   sklearn.ensemble              import RandomForestClassifier
from sklearn.preprocessing           import *
from sklearn.impute                  import SimpleImputer
from sklearn.compose                 import ColumnTransformer
nlp = spacy.load("en_core_web_lg")

def main():
    X_from_pipeline = pd.read_csv('ML-Modeling-Data/X_from_pipeline.csv', index_col='Unnamed: 0')
    y_from_pipeline = pd.read_csv('ML-Modeling-Data/y_from_pipeline.csv', index_col='Unnamed: 0')
    X_df, y_df, city_frequency = preprocess_pipeline(X_from_pipeline, y_from_pipeline)
    X_df.to_csv('ML-Modeling-Data/X_df_HashVectorized.csv', set_index=False)
    y_df.to_csv('ML-Modeling-Data/y_df_HashVectorized.csv', set_index=False)
    

def spacy_tokenizer(string: str) -> str:
    """Tokenize each row using spaCy and lemmatize all tokens"""
    
    doc = nlp(string)
    new_string = " ".join([token.lemma_ for token in doc if not token.is_stop])
    return new_string

def numerical_features(X_df: pd.DataFrame) -> None:
    """Extract character count, character count, and average word length as numerical features"""
    
    tokens = X_df['Context'].apply(lambda x: re.sub(r'[^\w\s]', '', spacy_tokenizer(x).strip()))
    X_df['Char Count'] = tokens.apply(lambda x: len(x))
    X_df['Word Count'] = tokens.apply(lambda x: len(x.split()))
    X_df['Avg Word Length'] = X_df['Char Count'] / X_df['Word Count']
    
def encode_cities_mean_frequency(X_df: pd.DataFrame) -> None:
    """Encode cities via mean frequency"""
    
    keys = X_df['City'].value_counts().index.values
    vals = (X_df['City'].value_counts() / len(X_df)).values
    encode_cities = dict(zip(keys, vals))
    # For production use, will need to save the frequency of cities in the case of new cities being included
    # when used.
    city_frequency = dict(zip(keys, vals * len(X_df)))
    

    X_df['City'] = X_df['City'].map(lambda x: encode_cities[x])
    
    return city_frequency
    
def encode_label(y_df: pd.DataFrame) -> None:

    encode_labels = {'max_dua'          : 0,
                     'minimum_lot_sqft' : 1,
                     'building_height'  : 2,
                     'units_per_lot'    : 3,
                     'max_far'          : 4, 
                     'none'             : 5
    }

    y_df['Attribute'] = y_df['Attribute'].map(lambda y: encode_labels[y])
    
def preprocess_pipeline(X_df, y_df):
    """
    Use the above functions to apply preprocessing/feature engineering on dataset
    
    Vectorize the lemmatized tokens such that the model can interpret them during training/testing
    
    Organize the extracted features such that the model can read it in.
    """
    
    print("Context Numerical Analysis")
    # Update X_df with number variable analysis
    numerical_features(X_df)
    
    print("Hash Vectorizing...")
    # Tokenize the Context column into a sparse matrix
    vectorizer = HashingVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
#     vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
    sparse = vectorizer.fit_transform(X_df['Context'])    
    
    print("Encoding cities...")
    #encode the "cities" feature
    city_frequency = encode_cities_mean_frequency(X_df)
    
    print("Transforming sparse matrix...")
    # transform sparse CV matrix such that each dimension is given its own column
    # drop context and join X_df with sparse (dataframe)
    X_df = X_df.join(pd.DataFrame(sparse.todense())).drop(['Context'], axis=1)
    
    print("Encoding the labels...")
    #encode the labels
    encode_label(y_df)
    
    return X_df, y_df, city_frequency

def train_test_split(X_df, y_df, frac):
    """Stratified train and test split by class to deal with any imbalance in classes"""
    
    X = np.array(X_df)
    y = np.array(y_df)
    skf = StratifiedShuffleSplit(n_splits=2, test_size=(1-frac))
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()