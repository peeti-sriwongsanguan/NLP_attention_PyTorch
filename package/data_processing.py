import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def remove_tags(text):
    cleaned_text = re.sub('<[^<]+?>', '', str(text))
    return cleaned_text

def load_and_preprocess_data(dataframe_path):
    df = pd.read_json(dataframe_path, lines=True)
    df['positive'] = df.overall.apply(lambda x: 1 if x >= 4 else 0)
    review_df = df[['reviewText', 'positive']]
    review_df = review_df.assign(reviewText=review_df['reviewText'].apply(remove_tags))
    return review_df

def balance_dataset(df):
    max_cnt = df.query('positive == 0').shape[0]
    return df.groupby(['positive']).apply(lambda x: x.sample(max_cnt, random_state=42)).reset_index(drop=True)

def preprocess_text(df):
    df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join([word.lower() for word in str(x).split() if word.lower() not in stop_words and word.isalnum()]))
    df = df[df['reviewText'].str.strip() != '']  # Remove empty reviews
    return df