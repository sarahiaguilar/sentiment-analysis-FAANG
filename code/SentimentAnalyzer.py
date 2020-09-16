import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords 
english_stopwords = set(stopwords.words("english"))
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from collections import Counter
import pickle
from typing import Tuple
import scipy

class SentimentAnalyzer:
    def __init__(self,
                 input_encoder_path:str,
                 classifier_path:str):
        with open(input_encoder_path, 'rb') as f:
            self.input_encoder = pickle.load(f)
        
        with open(classifier_path, 'rb') as f:
            self.sentiment_classifier = pickle.load(f)
        
    def analyze(self,
                df: pd.DataFrame,
                text_colname: str) -> pd.DataFrame:
        
        df, X = self._preprocess_data(df, text_colname)       
        y_pred = self.sentiment_classifier.predict(X)
        
        df['Sentiment'] = y_pred
        
        df = df[['Text_clean', 'Sentiment']]
        df.rename(columns = {'Text_clean': 'Text'}, inplace = True)
        
        return df
        
    def _preprocess_data(self,
                         df:pd.DataFrame,
                         text_colname:str) -> Tuple[pd.DataFrame]:
        df['Text_clean'] = df[text_colname].map(lambda x: self._clean_text(x))
        df['Text_clean'] = df['Text_clean'].map(lambda x: self._remove_stopwords(x))
        df = df.dropna(subset = ['Text_clean'])
        df = df[(df.Text_clean != '')]
        
        bag_of_words = []
        for text in df['Text_clean']:
            tokens = nltk.word_tokenize(text)
            bag_of_words.append(Counter(tokens))
            
        X = self.input_encoder.transform(bag_of_words)
        
        return df, X
    
    def _clean_text(self,
                    text:str) -> str:
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)        
        text = re.sub('@[^\s]+', '', text) 
        text = re.sub(r'#([^\s]+)', r'\1', text)
        text = re.sub(r'[^A-Za-z]+', ' ', text) 
        text = re.sub(r'rt|fb|nflx|goog|googl|axp|aapl', '', text, flags = re.I) 
        text = re.sub(r'\b[a-zA-Z]\b', '', text) 
        text = re.sub(r' [ ]+', ' ', text) 
        text = text.lower() 
        return text
    
    def _remove_stopwords(self,
                          text:str) -> str:
        tokens = nltk.word_tokenize(text, 'english')
        filtered_tokens = [i for i in tokens if i not in english_stopwords]
        text = ' '.join(filtered_tokens)
        return text