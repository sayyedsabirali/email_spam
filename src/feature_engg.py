import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Ensure the "logs" directory exists
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TfIdf to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('tfidf applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data : pd.DataFrame) -> None:
    """Save the dataframe to a CSV file."""
    try:
        data_path = os.path.join('data','3.feature_data')
        os.makedirs(data_path,exist_ok=True)
        train_data.to_csv(os.path.join(data_path,"train_feature_data.csv"),index = False)
        test_data.to_csv(os.path.join(data_path,"test_feature_data.csv"),index=False)

        logger.debug('Data saved.')
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        max_features = 50

        train_data = load_data(r'F:\email_spam\data\2.data_preprocessing\train_processed.csv')
        test_data = load_data(r'F:\email_spam\data\2.data_preprocessing\test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)
        save_data(train_df,test_df)

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
