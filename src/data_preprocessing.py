import logging
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
nltk.download('stopwords')
nltk.download('punkt')

# Create a logger object and set level
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# Define handler -> where log msg will output 2 type
# 1. console handler -> print in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# 2. file handler -> save log msg in the file 
log_dir="log"
os.makedirs(log_dir,exist_ok=True)
file_handler_path = os.path.join(log_dir,"data_preprocessing.log")
file_handler = logging.FileHandler(file_handler_path)
file_handler.setLevel("DEBUG")

# define formater -> stucture of log msg.
formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# connect handler to formater
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

# connnect logger to handler
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def transform_text(text):
    # sabse pehle lower mai convert karo , tokenize karo word level pr, alphanumeric rkho  bss,punc or stopword remove karo 
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df, text_column='text', target_column='target'):
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        logger.debug("Removing duplicates :")
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # text transformation to the text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column='text', target_column='target'):
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv(r'F:\email_spam\data\1_data_ingestion\train_preprocess_data.csv')
        test_data = pd.read_csv(r'F:\email_spam\data\1_data_ingestion\test_preprocess_data.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "2.data_preprocessing")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()