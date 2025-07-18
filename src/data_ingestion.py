import numpy as np
import pandas as pd
import logging
import os 
from sklearn.model_selection import train_test_split

# Create a logger object and set level
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# Define handler -> where log msg will output 2 type
# 1. console handler -> print in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# 2. file handler -> save log msg in the file 
log_dir="log"
os.makedirs(log_dir,exist_ok=True)
file_handler_path = os.path.join(log_dir,"data_ingestion.log")
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


def load_data(data_url: str):
    try:
        logger.debug("Data is fetching from url:")
        df = pd.read_csv(data_url)
        logger.debug("Data completly loaded from url. ")
        return df
    except FileNotFoundError as e:
        logger.error("there is no file at the given path : %s", e)
        raise
    except pd.errors.ParserError as e:
        logger.error("failed to parse the file : %s",e)
        raise
    except Exception as e:
        logger.error("Some Error raised while loading the data : %s",e)
        raise

def preprocess_data(df : pd.DataFrame):
    try:
        logger.debug("performing basic preprocessing drop and rename columns : ")
        df.drop(columns=['Unnamed: 2','Unnamed: 4','Unnamed: 3'],inplace=True)
        df.rename(columns={'v1':'target',
                           'v2':'text'},inplace=True)
        logger.debug("preprocessing Completed. ")
        return df 
    except KeyError as e:
        logger.error("column name dont exist in the DataFrame",e)
        raise
    except 	AttributeError as e:
        logger.error("df is none : %s",e)
        raise
    except ValueError as e:
        logger.error("df is none :%s",e)
        raise
    except Exception as e:
        logger.error("there are some error in the processing step : %s",e)
        raise

def save_data(train_data : pd.DataFrame,test_data:pd.DataFrame):
    try:
        data_dir='data'
        os.makedirs(data_dir,exist_ok=True)
        
        preprocess_data_path = os.path.join(data_dir,'1_data_ingestion')
        os.makedirs(preprocess_data_path, exist_ok=True)

        logger.debug("saving data into 1_prepocess_data :")

        train_data.to_csv(os.path.join(preprocess_data_path,"train_preprocess_data.csv"),index=False)
        test_data.to_csv(os.path.join(preprocess_data_path,'test_preprocess_data.csv'),index=False)
        logger.debug("Data saved completly")
    except Exception as e:
        logger.error("Error occurred while saving data : %s", e)
        raise

def main():
    try:
        data_url = "https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/refs/heads/main/experiments/spam.csv"
        test_size =  0.3
        random_state=2
        df=load_data(data_url=data_url)
        clean_df = preprocess_data(df=df)
        train_df,test_df = train_test_split(clean_df,test_size=test_size,random_state=random_state)
        save_data(train_data=train_df,test_data=test_df)
    except Exception as e:
        logger.error("An error occured in main pipeline %s",e)
        raise

if __name__ == '__main__':
    main()