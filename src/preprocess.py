import os
import re
import sys
import traceback
import configparser

import nltk
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from logger import Logger

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


class data_generator():
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        logger = Logger(self.config.getboolean('LOGGER', 'show'))
        self.log = logger.get_logger(__name__)

        self.project_path = os.path.join(os.getcwd(), "data")
        self.data_path = os.path.join(self.project_path, 'JEOPARDY_csv.csv')
        self.train_path = [os.path.join(self.project_path, "Train_JEOPARDY_csv_X.csv"), os.path.join(
            self.project_path, "Train_JEOPARDY_csv_y.csv")]
        self.test_path = [os.path.join(self.project_path, "Test_JEOPARDY_csv_X.csv"), os.path.join(
            self.project_path, "Test_JEOPARDY_csv_y.csv")]
        self.word2vec_model_path = os.path.join(self.project_path, 'word2vec.model')
        
        self.log.info("DataMaker is ready")

    def preprocess_text(self, text):
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        words = text.lower().split()

        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)
    
    def preprocess_label(self, text):
        valid_categories = ['BEFORE & AFTER', 'SCIENCE', 'LITERATURE', 'AMERICAN HISTORY', 'POTPOURRI']
        return valid_categories.index(text)
        
    def split(self) -> bool:
        try:
            dataset = pd.read_csv(self.data_path)
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        dataset.columns = dataset.columns.str.strip()
        valid_categories = ['BEFORE & AFTER', 'SCIENCE', 'LITERATURE', 'AMERICAN HISTORY', 'POTPOURRI']
        dataset = dataset[dataset['Category'].isin(valid_categories)]
        dataset['question_cleaned'] = dataset['Question'].apply(self.preprocess_text)
        dataset['label'] = dataset['Category'].apply(self.preprocess_label)

        # word2vec
        sentences = [word_tokenize(text) for text in dataset['question_cleaned']]
        model = Word2Vec(
            sentences=sentences, 
            vector_size=self.config.getint('WORD2VEC', 'vector_size'), 
            window=self.config.getint('WORD2VEC', 'window'), 
            min_count=self.config.getint('WORD2VEC', 'min_count'),
            epochs=self.config.getint('WORD2VEC', 'epochs'),
            sg=self.config.getboolean('WORD2VEC', 'sg')
        )
        model.save(self.word2vec_model_path)
        self.config['WORD2VEC']['model_path'] = self.word2vec_model_path
        self.log.info('Word2vec model saved')

        X_train, X_test, y_train, y_test = train_test_split(
            dataset['question_cleaned'], 
            dataset['label'],
            test_size=self.config.getfloat('DATA', 'test_ratio'),
            stratify=dataset['label'],
            random_state=42
        )

        self.save_splitted_data(X_train, self.train_path[0])
        self.save_splitted_data(y_train, self.train_path[1])
        self.save_splitted_data(X_test, self.test_path[0])
        self.save_splitted_data(y_test, self.test_path[1])
        self.config['DATA']['csv_path'] = self.data_path
        self.config["SPLIT_DATA"] = {'X_train': self.train_path[0],
                                     'y_train': self.train_path[1],
                                     'X_test': self.test_path[0],
                                     'y_test': self.test_path[1]}
        self.log.info("Train and test data is ready")
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        return os.path.isfile(self.train_path[0]) and\
            os.path.isfile(self.train_path[1]) and\
            os.path.isfile(self.test_path[0]) and \
            os.path.isfile(self.test_path[1]) and os.path.isfile(self.data_path)


    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        df = df.reset_index(drop=True)
        df.to_csv(path, index=True)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == '__main__':
    generator = data_generator()
    generator.split()