import os
import sys
import pickle
import traceback
import configparser

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from logger import Logger


class classification_models():
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        logger = Logger(self.config.getboolean('LOGGER', 'show'))
        self.log = logger.get_logger(__name__)

        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        
        self.word2vec_model = Word2Vec.load(self.config['WORD2VEC']['model_path'])
        self.X_train = np.array([self.text_to_vector(self.word2vec_model, text) for text in self.X_train['question_cleaned']])
        self.X_test = np.array([self.text_to_vector(self.word2vec_model, text) for text in self.X_test['question_cleaned']])
        self.y_train = self.y_train.squeeze()
        self.y_test = self.y_test.squeeze()
        self.log.info('X converted -> vec')
        
        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.logistic_regression_path = os.path.join(self.project_path, "logistic_regression.sav")
        

    def text_to_vector(self, model, text):
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    
    def logistic_regression(self, predict=True):
        classifier = LogisticRegression(class_weight='balanced')
        
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))

        params = {'path': self.logistic_regression_path}
        return self.save_model(classifier, self.logistic_regression_path, "LOG_REG", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

        self.log.info(f'{path} is saved')
        return os.path.isfile(path)

if __name__ == '__main__':
    model = classification_models()
    model.logistic_regression()
    