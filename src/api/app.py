import configparser

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from gensim.models import Word2Vec

app = FastAPI()

class InputData(BaseModel):
    text: str

class api_():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.model = joblib.load(self.config['LOG_REG']['path'])
        
        self.word2vec_model = Word2Vec.load(self.config['WORD2VEC']['model_path'])

    def text_to_vector(self, model, text):
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    def predict(self, input):
        input_data = np.array([self.text_to_vector(self.word2vec_model, input)])

        prediction = self.model.predict(input_data)
        return prediction

api_interface = api_()

@app.get("/predict/{data}")
def predict(data: str):
    prediction = api_interface.predict(data)
    return {"prediction": int(prediction)}