import os
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import classification_models

model = classification_models()

def test_logistic_regression():
    assert model.logistic_regression(True) == True