import os
import sys
import configparser

import pandas as pd

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from preprocess import data_generator

generator = data_generator()
config = configparser.ConfigParser()
config.read("config.ini")

def test_preprocess_text():
    text = "<html>This is a <b>test</b> text 1111</html>"
    result = generator.preprocess_text(text)
    
    assert result == "test text"

def test_split():
    assert generator.split() == True

def test_save_splitted_data():
    assert generator.save_splitted_data(
        pd.read_csv(config["SPLIT_DATA"]["x_test"], index_col=0), 
        config["SPLIT_DATA"]["x_test"]
    )

