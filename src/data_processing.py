import pandas as pd 
import numpy as np 


def load_iris_data(datapath):
    df= pd.read_csv(datapath)
    return df

def convert_labletoIds(df):
    df = df.map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})
    return df

#convert text to nums & nums to text configs