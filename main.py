from utils import Utils
from models import Models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    utils = Utils()
    models = Models()
    dataFrame1 = utils.load_from_csv('./in/ropa.csv')
    frames = [dataFrame1]
    alldata = pd.concat(frames)
    alldata_data = alldata.values[:, 0]
    alldata_tarjet = alldata.values[:, 1]
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(alldata_data)
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, alldata_tarjet, test_size=0.33, random_state=42)

    models.grid_training(x_train, y_train)
    print(alldata)
