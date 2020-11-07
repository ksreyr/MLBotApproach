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
    dataFrame1 = utils.load_from_csv('./in/dataAnimals.csv')
    dataFrame2 = utils.load_from_csv('./in/dataDataScience.csv')
    dataFrame3 = utils.load_from_csv('./in/dataEducation.csv')
    dataFrame4 = utils.load_from_csv('./in/dataFashion.csv')
    dataFrame5 = utils.load_from_csv('./in/dataPolitics.csv')
    frames = [dataFrame1, dataFrame2, dataFrame3, dataFrame4, dataFrame5]
    alldata = pd.concat(frames)
    alldata_data = alldata.values[:, 2]
    alldata_tarjet = alldata.values[:, 1]
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(alldata_data)
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, alldata_tarjet, test_size=0.33, random_state=42)
    # data = utils.load_from_csv('./in/felicidad.csv')
    # X, y = utils.features_target(data, ['score', 'rank', 'country'], ['score'])
    #models.grid_training(X, y)
    models.grid_training(x_train, y_train)
    print(alldata)
