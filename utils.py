import pandas as pd
import joblib


class Utils:

    def load_from_csv(self, path):
        return pd.read_csv(path)
 
    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X, y

    def model_export(self, clf, score):
        joblib.dump(clf, './models/'+str(score))
