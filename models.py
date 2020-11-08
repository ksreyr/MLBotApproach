import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from utils import Utils


class Models:

    def __init__(self):
        self.reg = {
            'NB': MultinomialNB(),
            'KNN': KNeighborsClassifier(),
            'Nn': MLPClassifier(),
            'TreeCl': tree.DecisionTreeClassifier()
        }

        self.params = {
            'NB': {
                'alpha': [.01, .04, .005]
            },
            'KNN': {
                'n_neighbors': [3, 5, 10]
            },
            'Nn': {
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'hidden_layer_sizes': [
                    (4,), (12,), (13,), (14,), (15,), (30,), (80,)
                ]
            },
            'TreeCl': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]
            }

        }

    def grid_training(self, X, y):
        best_score = 0.00999
        best_model = None
        for name, reg in self.reg.items():
            grid_reg = GridSearchCV(
                reg, self.params[name], cv=2, scoring='accuracy').fit(X, y)
            score = np.abs(grid_reg.best_score_)
            print("Score: "+str(score))
            print("Model: "+str(grid_reg.best_estimator_))
            if score > best_score:
                best_score = score
                best_model = grid_reg.best_estimator_

        print("BEST Score::: "+str(best_score))
        print("BEST Model::: "+str(best_model))
        utils = Utils()
        utils.model_export(best_model, best_score)
