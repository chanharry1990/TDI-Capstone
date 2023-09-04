import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge

"""
Load imputed_data.csv and then train Ridge model on data.
Cross-validates with GridSearch
"""


# read data
def read_data():
    return pd.read_csv('imputed_data.csv')


# remove Country and PISA score
def create_train_model():
    # load data
    imputed_data = read_data()

    # drop Country and Label
    drop_country_pisa = ['Country', 'PISA Score']
    X = imputed_data.drop(columns=drop_country_pisa)
    y = imputed_data['PISA Score']

    # Create train test split and premodel
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    premodel = Ridge(alpha=1.0)
    premodel.fit(X_train, y_train)

    # Cross-Validate data
    ridge_grid = GridSearchCV(premodel,
                              {'alpha': np.logspace(4, 5, 20)},
                              cv=5,
                              n_jobs=-1)
    ridge_grid.fit(X_train, y_train)
    ridge_best_parameters = ridge_grid.best_params_['alpha']

    # finalize model
    model = Ridge(alpha=ridge_best_parameters)
    model.fit(X, y)
    return model
